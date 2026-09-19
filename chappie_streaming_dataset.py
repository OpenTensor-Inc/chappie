"""
Chappie Multilingual Streaming Dataset (Fast Version)
=====================================================
Optimized for fast streaming with:
    - Parallel dataset loading (ThreadPoolExecutor)
    - Background prefetch thread (decouples I/O from training)
    - Batched tokenization (uses Rust fast tokenizer)
    - Reduced startup latency
"""

import os
import logging
import queue
import threading
import concurrent.futures
from typing import List, Dict, Iterator, Optional

import torch
from datasets import load_dataset, interleave_datasets, disable_caching
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

# Disable global datasets caching to avoid unnecessary disk usage
disable_caching()

logger = logging.getLogger(__name__)

# ============================================================
# Cache path configuration
# ============================================================
CACHE_ROOT = os.path.expanduser("~/tmp_chappie_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_ROOT, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_ROOT, "hub")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)


# ============================================================
# Background Prefetcher
# ============================================================
class BackgroundPrefetcher:
    """
    Wraps an iterator with a background daemon thread that fills a queue
    ahead of time so the consumer never blocks on network I/O.

    The queue has bounded size (backpressure). If the consumer stops
    consuming, the worker blocks on `queue.put`, which is what we want.
    """

    def __init__(self, iterable, prefetch_size: int = 256):
        self.iterable = iterable
        self.queue: "queue.Queue" = queue.Queue(maxsize=prefetch_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            for item in self.iterable:
                if self._stop.is_set():
                    break
                self.queue.put(item)
        except Exception as e:
            self.queue.put(e)
        finally:
            self.queue.put(None)  # sentinel

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self):
        self._stop.set()


# ============================================================
# Parallel dataset loading helpers
# ============================================================
def _load_one(name: str, config: Optional[str], split: str,
              rank: int, world_size: int):
    """Load a single streaming dataset (runs inside a worker thread)."""
    kwargs = {"split": split, "streaming": True}
    if config is not None:
        kwargs["name"] = config
    ds = load_dataset(name, **kwargs)
    if world_size > 1:
        ds = ds.shard(num_shards=world_size, index=rank)
    return ds


def _load_parallel(specs: List[dict], rank: int, world_size: int,
                   max_workers: int = 16):
    """
    Load multiple streaming datasets in parallel.

    Each spec = {"name": ..., "config": str | None, "split": "train",
                 "weight": float}

    Returns:
        (datasets_list, weights_list) — only successfully loaded entries.
    """
    results = [None] * len(specs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for i, s in enumerate(specs):
            futures[ex.submit(
                _load_one,
                s["name"], s.get("config"), s.get("split", "train"),
                rank, world_size,
            )] = i
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                logger.warning(f"Failed to load {specs[idx]['name']}: {e}")

    datasets_list, weights = [], []
    for i, ds in enumerate(results):
        if ds is not None:
            datasets_list.append(ds)
            weights.append(specs[i]["weight"])
    return datasets_list, weights


# ============================================================
# Streaming Multilingual Dataset (Fast)
# ============================================================
class ChappieStreamingDataset(IterableDataset):
    """
    Weighted multilingual streaming dataset for Chappie.

    Optimizations:
        - All 23 source datasets loaded in parallel (ThreadPoolExecutor).
        - Background thread prefetches raw samples ahead of training.
        - Tokenizer called on batches of texts (much faster with fast tokenizer).
        - Interleaved iterator re-used across epochs.

    Composition (default weights):
        - FineWeb-2 (multilingual web)  : 45%
        - The Stack v2 (code)           : 20%
        - Aya Collection (instructions) : 15%
        - Project Gutenberg (books)     : 10%
        - arXiv (scientific)            : 10%
    """

    LANG_TO_FINEWEB2 = {
        "en": "eng_Latn", "fa": "fas_Arab", "ar": "ara_Arab",
        "zh": "cmn_Hani", "de": "deu_Latn", "fr": "fra_Latn",
        "es": "spa_Latn", "ja": "jpn_Jpan", "ru": "rus_Cyrl",
        "pt": "por_Latn", "ko": "kor_Hang", "it": "ita_Latn",
        "tr": "tur_Latn", "pl": "pol_Latn", "nl": "nld_Latn",
        "hi": "hin_Deva", "bn": "ben_Beng", "ur": "urd_Arab",
        "id": "ind_Latn",
    }

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        languages: List[str],
        seq_len: int = 1024,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        prefetch_size: int = 256,
        tokenize_batch_size: int = 64,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.prefetch_size = prefetch_size
        self.tokenize_batch_size = tokenize_batch_size

        # Build specs for parallel loading -----------------------------
        specs = []

        # 1. FineWeb-2 (multilingual web) — 45% weight
        n_langs = max(len(languages), 1)
        for lang in languages:
            config = self.LANG_TO_FINEWEB2.get(lang)
            if config is None:
                logger.warning(f"No FineWeb-2 config for language: {lang}")
                continue
            specs.append({
                "name": "HuggingFaceFW/fineweb-2",
                "config": config,
                "split": "train",
                "weight": 0.45 / n_langs,
            })

        # 2-5. Other sources
        specs.extend([
            {"name": "bigcode/the-stack-v2",     "config": None,
             "split": "train", "weight": 0.20},
            {"name": "CohereForAI/aya_collection","config": None,
             "split": "train", "weight": 0.15},
            {"name": "manu/project_gutenberg",    "config": "en",
             "split": "train", "weight": 0.10},
            {"name": "gfissore/arxiv-abstracts-2021", "config": None,
             "split": "train", "weight": 0.10},
        ])

        logger.info(f"Loading {len(specs)} datasets in parallel...")
        datasets_list, weights = _load_parallel(
            specs, rank=rank, world_size=world_size, max_workers=16,
        )
        if not datasets_list:
            raise RuntimeError("No streaming datasets could be loaded.")
        logger.info(f"Loaded {len(datasets_list)} datasets.")

        # Normalize weights
        total_w = sum(weights)
        self.weights = [w / total_w for w in weights]
        self.datasets_list = datasets_list

        # Text fields to look for (in priority order)
        self.text_fields = [
            "text", "content", "abstract", "inputs", "code", "raw_content"
        ]

        # Pre-build the interleaved iterator once
        self._build_interleaved()

    # ----------------------------------------------------------------
    def _build_interleaved(self):
        """Create the interleaved iterator over all sources."""
        self.interleaved = interleave_datasets(
            self.datasets_list,
            probabilities=self.weights,
            seed=self.seed,
            stopping_strategy="all_exhausted",
        ).__iter__()

    def _extract_text(self, sample: dict) -> Optional[str]:
        """Extract text from a sample using known field names."""
        for field in self.text_fields:
            if field in sample and isinstance(sample[field], str):
                if len(sample[field]) >= 50:
                    return sample[field]
        return None

    # ----------------------------------------------------------------
    def _tokenize_batch(self, texts: List[str]) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Tokenize a batch of texts at once using the fast (Rust) tokenizer,
        then yield seq_len-sized chunks.
        """
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for ids in encoded["input_ids"]:
            if len(ids) < self.seq_len:
                continue
            # Yield non-overlapping chunks
            for i in range(0, len(ids) - self.seq_len + 1, self.seq_len):
                chunk = ids[i : i + self.seq_len]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long),
                }

    # ----------------------------------------------------------------
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Infinite iterator yielding tokenized chunks.

        Pipeline:
            interleaved raw samples
              -> BackgroundPrefetcher (I/O decoupling)
              -> text buffer (up to tokenize_batch_size)
              -> batched tokenization
              -> seq_len chunks
        """
        prefetcher = BackgroundPrefetcher(
            self.interleaved, prefetch_size=self.prefetch_size,
        )

        text_buffer: List[str] = []

        try:
            for sample in prefetcher:
                text = self._extract_text(sample)
                if text is None:
                    continue
                text_buffer.append(text)

                if len(text_buffer) >= self.tokenize_batch_size:
                    yield from self._tokenize_batch(text_buffer)
                    text_buffer.clear()

            # Flush remaining buffer (in practice unreachable for infinite stream)
            if text_buffer:
                yield from self._tokenize_batch(text_buffer)
        finally:
            prefetcher.close()