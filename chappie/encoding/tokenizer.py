"""Tokenizers (spec §5/§16).

The tokenizer is responsible for converting text to discrete token IDs.
It is independent of the PDE — no embedding matrix is involved.
Character-level (default) or BytePairEncoding (pure Python, no deps).
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from jax import Array

# -- Character sets -------------------------------------------------------

PRINTABLE = (
    "\n !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)
PERSIAN_CHARS = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"
DEFAULT_ALPHABET = PRINTABLE + PERSIAN_CHARS


def _dedup(s: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for c in s:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return "".join(out)


# -- Character-level tokenizer -------------------------------------------

class CharTokenizer:
    """Deterministic character ↔ integer mapping.

    Vocabulary is the deduplicated alphabet, optionally truncated to
    ``vocab_size`` (if given).
    """

    def __init__(self, alphabet: str | None = None, vocab_size: int | None = None) -> None:
        chars = _dedup(alphabet if alphabet is not None else DEFAULT_ALPHABET)
        if vocab_size is not None and vocab_size > 0:
            chars = chars[:vocab_size]
        self.itos: list[str] = list(chars)
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}
        self.name = "char"

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> Array:
        ids = [self.stoi.get(c, 0) for c in text]
        return jnp.asarray(ids, dtype=jnp.int32)

    def decode(self, ids: Array) -> str:
        flat = ids.reshape(-1).tolist() if ids.ndim > 0 else [int(ids)]
        return "".join(self.itos[i] for i in flat)

    def to_list(self) -> list[str]:
        return list(self.itos)


# -- Byte Pair Encoding ---------------------------------------------------

class BytePairTokenizer:
    """Pure-Python BPE: train on a text corpus, encode/decode losslessly."""

    def __init__(self, base_vocab: list[str], merges: list[tuple[str, str]]) -> None:
        self.base_vocab = base_vocab
        self.merges = merges
        # Build full vocab: base chars + merged tokens
        vocab = list(base_vocab)
        for a, b in merges:
            vocab.append(a + b)
        self.vocab: list[str] = vocab
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(vocab)}
        self.name = "bpe"

    def __len__(self) -> int:
        return len(self.vocab)

    def to_list(self) -> list[str]:
        return list(self.vocab)

    @classmethod
    def train(cls, texts: Sequence[str], vocab_size: int,
              alphabet: str | None = None) -> "BytePairTokenizer":
        base = list(_dedup(alphabet or DEFAULT_ALPHABET))
        base_set = set(base)
        # Tokenize into base chars
        seqs = [[c for c in t if c in base_set] for t in texts]
        merges: list[tuple[str, str]] = []
        while len(base) + len(merges) < vocab_size:
            counts: dict[tuple[str, str], int] = {}
            for s in seqs:
                for pair in zip(s, s[1:]):
                    counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            best = max(counts, key=counts.get)
            if counts[best] < 2:
                break
            merges.append(best)
            # Merge in all sequences
            new_seqs = []
            for s in seqs:
                ns: list[str] = []
                i = 0
                while i < len(s):
                    if i + 1 < len(s) and s[i] == best[0] and s[i + 1] == best[1]:
                        ns.append(best[0] + best[1])
                        i += 2
                    else:
                        ns.append(s[i])
                        i += 1
                new_seqs.append(ns)
            seqs = new_seqs
        return cls(base, merges)

    def encode(self, text: str) -> Array:
        base_set = set(self.base_vocab)
        tokens = [c if c in base_set else " " for c in text]
        for a, b in self.merges:
            merged = a + b
            nt: list[str] = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                    nt.append(merged)
                    i += 2
                else:
                    nt.append(tokens[i])
                    i += 1
            tokens = nt
        return jnp.asarray(
            [self.stoi.get(t, 0) for t in tokens], dtype=jnp.int32
        )

    def decode(self, ids: Array) -> str:
        flat = ids.reshape(-1).tolist() if ids.ndim > 0 else [int(ids)]
        return "".join(self.vocab[i] for i in flat)


# -- Factory --------------------------------------------------------------

def build_tokenizer(vocab_size: int = 96, method: str = "char",
                    texts: Sequence[str] | None = None,
                    alphabet: str | None = None):
    if method == "bpe" and texts:
        return BytePairTokenizer.train(texts, vocab_size, alphabet)
    return CharTokenizer(alphabet, vocab_size)
