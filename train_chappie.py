"""
Chappie-aleph-v1 Trainer
========================
Staged trainer with fast streaming multilingual datasets and cache management.

Usage:
    python train_chappie.py \
        --steps 200000 \
        --batch_size 4 \
        --seq_len 1024 \
        --device cuda
"""

import os
import argparse
import logging
import time
from pathlib import Path

# --- HuggingFace fast download & tokenizer parallelism ---
# These must be set BEFORE importing transformers/datasets.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import ChappieConfig, ChappieModel
from chappie_tokenizer import load_standard_tokenizer
from chappie_streaming_dataset import ChappieStreamingDataset
from chappie_cache_manager import cleanup_cache, full_cleanup, report_cache_size

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Phase Scheduler
# ============================================================
class PhaseScheduler:
    """Determines current training phase based on step count."""

    def __init__(self, boundaries: list):
        self.boundaries = boundaries

    def current_phase(self, step: int) -> int:
        for i, b in enumerate(self.boundaries):
            if step < b:
                return i
        return len(self.boundaries)


# ============================================================
# Trainer
# ============================================================
class ChappieTrainer:
    """Main trainer for Chappie-aleph-v1."""

    def __init__(
        self,
        model: ChappieModel,
        cfg: ChappieConfig,
        device: torch.device,
        lr: float = 3e-4,
        phase_boundaries: list = None,
        use_pcgrad: bool = True,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.use_pcgrad = use_pcgrad

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        self.analyzer_optimizer = torch.optim.Adam(
            model.analyzer.parameters(), lr=1e-4
        )
        self.rss_optimizer = torch.optim.Adam(
            model.rss.parameters(), lr=1e-4
        )

        self.loss_weights = nn.Parameter(torch.zeros(3))
        self.loss_optimizer = torch.optim.Adam([self.loss_weights], lr=1e-3)

        self.phase_scheduler = PhaseScheduler(
            phase_boundaries or [2000, 5000, 8000, 12000]
        )
        self.step = 0
        self.prev_losses = {}
        self.prev_grad_norm = 0.0

    def _apply_phase(self, phase: int):
        """Freeze/unfreeze parameters based on current phase."""
        if phase == 0:
            for p in self.model.analyzer.parameters():
                p.requires_grad = False
            for p in self.model.rss.parameters():
                p.requires_grad = False
            for p in self.model.diffusion_head.parameters():
                p.requires_grad = False
            for p in self.model.gan_head.parameters():
                p.requires_grad = False
        elif phase == 1:
            for p in self.model.analyzer.parameters():
                p.requires_grad = True
        elif phase == 2:
            for p in self.model.diffusion_head.parameters():
                p.requires_grad = True
            for p in self.model.gan_head.parameters():
                p.requires_grad = True
        elif phase >= 4:
            for p in self.model.rss.parameters():
                p.requires_grad = True

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()
        phase = self.phase_scheduler.current_phase(self.step)
        self._apply_phase(phase)

        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        out = self.model(input_ids, labels=labels)

        loss_dict = {"lm": out["lm_loss"]}
        if "diffusion_loss" in out:
            loss_dict["diffusion"] = out["diffusion_loss"]
        if "gan_score" in out:
            fake = out["gan_score"]
            real = fake.detach() + torch.randn_like(fake) * 0.1
            loss_dict["gan"] = F.binary_cross_entropy_with_logits(
                fake - real, torch.ones_like(fake)
            )

        loss_values = list(loss_dict.values())
        if len(loss_values) >= 2:
            weighted = self.loss_weights[0] * loss_values[0]
            weighted = weighted + self.loss_weights[1] * loss_values[1]
            if len(loss_values) > 2:
                weighted = weighted + self.loss_weights[2] * loss_values[2]
        else:
            weighted = loss_values[0]

        self.optimizer.zero_grad(set_to_none=True)
        weighted.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        ).item()
        self.optimizer.step()

        self.prev_losses = {k: v.item() for k, v in loss_dict.items()}
        self.prev_grad_norm = grad_norm
        self.step += 1

        return {
            "step": self.step,
            "phase": phase,
            "grad_norm": grad_norm,
            **{f"loss/{k}": v.item() for k, v in loss_dict.items()},
        }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--tokenizer", default="xlm-roberta-base")
    parser.add_argument("--out_dir", default="./checkpoints_chappie")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--cache_cleanup_every", type=int, default=500)
    parser.add_argument("--max_cache_gb", type=float, default=30.0)
    # Streaming tuning knobs
    parser.add_argument("--prefetch_size", type=int, default=256)
    parser.add_argument("--tokenize_batch_size", type=int, default=64)
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="List of ISO 639-1 language codes. Default: 19 languages.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- Tokenizer first (we need its max length for seq_len validation) ----
    tokenizer = load_standard_tokenizer(args.tokenizer)

    # Guard: seq_len must not exceed tokenizer's max length.
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max is not None and args.seq_len > tok_max:
        logger.warning(
            f"seq_len={args.seq_len} > tokenizer max length={tok_max}. "
            f"Clamping seq_len to {tok_max}."
        )
        args.seq_len = tok_max

    # ---- Config ----
    cfg = ChappieConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_layers=8,
        max_seq_len=args.seq_len,
    )

    # ---- Languages ----
    default_languages = [
        "en", "fa", "ar", "zh", "de", "fr", "es", "ja", "ru",
        "pt", "ko", "it", "tr", "pl", "nl", "hi", "bn", "ur", "id",
    ]
    languages = args.languages if args.languages else default_languages
    logger.info(f"Languages: {languages}")

    # ---- Dataset ----
    logger.info("Building streaming dataset (parallel load)...")
    t_dataset_start = time.time()
    dataset = ChappieStreamingDataset(
        tokenizer=tokenizer,
        languages=languages,
        seq_len=args.seq_len,
        world_size=1,
        rank=0,
        prefetch_size=args.prefetch_size,
        tokenize_batch_size=args.tokenize_batch_size,
    )
    logger.info(f"Dataset ready in {time.time() - t_dataset_start:.1f}s")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,           # streaming dataset handles its own threading
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    model = ChappieModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {n_params / 1e6:.2f}M")
    logger.info(f"Device: {device}")

    # ---- Trainer ----
    trainer = ChappieTrainer(model, cfg, device, lr=args.lr)

    # ---- Output dir ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----
    data_iter = iter(loader)
    t0 = time.time()

    for step in range(args.steps):
        # Fetch batch (blocking here is fine — prefetcher keeps it fast)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        metrics = trainer.train_step(batch)

        # Cache cleanup (conservative, non-destructive)
        if step > 0 and step % args.cache_cleanup_every == 0:
            cleanup_cache(
                step,
                every_n=args.cache_cleanup_every,
                max_size_gb=args.max_cache_gb,
            )

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / max(elapsed, 1e-6)
            msg = (
                f"[step {step:>6}] "
                f"phase={metrics['phase']} "
                f"grad={metrics['grad_norm']:.3f} "
                f"lm={metrics.get('loss/lm', 0):.4f}"
            )
            if "loss/diffusion" in metrics:
                msg += f" diff={metrics['loss/diffusion']:.4f}"
            if "loss/gan" in metrics:
                msg += f" gan={metrics['loss/gan']:.4f}"
            msg += f" | {elapsed:.1f}s | {steps_per_sec:.2f} steps/s"
            logger.info(msg)

        # Checkpointing
        if step > 0 and step % args.save_every == 0:
            ckpt_path = out_dir / f"chappie_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")
            report_cache_size()

    # ---- Final cleanup ----
    full_cleanup()
    logger.info("Training complete. Cache cleared.")


if __name__ == "__main__":
    main()