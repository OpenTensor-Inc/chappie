"""Training datasets (spec §17).

No heavy datasets needed.  Synthetic curriculum (levels 1–6) provides
increasingly structured text; a TextDataset wrapper reads a plain text file.
"""

from __future__ import annotations

import random as _pyrandom
from pathlib import Path

import jax.numpy as jnp
from jax import Array

from chappie.core.config import DataConfig


# -- Synthetic curriculum (no external data required) ---------------------

class SyntheticCurriculum:
    """Generate sequences of increasing complexity (levels 1–6)."""

    def __init__(self, tokenizer, cfg: DataConfig) -> None:
        self.tok = tokenizer
        self.cfg = cfg
        safe = [c for c in "abcdefghijklmnopqrstuvwxyz" if c in tokenizer.stoi]
        self._letters = safe if safe else [tokenizer.itos[i] for i in range(min(26, len(tokenizer)))]
        self._digits = [c for c in "0123456789" if c in tokenizer.stoi]
        if not self._digits:
            self._digits = self._letters[:10] if len(self._letters) >= 10 else self._letters

    def gen_text(self, level: int, seq_len: int, rng: _pyrandom.Random) -> str:
        L = self._letters
        D = self._digits
        if level <= 1:
            # Uniform random chars — maximum entropy (NLL ≈ log V)
            return "".join(rng.choice(L) for _ in range(seq_len))
        if level == 2:
            # Bigram Markov: a→b, b→a, c→d, … (predictable next char)
            pairs = [(L[i], L[i+1]) for i in range(0, len(L)-1, 2)]
            if not pairs:
                pairs = [(L[0], L[1])] if len(L) > 1 else [(L[0], L[0])]
            pair = rng.choice(pairs)
            return "".join(rng.choice(pair) for _ in range(seq_len))
        if level == 3:
            # Repeated words: "cat dog cat dog …"
            words = ["".join(rng.sample(L, min(3, len(L)))) for _ in range(4)]
            out = []
            for _ in range(seq_len):
                out.append(" ")
                out.append(rng.choice(words))
            return "".join(out)[:seq_len]
        if level == 4:
            # Cyclic pattern: "abc abc abc …"
            pat_len = min(4, len(L))
            pat = "".join(rng.sample(L, pat_len))
            return (pat * ((seq_len // pat_len) + 1))[:seq_len]
        if level == 5:
            # Variable binding: "x5 x5 x5 … y3 y3 y3 …"
            c1, c2 = rng.sample(L, 2)
            d1, d2 = rng.sample(D, 2)
            half = seq_len // 2
            return (f"{c1}{d1} " * (half // 2) + f"{c2}{d2} " * (half // 2))[:seq_len]
        # level 6: counting pairs "a1 b2 c3 …"
        cycled = list(zip(L, D))
        rng.shuffle(cycled)
        out = ""
        for c, d in cycled:
            out += f"{c}{d} "
            if len(out) >= seq_len:
                break
        return out[:seq_len]

    def sample_batch(self, batch_size: int, seq_len: int, level: int,
                     key: int | None = None) -> Array:
        rng = _pyrandom.Random(key)
        seqs = [self.gen_text(level, seq_len, rng) for _ in range(batch_size)]
        return jnp.stack([self.tok.encode(s)[:seq_len] for s in seqs])


# -- Text file dataset ----------------------------------------------------

class TextDataset:
    """Read a plain text file and split into fixed-length windows."""

    def __init__(self, path: str, tokenizer, seq_len: int) -> None:
        self.seq_len = seq_len
        text = Path(path).read_text(encoding="utf-8")
        self.ids = tokenizer.encode(text)

    def sample_batch(self, batch_size: int, key: int | None = None) -> Array:
        rng = _pyrandom.Random(key)
        N = len(self.ids)
        if N < self.seq_len + 1:
            raise ValueError(f"Text too short ({N} tokens) for seq_len={self.seq_len}")
        offsets = [rng.randint(0, max(1, N - self.seq_len - 1)) for _ in range(batch_size)]
        return jnp.stack([self.ids[o:o + self.seq_len] for o in offsets])
