"""Token sampling (spec §6).

Supports argmax, temperature scaling, top-k truncation, and nucleus (top-p)
sampling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def sample(
    logits: Array,
    key: Array,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Array:
    """Sample one token from a logit distribution.

    Args:
        logits:    (V,) — raw logits
        key:       PRNG key
        temperature: softmax temperature
        top_k:     if > 0, keep only top-k logits
        top_p:     nucleus sampling threshold (1.0 = disabled)

    Returns:
        sampled token id (scalar int32 array)
    """
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    V = logits.shape[-1]

    # Top-k truncation
    if top_k > 0:
        k = min(int(top_k), V)
        vals, _ = jax.lax.top_k(logits, k)
        threshold = vals[-1]
        logits = jnp.where(logits < threshold, -jnp.inf, logits)

    # Nucleus (top-p) sampling
    if top_p < 1.0:
        idx_sorted = jnp.argsort(logits, descending=True)
        sorted_logits = logits[idx_sorted]
        probs = jax.nn.softmax(sorted_logits)
        cum = jnp.cumsum(probs, axis=0)
        keep = (cum - probs) <= top_p
        # Always keep at least one token (the top)
        keep = keep.at[0].set(True)
        sorted_logits = jnp.where(keep, sorted_logits, -jnp.inf)
        sampled_sorted = jax.random.categorical(key, sorted_logits)
        return idx_sorted[sampled_sorted]

    # Default categorical sampling
    return jax.random.categorical(key, logits)
