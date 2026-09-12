"""Loss functions (spec §7/§17)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def nll(logits: Array, target: Array) -> Array:
    """Average negative log-likelihood of a target token."""
    return -jax.nn.log_softmax(logits)[target]


def accuracy(logits: Array, target: Array) -> Array:
    """Next-token prediction accuracy (scalar 0 or 1)."""
    return (jnp.argmax(logits) == target).astype(jnp.float32)


def perplexity(logits: Array, target: Array) -> Array:
    """Perplexity = exp(NLL)."""
    return jnp.exp(nll(logits, target))
