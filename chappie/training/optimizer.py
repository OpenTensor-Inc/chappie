"""Adam optimizer for small parameter sets (no external dependencies).

Used for the few learnable scalars (basis widths/amps, decoder bias).
Only float leaves are updated; integer arrays (e.g. token IDs) are left alone.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import tree_util


def _zeros_like(pytree):
    return tree_util.tree_map(lambda x: jnp.zeros_like(x) if hasattr(x, 'shape') else 0.0, pytree)


def adam_init(params: Any) -> tuple:
    """Initialise Adam state (first and second moments)."""
    m = _zeros_like(params)
    v = _zeros_like(params)
    return m, v


def adam_update(
    params: Any,
    grads: Any,
    m: Any,
    v: Any,
    step: int,
    lr: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Any, Any, Any]:
    """One Adam step. Returns (new_params, new_m, new_v).

    Stepsize is corrected for bias as:  lr_corrected = lr / (1 − b1^step).
    The step count is a plain Python int.
    """
    step_c = jnp.maximum(step, 1)
    m = tree_util.tree_map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, grads)
    v = tree_util.tree_map(lambda vi, gi: b2 * vi + (1 - b2) * gi**2, v, grads)
    mh = tree_util.tree_map(lambda mi: mi / (1 - b1**step_c), m)
    vh = tree_util.tree_map(lambda vi: vi / (1 - b2**step_c), v)
    new_params = tree_util.tree_map(
        lambda pi, mi, vi: pi - (lr / (1 - b1**step_c)) * mi / (jnp.sqrt(vi) + eps),
        params, mh, vh,
    )
    return new_params, m, v
