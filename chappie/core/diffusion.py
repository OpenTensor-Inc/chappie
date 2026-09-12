"""Diffusion term helper.

∂t Φ_d = D ∇²Φ (standalone for ablation tests, §29).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def diffusion_rhs(field: Array, D: float, lap) -> Array:
    """Compute D∇²field."""
    return D * lap(field)
