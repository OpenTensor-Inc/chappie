"""Wave equation terms (spec §10).

∂t Φ = W,  ∂t W ⊇ c²∇²Φ + D∇²W (linear wave part).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def wave_terms(phi: Array, phi_t: Array, c: float, D: float, lap) -> Array:
    """Compute c²∇²Φ + D∇²W."""
    return c**2 * lap(phi) + D * lap(phi_t)
