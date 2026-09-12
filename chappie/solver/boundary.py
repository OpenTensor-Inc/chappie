"""Boundary conditions (spec §15).

Absorbing boundaries implemented as a sponge damping layer σ(x) applied
to the velocity field and memory near the domain edges.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from chappie.core.field import Grid3D


def absorbing_mask(grid: Grid3D, width_frac: float = 0.1, strength: float = 2.0) -> Array:
    """Sponge-layer damping mask σ(x): 0 in interior, strength near walls.

    Applied as −σ(x)·∂tΦ in the CLEFPE RHS (and −σ(x)·M_s for memory).
    """
    def axis_mask(x_1d: Array) -> Array:
        d = jnp.minimum(x_1d, grid.length - x_1d)
        t = jnp.clip(d / (width_frac * grid.length / 2.0 + 1e-12), 0.0, 1.0)
        return strength * (1.0 - t) ** 2

    m1 = axis_mask(grid.axes[0])[:, None, None]
    m2 = axis_mask(grid.axes[1])[None, :, None]
    m3 = axis_mask(grid.axes[2])[None, None, :]
    return jnp.maximum(jnp.maximum(m1, m2), m3)
