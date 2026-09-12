"""Finite-difference Laplacian for 3D fields (spec §12/§15).

Supports periodic (via `jnp.roll`), Neumann (pad edge), and Dirichlet
(zero padding) boundary conditions.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def laplacian_fd(f: Array, dx: float, bc: str = "periodic") -> Array:
    """7-point 3D Laplacian of field f.

    Args:
        f:  (n1, n2, n3)
        dx: uniform grid spacing (assumed equal in all dimensions)
        bc: boundary condition — periodic | neumann | dirichlet

    Returns:
        ∇²f  (same shape as f)
    """
    dx2 = dx * dx
    if bc in ("periodic", "absorbing"):
        return (
            jnp.roll(f, 1, 0) + jnp.roll(f, -1, 0)
            + jnp.roll(f, 1, 1) + jnp.roll(f, -1, 1)
            + jnp.roll(f, 1, 2) + jnp.roll(f, -1, 2)
            - 6.0 * f
        ) / dx2
    if bc == "neumann":
        fe = jnp.pad(f, ((1, 1), (1, 1), (1, 1)), mode="edge")
        return (
            fe[2:, 1:-1, 1:-1] + fe[:-2, 1:-1, 1:-1]
            + fe[1:-1, 2:, 1:-1] + fe[1:-1, :-2, 1:-1]
            + fe[1:-1, 1:-1, 2:] + fe[1:-1, 1:-1, :-2]
            - 6.0 * f
        ) / dx2
    if bc == "dirichlet":
        fe = jnp.pad(f, ((1, 1), (1, 1), (1, 1)), mode="constant")
        return (
            fe[2:, 1:-1, 1:-1] + fe[:-2, 1:-1, 1:-1]
            + fe[1:-1, 2:, 1:-1] + fe[1:-1, :-2, 1:-1]
            + fe[1:-1, 1:-1, 2:] + fe[1:-1, 1:-1, :-2]
            - 6.0 * f
        ) / dx2
    raise ValueError(f"Unknown boundary condition: {bc!r}")
