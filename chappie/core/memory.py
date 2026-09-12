"""Memory field dynamics (spec §3).

Each memory field evolves as:
  ∂t M_s = ρ_s ∇²M_s − μ_s M_s + η_s Φ

S fields, short-term (high ρ, high μ) and long-term (low ρ, low μ).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def memory_rhs(
    mem: Array, phi: Array, lap, cfg,
) -> Array:
    """Compute ∂t M_s for all S memory fields.

    Args:
        mem: (S, n1, n2, n3)
        phi: (n1, n2, n3) — CLEFPE field
        lap: callable Laplacian operator
        cfg: MemoryConfig

    Returns:
        (S, n1, n2, n3) — time derivatives of memory fields
    """
    rho = jnp.asarray(cfg.rho, dtype=jnp.float32)[:, None, None, None]
    mu  = jnp.asarray(cfg.mu,  dtype=jnp.float32)[:, None, None, None]
    eta = jnp.asarray(cfg.eta, dtype=jnp.float32)[:, None, None, None]
    # laplacian for each field (vmap over the S dimension)
    lap_mem = jax.vmap(lap)(mem)   # (S, n1, n2, n3)
    return rho * lap_mem - mu * mem + eta * phi[None, ...]
