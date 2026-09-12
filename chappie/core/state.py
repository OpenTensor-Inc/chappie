"""ChappieState: the state of the CLEFPE+HH+memory system.

Holds fields: phi, phi_t, mem (S fields), V, m, h, n.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from chappie.core.config import ChappieConfig
from chappie.core.field import Grid3D
from chappie.core.hh import gating_steady_state


class ChappieState(NamedTuple):
    """Full state of the coupled PDE system (§10)."""
    phi: Array      # (n1, n2, n3) — CLEFPE field Φ
    phi_t: Array    # (n1, n2, n3) — ∂t Φ = W
    mem: Array      # (S, n1, n2, n3) — memory fields
    v:  Array       # (n1, n2, n3) — HH membrane potential
    m:  Array       # (n1, n2, n3) — gating
    h:  Array       # (n1, n2, n3) — gating
    n:  Array       # (n1, n2, n3) — gating


def init_rest(grid: Grid3D, cfg: ChappieConfig) -> ChappieState:
    """Rest state: all fields at their equilibrium values."""
    shape = grid.shape
    S = cfg.memory.n_fields
    v_rest = cfg.hh.rest
    v = jnp.full(shape, v_rest, dtype=jnp.float32)
    m0, h0, n0 = gating_steady_state(v_rest)
    return ChappieState(
        phi=jnp.zeros(shape, dtype=jnp.float32),
        phi_t=jnp.zeros(shape, dtype=jnp.float32),
        mem=jnp.zeros((S, *shape), dtype=jnp.float32),
        v=v,
        m=jnp.full(shape, m0, dtype=jnp.float32),
        h=jnp.full(shape, h0, dtype=jnp.float32),
        n=jnp.full(shape, n0, dtype=jnp.float32),
    )


def reset_activity(state: ChappieState, grid: Grid3D, cfg: ChappieConfig) -> ChappieState:
    """Reset activity fields (phi, phi_t, V, m, h, n) to rest.
    
    Memory fields are kept (they accumulate information).
    """
    return state._replace(
        phi=jnp.zeros(grid.shape, dtype=jnp.float32),
        phi_t=jnp.zeros(grid.shape, dtype=jnp.float32),
        v=jnp.full(grid.shape, cfg.hh.rest, dtype=jnp.float32),
        m=jnp.full(grid.shape, gating_steady_state(cfg.hh.rest)[0], dtype=jnp.float32),
        h=jnp.full(grid.shape, gating_steady_state(cfg.hh.rest)[1], dtype=jnp.float32),
        n=jnp.full(grid.shape, gating_steady_state(cfg.hh.rest)[2], dtype=jnp.float32),
    )
