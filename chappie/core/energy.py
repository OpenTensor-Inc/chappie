"""Energy functional (spec §8).

E = E_diff + E_wave + E_HH + E_mem + E_reg + E_task

Each term is computed spectrally for stability monitoring.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from chappie.core.config import ChappieConfig
from chappie.core.field import Grid3D
from chappie.core.state import ChappieState
from chappie.core.clefpe import potential_force


def energy_components(
    state: ChappieState, cfg: ChappieConfig, grid: Grid3D,
) -> dict[str, Array]:
    """Compute all energy components (excluding E_task, which is log-prob-based)."""
    phi   = state.phi
    phi_t = state.phi_t
    mem   = state.mem
    v     = state.v
    m     = state.m
    h     = state.h
    n     = state.n
    dV    = grid.dV

    # E_diff = ½ ∫ |∇Φ|² dV
    e_diff = grid.grad_energy(phi)

    # E_wave = ½ ∫ (W² + c²|∇Φ|²) dV
    e_wave = 0.5 * jnp.sum(phi_t**2) * dV + cfg.clefpe.c**2 * e_diff

    # E_HH = ½ C_m ∫ V² dV + ½ ∫ (m²+h²+n²) dV
    e_hh = (0.5 * cfg.hh.C_m * jnp.sum(v**2) * dV
            + 0.5 * jnp.sum(m**2 + h**2 + n**2) * dV)

    # E_mem = Σ_s ½ ∫ M_s² dV
    e_mem = 0.5 * jnp.sum(mem**2) * dV

    # E_reg = ∫ (a/2 Φ² + b/4 Φ⁴) dV
    e_reg = jnp.sum(
        cfg.clefpe.a / 2.0 * phi**2
        + cfg.clefpe.b / 4.0 * phi**4
    ) * dV

    return {
        "diffusion": e_diff,
        "wave":      e_wave,
        "hh":        e_hh,
        "memory":    e_mem,
        "regularization": e_reg,
    }


def total_energy(
    state: ChappieState, cfg: ChappieConfig, grid: Grid3D,
) -> Array:
    """Sum of all energy components (λ weights = 1 for PoC)."""
    comps = energy_components(state, cfg, grid)
    return sum(comps.values())
