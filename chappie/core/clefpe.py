"""CLEFPE equation (spec §2/§10).

The core CLEFPE (Continuous Learning by Excitable-Field PDE) equation:

  ∂t Φ = W
  ∂t W = c²∇²Φ + D∇²W − γW + κ(V−V₀) + Σ_s χ_s M_s + F_learning − λ_R(aΦ + bΦ³)

The linear diffusive/wave terms (c²∇²Φ, D∇²W, −γW) are handled by the
IMEX implicit solver; the nonlinear coupling (HH, memory, Φ³ potential,
learning force) is explicit.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from chappie.core.config import ChappieConfig
from chappie.core.field import Grid3D
from chappie.core.state import ChappieState


def potential_force(phi: Array, a: float, b: float) -> Array:
    """U'(Φ) = aΦ + bΦ³ (the negative enters the CLEFPE RHS)."""
    return a * phi + b * phi**3


def clefpe_rhs(
    state: ChappieState,
    force: Array,
    cfg: ChappieConfig,
    lap,
    sponge: Array | None = None,
) -> tuple[Array, Array]:
    """Compute (∂t Φ, ∂t W) from the CLEFPE equation.

    The linear parts c²∇²Φ, D∇²W, −γW are included here (handled
    implicitly by the IMEX solver or explicitly by RK4).
    """
    phi = state.phi
    phi_t = state.phi_t
    chi = jnp.asarray(cfg.memory.chi, dtype=jnp.float32)
    
    # Nonlinear coupling terms (in real space)
    nl = (cfg.clefpe.kappa * (state.v - cfg.clefpe.V0)
          + jnp.tensordot(chi, state.mem, axes=1)   # Σ_s χ_s M_s
          + force
          - cfg.clefpe.lam_R * potential_force(phi, cfg.clefpe.a, cfg.clefpe.b))
    
    # Sponge absorbing boundary term (applied to velocity)
    if sponge is not None:
        nl = nl - sponge * phi_t
    
    # Linear terms: c²∇²Φ + D∇²W − γW
    dphi = phi_t
    dphi_t = (cfg.clefpe.c**2 * lap(phi)
              + cfg.clefpe.D * lap(phi_t)
              - cfg.clefpe.gamma * phi_t
              + nl)
    return dphi, dphi_t
