"""Spectral (FFT) operators for the CLEFPE system.

The Laplacian is exact on a periodic grid; the IMEX solver uses
per-mode backward Euler for the linear diffusive/wave terms, making
the scheme unconditionally stable for those terms.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from chappie.core.config import ChappieConfig
from chappie.core.field import Grid3D


class SpectralOps:
    """Pre-computed spectral helpers: exact Laplacian + implicit solves."""

    def __init__(self, grid: Grid3D) -> None:
        self.grid = grid

    def laplacian(self, f: Array) -> Array:
        return self.grid.laplacian(f)

    def to_spectral(self, f: Array) -> Array:
        return self.grid.to_spectral(f)

    def from_spectral(self, h: Array) -> Array:
        return self.grid.from_spectral(h)

    # -- IMEX linear solves (per-mode) -------------------------------------

    def implicit_wave_update(
        self,
        phi_hat: Array,   # (n1, n2, n3//2+1) complex
        w_hat: Array,
        nl_w_hat: Array,  # nonlinear acceleration for W, in spectral space
        dt: float,
        cfg: ChappieConfig,
    ) -> tuple[Array, Array]:
        """Implicit update of the (Φ, W) system:

            ∂t Φ = W
            ∂t W = c²∇²Φ + D∇²W − γW + nl_w

        Solve (I − dt·A)·[Φ̂, Ŵ] = [Φ̂₀, Ŵ₀ + dt·nl_ŵ] per Fourier mode.

        Returns:
            (Φ̂_new, Ŵ_new) — updated spectral coefficients
        """
        c2 = cfg.clefpe.c**2
        D  = cfg.clefpe.D
        gam = cfg.clefpe.gamma
        k2 = self.grid.k2sum       # |k|² — shape broadcasted

        b1 = phi_hat
        b2 = w_hat + dt * nl_w_hat

        det = 1.0 + dt * (D * k2 + gam) + dt**2 * c2 * k2

        new_w_hat   = (b2 - dt * c2 * k2 * b1) / det
        new_phi_hat = b1  + dt * new_w_hat

        return new_phi_hat, new_w_hat

    def implicit_memory_update(
        self,
        m_hat: Array,    # spectral coefficients of one memory field
        src_hat: Array,  # spectral coefficients of the explicit source η·Φ
        dt: float,
        rho: float,
        mu: float,
    ) -> Array:
        """Backward Euler for a single memory field:

            ∂t M = ρ∇²M − μM + ηΦ

        (I − dt(ρ∇²−μ)) M̂_new = M̂ + dt·src_hat
        """
        k2 = self.grid.k2sum
        return (m_hat + dt * src_hat) / (1.0 + dt * (rho * k2 + mu))
