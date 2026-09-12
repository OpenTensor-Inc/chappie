"""Token ↔ field encoder (spec §5/§25/§26).

Token → bump source (ψ_y), initial condition (Σ_i ψ_{y_i} normalized),
context field C(x,t), and pulse-train injection.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from chappie.core.config import EncodingConfig
from chappie.core.field import Grid3D
from chappie.encoding.basis import BasisParams


class TokenFieldEncoder:
    """Maps token IDs to 3D source fields on the torus."""

    def __init__(self, basis, cfg: EncodingConfig) -> None:
        self.basis = basis
        self.cfg = cfg

    def source(self, token_id, p: BasisParams) -> Array:
        """Bump source field for a single token (raw, unnormalized)."""
        return self.bump(token_id, p)

    def bump(self, token_id, p: BasisParams) -> Array:
        return self.basis.bump(token_id, p)

    def initial_condition(self, ids: Array, p: BasisParams) -> Array:
        """Φ(x,0) = Σ_i ψ_{y_i}(x), then normalized (L∞ to 1)."""
        def _sum_bumps(t):
            return jax.vmap(lambda tid: self.bump(tid, p))(ids[t:])[0]
        # vmap over all tokens in the sequence and sum
        bumps = jax.vmap(lambda tid: self.bump(tid, p))(ids)  # (T, N1, N2, N3)
        phi0 = jnp.sum(bumps, axis=0)
        if self.cfg.init_mode == "sum_normalized":
            max_val = jnp.max(jnp.abs(phi0)) + 1e-12
            phi0 = phi0 / max_val
        return phi0

    def context_field(self, phi: Array, ids: Array, p: BasisParams,
                      grid: Grid3D) -> Array:
        """C(x,t) = Σ_i w_i(t) ψ_{x_i}(x), weights from overlap dynamics."""
        # Compute per-token overlaps
        R = self.basis.project(phi, p)  # (V,)
        w = jax.nn.softmax(R / self.cfg.sigma)  # overlap-based weights
        def _weighted_bump(tok_id):
            return w[tok_id] * self.basis.bump(tok_id, p)
        C = jax.vmap(_weighted_bump)(ids)
        return jnp.sum(C, axis=0)
