"""Field decoder (spec §6).

R[Φ](y) = ∫ Φ ψ̄_y dV  (normalized projection)
logits = R / τ + b       (sign convention fix: P ∝ exp(+R/τ), §6 footnote)
P(y) = softmax(logits)

The sign convention is consistent with the CLEFPE learning force:
  F = (1/τ)(ψ_{y*} − Σ_y P(y)ψ_y)
pushing Φ toward the target basis.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from chappie.core.config import ChappieConfig
from chappie.core.field import Grid3D
from chappie.core.state import ChappieState
from chappie.encoding.basis import BasisParams


class DecoderParams(NamedTuple):
    """Learnable readout parameters."""
    bias: Array  # (V,)


class FieldDecoder:
    """Readout: field projections → logits over vocabulary."""

    def __init__(self, basis, cfg: ChappieConfig) -> None:
        self.basis = basis
        self.cfg = cfg
        self.modalities: tuple[str, ...] = tuple(cfg.decoding.read_modalities)

    def init_params(self, vocab_size: int) -> DecoderParams:
        if self.cfg.decoding.bias:
            return DecoderParams(bias=jnp.zeros(vocab_size, dtype=jnp.float32))
        return DecoderParams(bias=jnp.zeros(vocab_size, dtype=jnp.float32))

    def logits(self, state: ChappieState, p: BasisParams,
               dec_p: DecoderParams) -> Array:
        """Compute logits (V,) from the current state."""
        R = self.basis.project(state.phi, p)  # (V,)
        if "mem" in self.modalities:
            R = R + self.basis.project(jnp.mean(state.mem, axis=0), p)
        if "v" in self.modalities:
            R = R + self.basis.project(
                state.v - self.cfg.hh.rest, p
            )
        return R / self.cfg.decoding.tau + dec_p.bias

    def logits_from_phi(self, phi: Array, p: BasisParams,
                        dec_p: DecoderParams) -> Array:
        """Logits from a bare Φ field (for the force computation)."""
        R = self.basis.project(phi, p)
        return R / self.cfg.decoding.tau + dec_p.bias

    def probs(self, state: ChappieState, p: BasisParams,
              dec_p: DecoderParams) -> Array:
        return jax.nn.softmax(self.logits(state, p, dec_p))

    def probs_from_phi(self, phi: Array, p: BasisParams,
                       dec_p: DecoderParams) -> Array:
        return jax.nn.softmax(self.logits_from_phi(phi, p, dec_p))
