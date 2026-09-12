"""Token-to-field basis functions (spec §5).

Each token is mapped to a spatial excitation ψ_y(x) on the torus Ω.
The basis is swappable (Gaussian, Fourier, Wavelet) via config.
All use bump + projection-normalized reads, no dense embedding matrix.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from chappie.core.field import Grid3D


# -- Params (learnable, differentiable) -----------------------------------

class BasisParams(NamedTuple):
    """Learnable parameters for a basis set."""
    widths: Array   # (V,)  — scale/width per token
    amps: Array     # (V,)  — amplitude per token


def _wrap(delta: Array, L: float) -> Array:
    """Wrap displacement onto torus: δ → δ − L·round(δ/L)."""
    return delta - L * jnp.round(delta / L)


# -- Gaussian basis -------------------------------------------------------

class GaussianBasis:
    """RBF bumps centred on a golden-spiral layout (no learned centres)."""
    name = "gaussian"

    def __init__(self, grid: Grid3D, n_tokens: int, key,
                 sigma: float = 0.4, amp: float = 1.0) -> None:
        self.grid = grid
        self.n_tokens = n_tokens
        # Golden-spiral low-discrepancy layout for centres
        phi = (1.0 + jnp.sqrt(5.0)) / 2.0
        idx = jnp.arange(n_tokens, dtype=jnp.float32)
        frac = idx / n_tokens
        x1 = frac * grid.length
        x2 = (frac * phi) % 1.0 * grid.length
        x3 = (frac * phi**2) % 1.0 * grid.length
        self.centers: Array = jnp.stack([x1, x2, x3], axis=-1)  # (V, 3) fixed
        self._sigma0 = sigma
        self._amp0 = amp

    def init_params(self) -> BasisParams:
        V = self.n_tokens
        return BasisParams(
            widths=jnp.full((V,), self._sigma0, dtype=jnp.float32),
            amps=jnp.full((V,), self._amp0, dtype=jnp.float32),
        )

    def num_params(self) -> int:
        return 2 * self.n_tokens

    def bump(self, token_id, p: BasisParams) -> Array:
        """Raw (unnormalized) Gaussian bump at token_id."""
        g = self.grid
        cx = self.centers[token_id, 0]
        cy = self.centers[token_id, 1]
        cz = self.centers[token_id, 2]
        w = p.widths[token_id]
        a = p.amps[token_id]
        d1 = _wrap(g.x1 - cx, g.length)
        d2 = _wrap(g.x2 - cy, g.length)
        d3 = _wrap(g.x3 - cz, g.length)
        r2 = d1**2 + d2**2 + d3**2
        return a * jnp.exp(-0.5 * r2 / (w**2 + 1e-12))

    def bump_norm(self, token_id, p: BasisParams) -> Array:
        """L2-normalized bump ψ̄ = ψ / ‖ψ‖."""
        psi = self.bump(token_id, p)
        norm = jnp.sqrt(jnp.sum(psi**2) * self.grid.dV + 1e-12)
        return psi / norm

    def project(self, field: Array, p: BasisParams) -> Array:
        """Project field onto all token bumps: R_y = ⟨field, ψ̄_y⟩."""
        def _proj(tok_id):
            psi = self.bump_norm(tok_id, p)
            return jnp.sum(field * psi) * self.grid.dV
        return jax.vmap(_proj)(jnp.arange(self.n_tokens))

    def expectation_field(self, probs: Array, p: BasisParams) -> Array:
        """Σ_y P(y) ψ̄_y — weighted sum of normalized bumps."""
        def _bump(tok_id):
            return self.bump_norm(tok_id, p)
        bumps = jax.vmap(_bump)(jnp.arange(self.n_tokens))  # (V, N1, N2, N3)
        return jnp.tensordot(probs, bumps, axes=1)


# -- Fourier basis --------------------------------------------------------

class FourierBasis:
    """Cosine modes: token y → low-frequency Fourier mode."""
    name = "fourier"

    def __init__(self, grid: Grid3D, n_tokens: int, key,
                 n_modes: int = 48) -> None:
        self.grid = grid
        self.n_tokens = n_tokens
        k2_flat = grid.k2sum.ravel()
        n_modes = min(n_modes, k2_flat.shape[0])
        mode_idx = jnp.argsort(k2_flat)[:n_modes]
        self.mode_idx: Array = mode_idx
        k1f, k2f, k3f = grid.k1.ravel(), grid.k2.ravel(), grid.k3.ravel()
        self.kvecs: Array = jnp.stack(
            [k1f[mode_idx], k2f[mode_idx], k3f[mode_idx]], axis=-1
        )  # (n_modes, 3)
        self.n_modes = n_modes

    def init_params(self) -> BasisParams:
        V = self.n_tokens
        return BasisParams(
            widths=jnp.zeros(V, dtype=jnp.float32),
            amps=jnp.full(V, 1.0, dtype=jnp.float32),
        )

    def num_params(self) -> int:
        return self.n_tokens

    def bump(self, token_id, p: BasisParams) -> Array:
        k = self.kvecs[token_id % self.n_modes]
        phase = k[0] * self.grid.x1 + k[1] * self.grid.x2 + k[2] * self.grid.x3
        return p.amps[token_id] * jnp.cos(phase)

    def bump_norm(self, token_id, p: BasisParams) -> Array:
        psi = self.bump(token_id, p)
        norm = jnp.sqrt(jnp.sum(psi**2) * self.grid.dV + 1e-12)
        return psi / norm

    def project(self, field: Array, p: BasisParams) -> Array:
        def _proj(tok_id):
            psi = self.bump_norm(tok_id, p)
            return jnp.sum(field * psi) * self.grid.dV
        return jax.vmap(_proj)(jnp.arange(self.n_tokens))

    def expectation_field(self, probs: Array, p: BasisParams) -> Array:
        def _bump(tok_id):
            return self.bump_norm(tok_id, p)
        bumps = jax.vmap(_bump)(jnp.arange(self.n_tokens))
        return jnp.tensordot(probs, bumps, axes=1)


# -- Wavelet (Mexican-hat) basis ------------------------------------------

class WaveletBasis:
    """Mexican-hat (Ricker) bumps: (1 − r²/σ²) exp(−r²/2σ²)."""
    name = "wavelet"

    def __init__(self, grid: Grid3D, n_tokens: int, key,
                 sigma: float = 0.4, amp: float = 1.0) -> None:
        self.grid = grid
        self.n_tokens = n_tokens
        # Reuse golden-spiral centres
        phi = (1.0 + jnp.sqrt(5.0)) / 2.0
        idx = jnp.arange(n_tokens, dtype=jnp.float32)
        frac = idx / n_tokens
        x1 = frac * grid.length
        x2 = (frac * phi) % 1.0 * grid.length
        x3 = (frac * phi**2) % 1.0 * grid.length
        self.centers: Array = jnp.stack([x1, x2, x3], axis=-1)
        self._sigma0 = sigma
        self._amp0 = amp

    def init_params(self) -> BasisParams:
        V = self.n_tokens
        return BasisParams(
            widths=jnp.full((V,), self._sigma0, dtype=jnp.float32),
            amps=jnp.full((V,), self._amp0, dtype=jnp.float32),
        )

    def num_params(self) -> int:
        return 2 * self.n_tokens

    def bump(self, token_id, p: BasisParams) -> Array:
        g = self.grid
        cx, cy, cz = self.centers[token_id]
        w = p.widths[token_id]
        a = p.amps[token_id]
        r2 = (_wrap(g.x1 - cx, g.length)**2
              + _wrap(g.x2 - cy, g.length)**2
              + _wrap(g.x3 - cz, g.length)**2)
        w2 = w**2 + 1e-12
        return a * (1.0 - r2 / w2) * jnp.exp(-0.5 * r2 / w2)

    def bump_norm(self, token_id, p: BasisParams) -> Array:
        psi = self.bump(token_id, p)
        norm = jnp.sqrt(jnp.sum(psi**2) * self.grid.dV + 1e-12)
        return psi / norm

    def project(self, field: Array, p: BasisParams) -> Array:
        def _proj(tok_id):
            psi = self.bump_norm(tok_id, p)
            return jnp.sum(field * psi) * self.grid.dV
        return jax.vmap(_proj)(jnp.arange(self.n_tokens))

    def expectation_field(self, probs: Array, p: BasisParams) -> Array:
        def _bump(tok_id):
            return self.bump_norm(tok_id, p)
        bumps = jax.vmap(_bump)(jnp.arange(self.n_tokens))
        return jnp.tensordot(probs, bumps, axes=1)


# -- Factory --------------------------------------------------------------

BASIS_REGISTRY = {
    "gaussian": GaussianBasis,
    "fourier": FourierBasis,
    "wavelet": WaveletBasis,
}


def build_basis(name: str, grid: Grid3D, n_tokens: int, key, **kw):
    cls = BASIS_REGISTRY[name]
    return cls(grid, n_tokens, key, **kw)
