"""Grid3D: collocation points, wavenumbers and spectral transforms on a 3D torus.

The CLEFPE/HH system lives on Ω = [0, L)³.  The spectral Laplacian is *exact*
(diagonal in Fourier space) for periodic boundary conditions; the
finite-difference path (with other boundary conditions) lives in
``chappie.solver.finite_difference``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from chappie.core.config import GridConfig


class Grid3D:
    """Regular grid on the 3D torus with FFT helpers."""

    def __init__(self, cfg: GridConfig) -> None:
        self.n1, self.n2, self.n3 = cfg.n1, cfg.n2, cfg.n3
        self.length = float(cfg.length)
        self.boundary = cfg.boundary
        if cfg.dtype == "float32":
            self.dtype = jnp.float32
        elif cfg.dtype == "float64":
            self.dtype = jnp.float64
        else:
            self.dtype = jnp.bfloat16
        self.dx = self.length / self.n1
        self.dV = self.dx**3
        axes = tuple(
            jnp.linspace(0.0, self.length, n, endpoint=False, dtype=self.dtype)
            for n in (self.n1, self.n2, self.n3)
        )
        self.axes = axes
        x1, x2, x3 = jnp.meshgrid(*axes, indexing="ij")
        self.x1, self.x2, self.x3 = x1, x2, x3
        # wavenumbers in radians per unit length; axis 3 uses the rfft half grid
        k1 = jnp.fft.fftfreq(self.n1) * (2.0 * jnp.pi * self.n1 / self.length)
        k2 = jnp.fft.fftfreq(self.n2) * (2.0 * jnp.pi * self.n2 / self.length)
        k3 = jnp.fft.rfftfreq(self.n3) * (2.0 * jnp.pi * self.n3 / self.length)
        self.k1 = k1[:, None, None]
        self.k2 = k2[None, :, None]
        self.k3 = k3[None, None, :]
        self.k2sum = self.k1**2 + self.k2**2 + self.k3**2  # (n1, n2, n3//2+1)
        self.shape = (self.n1, self.n2, self.n3)

    # -- transforms --------------------------------------------------------

    def to_spectral(self, f: Array) -> Array:
        return jnp.fft.rfftn(f)

    def from_spectral(self, h: Array) -> Array:
        return jnp.fft.irfftn(h, s=self.shape)

    def laplacian(self, f: Array) -> Array:
        """Exact (spectral, periodic) Laplacian of a real field."""
        return self.from_spectral(-self.k2sum * self.to_spectral(f))

    # -- energy helpers ----------------------------------------------------

    def grad_energy(self, f: Array) -> Array:
        """E_diff = ½ ∫Ω |∇f|² dV  (spectral gradients, exact)."""
        h = self.to_spectral(f)
        gx = self.from_spectral(1j * self.k1 * h)
        gy = self.from_spectral(1j * self.k2 * h)
        gz = self.from_spectral(1j * self.k3 * h)
        return 0.5 * jnp.sum(gx**2 + gy**2 + gz**2) * self.dV

    def l2_energy(self, f: Array) -> Array:
        """½ ∫Ω f² dV."""
        return 0.5 * jnp.sum(f**2) * self.dV

    def l2_norm(self, f: Array) -> Array:
        return jnp.sqrt(jnp.mean(f**2))
