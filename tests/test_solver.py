"""Tests for chappie.solver — spectral, finite-difference, integrators."""

import jax
import jax.numpy as jnp
import pytest

from chappie.core.config import GridConfig, SolverConfig, MemoryConfig, HHConfig, ChappieConfig, default_config
from chappie.core.field import Grid3D
from chappie.core.state import init_rest
from chappie.solver.boundary import absorbing_mask
from chappie.solver.finite_difference import laplacian_fd
from chappie.solver.spectral import SpectralOps
from chappie.solver.integrators import (
    evolve, rk4_step, imex_step, suggest_dt, coupled_rhs,
)


class TestBoundary:
    def test_absorbing_mask_shape(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        mask = absorbing_mask(grid)
        assert mask.shape == (8, 8, 8)

    def test_absorbing_mask_zero_center(self):
        grid = Grid3D(GridConfig(n1=16, n2=16, n3=16))
        mask = absorbing_mask(grid, width_frac=0.1)
        # Center should be near zero
        assert float(mask[8, 8, 8]) < 0.5


class TestFiniteDifference:
    def test_laplacian_periodic_constant(self):
        f = jnp.ones((8, 8, 8))
        lap = laplacian_fd(f, 0.125, bc="periodic")
        assert jnp.allclose(lap, 0.0, atol=1e-6)

    def test_laplacian_periodic_sine(self):
        N = 16
        x = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
        f = jnp.cos(x[:, None, None]) * jnp.ones((N, N, N))
        dx = 2 * jnp.pi / N
        lap = laplacian_fd(f, dx, bc="periodic")
        # ∇²cos(x) = -cos(x)
        assert jnp.allclose(lap, -f, atol=0.2)

    def test_laplacian_neumann(self):
        f = jnp.ones((8, 8, 8))
        lap = laplacian_fd(f, 0.125, bc="neumann")
        assert jnp.allclose(lap, 0.0, atol=1e-5)


class TestSpectral:
    def test_laplacian_spectral(self):
        grid = Grid3D(GridConfig(n1=16, n2=16, n3=16, length=2*jnp.pi))
        spec = SpectralOps(grid)
        f = jnp.cos(grid.x1)
        lap = spec.laplacian(f)
        assert jnp.allclose(lap, -f, atol=0.01)

    def test_implicit_wave_update_stable(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        spec = SpectralOps(grid)
        cfg = default_config()
        phi_hat = spec.to_spectral(jnp.ones(grid.shape))
        w_hat = spec.to_spectral(jnp.ones(grid.shape))
        nl_w_hat = spec.to_spectral(jnp.zeros(grid.shape))
        new_phi, new_w = spec.implicit_wave_update(phi_hat, w_hat, nl_w_hat, 0.01, cfg)
        # Should be finite (no blow-up)
        assert jnp.all(jnp.isfinite(new_phi))


class TestIntegrators:
    def test_cfl_suggest_dt(self):
        grid = Grid3D(GridConfig(n1=16, n2=16, n3=16))
        cfg = default_config()
        dt = suggest_dt(cfg, grid)
        assert 0 < dt < 1.0

    def test_evolve_produces_finite(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        cfg = ChappieConfig(
            grid=GridConfig(n1=8, n2=8, n3=8),
            solver=SolverConfig(dt=0.005, n_steps_per_token=4, type="rk4"),
            memory=MemoryConfig(),
            hh=HHConfig(),
        )
        cfg.memory = MemoryConfig()  # use defaults
        spec = SpectralOps(grid)
        s = init_rest(grid, cfg)
        src = jnp.zeros(grid.shape)
        fce = jnp.zeros(grid.shape)
        s2 = evolve(s, src, fce, cfg, spec, n_steps=4, dt=0.005)
        assert jnp.all(jnp.isfinite(s2.phi))
        assert jnp.all(jnp.isfinite(s2.v))
