"""Tests for chappie.core — grid, HH, state, CLEFPE, memory, energy."""

import jax
import jax.numpy as jnp
import pytest

from chappie.core.config import (
    ChappieConfig, GridConfig, SolverConfig, ClefpeConfig,
    MemoryConfig, HHConfig, EncodingConfig, DecodingConfig,
    TrainingConfig, DataConfig, SamplingConfig,
    default_config, load_yaml_subset, dump_yaml_subset,
)
from chappie.core.field import Grid3D
from chappie.core.hh import hh_rates, gating_steady_state, hh_rhs
from chappie.core.state import ChappieState, init_rest, reset_activity
from chappie.core.memory import memory_rhs
from chappie.core.clefpe import clefpe_rhs, potential_force
from chappie.core.energy import energy_components, total_energy
from chappie.core.diffusion import diffusion_rhs
from chappie.core.wave import wave_terms


# -- Config tests ---------------------------------------------------------

class TestConfig:
    def test_default_roundtrip(self):
        cfg = default_config()
        d = cfg.to_dict()
        cfg2 = ChappieConfig.from_dict(d)
        assert cfg.grid.n1 == cfg2.grid.n1
        assert cfg.clefpe.c == cfg2.clefpe.c
        assert cfg.training.lr == cfg2.training.lr

    def test_yaml_roundtrip(self):
        cfg = default_config()
        y = cfg.to_yaml()
        cfg2 = ChappieConfig.from_yaml_text(y)
        assert cfg.grid.boundary == cfg2.grid.boundary
        assert cfg.memory.rho == cfg2.memory.rho

    def test_overrides(self):
        cfg = default_config()
        cfg2 = cfg.with_overrides({"training.lr": 0.01, "clefpe.c": 2.0})
        assert cfg2.training.lr == 0.01
        assert cfg2.clefpe.c == 2.0
        assert cfg.training.lr == 0.002  # original unchanged


# -- Grid tests -----------------------------------------------------------

class TestGrid:
    def test_shape(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        assert grid.shape == (8, 8, 8)
        assert grid.x1.shape == (8, 8, 8)

    def test_laplacian_of_constant(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        c = jnp.ones(grid.shape)
        lap = grid.laplacian(c)
        assert jnp.allclose(lap, 0.0, atol=1e-6)

    def test_laplacian_of_sine(self):
        grid = Grid3D(GridConfig(n1=16, n2=16, n3=16, length=2*jnp.pi))
        # cos(x1): ∇²cos = −cos
        f = jnp.cos(grid.x1)
        lap = grid.laplacian(f)
        assert jnp.allclose(lap, -f, atol=0.1)  # FD order ~1%; spectral should be exact


# -- HH tests ------------------------------------------------------------

class TestHH:
    def test_rates_no_nan(self):
        v = jnp.linspace(-80, 40, 50)
        am, bm, ah, bh, an, bn = hh_rates(v)
        assert not jnp.any(jnp.isnan(am))
        assert not jnp.any(jnp.isnan(bm))
        assert not jnp.any(jnp.isnan(ah))
        assert not jnp.any(jnp.isnan(bh))
        assert not jnp.any(jnp.isnan(an))
        assert not jnp.any(jnp.isnan(bn))

    def test_steady_state_bounded(self):
        m, h, n = gating_steady_state(-65.0)
        assert 0.0 < float(m) < 1.0
        assert 0.0 < float(h) < 1.0
        assert 0.0 < float(n) < 1.0

    def test_rest_equilibrium(self):
        cfg = default_config()
        shape = (cfg.grid.n1, cfg.grid.n2, cfg.grid.n3)
        v = jnp.full(shape, cfg.hh.rest)
        m0, h0, n0 = gating_steady_state(cfg.hh.rest)
        i_in = jnp.zeros(shape)
        dv, dm, dh, dn = hh_rhs(v, m0 * jnp.ones_like(v),
                                  h0 * jnp.ones_like(v),
                                  n0 * jnp.ones_like(v), i_in, cfg.hh)
        # At rest, dv ≈ 0 (currents balance)
        assert jnp.abs(dv).max() < 1.0


# -- State tests ----------------------------------------------------------

class TestState:
    def test_init_rest(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        cfg = default_config()
        s = init_rest(grid, cfg)
        assert s.phi.shape == (8, 8, 8)
        assert s.mem.shape[0] == cfg.memory.n_fields
        assert float(jnp.mean(jnp.abs(s.phi))) == 0.0
        assert jnp.allclose(s.v, cfg.hh.rest, atol=0.1)

    def test_reset_activity(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        cfg = default_config()
        s = init_rest(grid, cfg)
        s2 = s._replace(phi=jnp.ones(grid.shape))  # perturb
        s3 = reset_activity(s2, grid, cfg)
        assert float(jnp.mean(s3.phi)) == 0.0


# -- Memory tests --------------------------------------------------------

class TestMemory:
    def test_decay_no_source(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        cfg = default_config()
        mem0 = jnp.ones((cfg.memory.n_fields, *grid.shape))
        phi0 = jnp.zeros(grid.shape)
        from chappie.solver.spectral import SpectralOps
        spectral = SpectralOps(grid)
        dmem = memory_rhs(mem0, phi0, spectral.laplacian, cfg.memory)
        # Memory should decay: dmem < 0 where mem > 0
        assert jnp.all(dmem[:, 0, 0, 0] < 0)


# -- CLEFPE tests --------------------------------------------------------

class TestCLEFPE:
    def test_no_force_zero_rhs(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        cfg = default_config()
        from chappie.solver.spectral import SpectralOps
        spectral = SpectralOps(grid)
        s = init_rest(grid, cfg)
        dphi, dW = clefpe_rhs(s, jnp.zeros_like(s.phi), cfg, spectral.laplacian)
        # All fields zero → RHS ≈ 0 (tiny residual from coupling)
        assert jnp.abs(dphi).max() < 0.1


# -- Energy tests --------------------------------------------------------

class TestEnergy:
    def test_energy_components(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        cfg = default_config()
        s = init_rest(grid, cfg)
        comps = energy_components(s, cfg, grid)
        assert "diffusion" in comps
        assert "wave" in comps
        assert "hh" in comps
        assert "memory" in comps
        assert "regularization" in comps
        assert float(total_energy(s, cfg, grid)) > 0  # HH energy > 0
