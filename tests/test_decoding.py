"""Tests for chappie.decoding — field decoder and sampler."""

import jax
import jax.numpy as jnp
import pytest

from chappie.core.config import GridConfig, DecodingConfig, ChappieConfig, default_config
from chappie.core.field import Grid3D
from chappie.core.state import init_rest
from chappie.encoding.basis import GaussianBasis
from chappie.decoding.field_decoder import FieldDecoder, DecoderParams
from chappie.decoding.sampler import sample


class TestFieldDecoder:
    @pytest.fixture
    def setup(self):
        cfg = default_config()
        grid = Grid3D(cfg.grid)
        V = 10
        basis = GaussianBasis(grid, V, jax.random.PRNGKey(0))
        dec = FieldDecoder(basis, cfg)
        p = basis.init_params()
        dp = dec.init_params(V)
        return grid, basis, dec, p, dp, cfg

    def test_logits_shape(self, setup):
        grid, basis, dec, p, dp, cfg = setup
        s = init_rest(grid, cfg)
        logits = dec.logits(s, p, dp)
        assert logits.shape == (10,)

    def test_probs_sum_to_one(self, setup):
        grid, basis, dec, p, dp, cfg = setup
        s = init_rest(grid, cfg)
        probs = dec.probs(s, p, dp)
        assert jnp.abs(jnp.sum(probs) - 1.0) < 1e-5

    def test_logits_from_phi(self, setup):
        grid, basis, dec, p, dp, cfg = setup
        phi = basis.bump(3, p)
        logits = dec.logits_from_phi(phi, p, dp)
        # Token 3 has highest projection → highest logit
        assert int(jnp.argmax(logits)) == 3


class TestSampler:
    def test_argmax(self):
        logits = jnp.array([1.0, 5.0, 0.5, -1.0])
        key = jax.random.PRNGKey(42)
        out = sample(logits, key, temperature=0.01)
        assert int(out) == 1

    def test_top_k(self):
        logits = jnp.array([1.0, 5.0, 0.5, -1.0])
        key = jax.random.PRNGKey(42)
        # top_k=1 always picks the max
        out = sample(logits, key, top_k=1)
        assert int(out) == 1
