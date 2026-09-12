"""Tests for chappie.training — loss, force, optimizer, dataset."""

import jax
import jax.numpy as jnp
import pytest

from chappie.core.config import GridConfig, ChappieConfig, default_config
from chappie.core.field import Grid3D
from chappie.core.state import init_rest
from chappie.encoding.basis import GaussianBasis
from chappie.encoding.tokenizer import CharTokenizer
from chappie.encoding.token_field import TokenFieldEncoder
from chappie.decoding.field_decoder import FieldDecoder
from chappie.training.loss import nll, accuracy, perplexity
from chappie.training.force import learning_force
from chappie.training.optimizer import adam_init, adam_update
from chappie.training.dataset import SyntheticCurriculum


class TestLoss:
    def test_nll_random(self):
        logits = jnp.zeros(10)  # uniform distribution
        target = jnp.int32(3)
        val = nll(logits, target)
        assert jnp.abs(val - jnp.log(10.0)) < 0.01

    def test_accuracy_correct(self):
        logits = jnp.array([0.1, 5.0, 0.1, 0.1])
        assert float(accuracy(logits, jnp.int32(1))) == 1.0

    def test_perplexity(self):
        logits = jnp.zeros(10)
        val = perplexity(logits, jnp.int32(0))
        assert jnp.abs(val - 10.0) < 0.01


class TestForce:
    def test_force_positive_in_direction(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        cfg = default_config()
        gb = GaussianBasis(grid, 5, jax.random.PRNGKey(0))
        p = gb.init_params()
        phi = gb.bump(0, p)
        probs = jnp.array([0.5, 0.2, 0.1, 0.1, 0.1])
        fce = learning_force(phi, jnp.int32(2), probs, gb, p, grid, cfg)
        # Force should have non-zero norm
        assert jnp.sqrt(jnp.sum(fce**2)) > 0


class TestOptimizer:
    def test_adam_step(self):
        params = {"w": jnp.ones(5)}
        grads = {"w": jnp.ones(5)}
        m, v = adam_init(params)
        new_p, new_m, new_v = adam_update(params, grads, m, v, step=1, lr=0.01)
        assert float(new_p["w"][0]) < 1.0  # weights decreased

    def test_zero_grad_no_change(self):
        params = {"w": jnp.array([1.0, 2.0, 3.0])}
        grads = {"w": jnp.zeros(3)}
        m, v = adam_init(params)
        new_p, _, _ = adam_update(params, grads, m, v, step=1, lr=0.01)
        assert jnp.allclose(new_p["w"], params["w"], atol=1e-6)


class TestDataset:
    def test_synthetic_sample_batch(self):
        tok = CharTokenizer(vocab_size=96)
        cfg = default_config()
        ds = SyntheticCurriculum(tok, cfg.data)
        batch = ds.sample_batch(2, 16, level=1, key=0)
        assert batch.shape == (2, 16)
        assert batch.dtype == jnp.int32

    def test_gen_text_level1(self):
        tok = CharTokenizer(vocab_size=96)
        cfg = default_config()
        ds = SyntheticCurriculum(tok, cfg.data)
        import random
        rng = random.Random(42)
        text = ds.gen_text(1, 20, rng)
        assert len(text) == 20
