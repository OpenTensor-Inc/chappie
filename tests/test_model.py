"""Integration tests for Chappie facade (spec §21)."""

import jax
import jax.numpy as jnp
import pytest
import numpy as np

from chappie.core.config import (
    ChappieConfig, GridConfig, SolverConfig, ClefpeConfig,
    MemoryConfig, HHConfig, EncodingConfig, DecodingConfig,
    TrainingConfig, DataConfig, SamplingConfig, default_config,
)
from chappie.model import Chappie


def _small_config():
    """Config for fast testing (small grid, few steps)."""
    return ChappieConfig(
        grid=GridConfig(n1=8, n2=8, n3=8, length=1.0, boundary="periodic"),
        solver=SolverConfig(dt=0.005, n_steps_per_token=4, type="rk4",
                            clip_phi=2.0, cfl_safety=0.5),
        clefpe=ClefpeConfig(),
        memory=MemoryConfig(),
        hh=HHConfig(),
        encoding=EncodingConfig(basis="gaussian", sigma=0.4, amp=1.0,
                                pulse_width=0.8),
        decoding=DecodingConfig(tau=0.5, read_modalities=("phi",)),
        training=TrainingConfig(
            vocab_size=96, seq_len=8, batch_size=1,
            steps=10, lr=0.002, n_unroll=2, param_every=2,
            force_gain=0.3,
        ),
        data=DataConfig(source="synthetic", max_level=1),
        sampling=SamplingConfig(temperature=0.8),
    )


class TestChappieModel:
    @pytest.fixture
    def model(self):
        cfg = _small_config()
        return Chappie(cfg, key=jax.random.PRNGKey(0))

    def test_init(self, model):
        assert model.config.grid.n1 == 8
        assert model.param_count() > 0
        print(f"Param count: {model.param_count()}")

    def test_encode_decode(self, model):
        ids = model.encode("hello")
        text = model.decode(ids)
        assert text == "hello"

    def test_logits_shape(self, model):
        from chappie.core.state import init_rest
        s = init_rest(model.grid, model.config)
        logits = model.logits(s)
        assert logits.shape == (model.config.training.vocab_size,)

    def test_predict_sums_to_one(self, model):
        from chappie.core.state import init_rest
        s = init_rest(model.grid, model.config)
        probs = model.predict(s)
        assert jnp.abs(jnp.sum(probs) - 1.0) < 1e-4

    def test_initial_condition(self, model):
        phi0 = model.initial_condition("abc")
        assert phi0.shape == model.grid.shape
        assert jnp.max(jnp.abs(phi0)) <= 1.0 + 1e-6

    def test_train_step_returns_finite(self, model):
        ids = model.encode("abcde")
        loss = model.train_step(ids)
        assert jnp.isfinite(loss)
        print(f"Train step loss: {loss}")

    def test_generate_returns_string(self, model):
        text = model.generate(prompt="a", max_new_tokens=5,
                              key=jax.random.PRNGKey(42))
        assert isinstance(text, str)
        assert len(text) > 0
        print(f"Generated: {text!r}")

    def test_nll_finite(self, model):
        val = model.nll("hello")
        assert jnp.isfinite(val)
        print(f"NLL of 'hello': {val}")

    def test_multi_train_step(self, model):
        losses = []
        for _ in range(5):
            ids = model.encode("abcde")
            loss = model.train_step(ids)
            losses.append(float(loss))
        # All losses finite
        assert all(np.isfinite(l) for l in losses)
        print(f"Losses: {losses}")

    def test_save_load_roundtrip(self, model, tmp_path):
        path = str(tmp_path / "test_ckpt.npz")
        model.save(path, step=42, metrics={"val": 1.5})
        model2 = Chappie.load(path)
        assert model2.step == 42
        # Params should match
        w1 = np.asarray(model.params.basis.widths)
        w2 = np.asarray(model2.params.basis.widths)
        assert np.allclose(w1, w2, atol=1e-6)
