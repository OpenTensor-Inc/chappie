"""Tests for chappie.encoding — tokenizer, basis, token_field."""

import jax
import jax.numpy as jnp
import pytest

from chappie.core.config import GridConfig, EncodingConfig, default_config
from chappie.core.field import Grid3D
from chappie.encoding.tokenizer import CharTokenizer, BytePairTokenizer
from chappie.encoding.basis import (
    GaussianBasis, FourierBasis, WaveletBasis, BasisParams,
)
from chappie.encoding.token_field import TokenFieldEncoder


# -- Tokenizer tests -----------------------------------------------------

class TestCharTokenizer:
    def test_encode_decode_roundtrip(self):
        tok = CharTokenizer(vocab_size=96)
        text = "Hello world! 123"
        ids = tok.encode(text)
        assert ids.dtype == jnp.int32
        assert tok.decode(ids) == text

    def test_vocab_size(self):
        tok = CharTokenizer(vocab_size=96)
        assert len(tok) == 96

    def test_persian_chars(self):
        tok = CharTokenizer()
        text = "سلام"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text


class TestBPETokenizer:
    def test_roundtrip(self):
        texts = ["the cat sat on the mat", "the dog sat on the log"]
        tok = BytePairTokenizer.train(texts, vocab_size=32)
        for t in texts:
            ids = tok.encode(t)
            assert tok.decode(ids) == t


# -- Basis tests ---------------------------------------------------------

class TestGaussianBasis:
    @pytest.fixture
    def gbasis(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        return GaussianBasis(grid, 10, jax.random.PRNGKey(0), sigma=0.4, amp=1.0), grid

    def test_bump_shape(self, gbasis):
        basis, grid = gbasis
        p = basis.init_params()
        psi = basis.bump(0, p)
        assert psi.shape == grid.shape

    def test_project_positive(self, gbasis):
        basis, grid = gbasis
        p = basis.init_params()
        # Project identity-like field: should be > 0 for token 0
        field = basis.bump(0, p)
        R = basis.project(field, p)
        assert float(R[0]) > 0

    def test_project_normalization(self, gbasis):
        basis, grid = gbasis
        p = basis.init_params()
        psi = basis.bump_norm(0, p)
        norm_sq = jnp.sum(psi**2) * grid.dV
        assert jnp.abs(norm_sq - 1.0) < 0.01


class TestFourierBasis:
    def test_project_positive(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        fb = FourierBasis(grid, 10, jax.random.PRNGKey(0), n_modes=8)
        p = fb.init_params()
        field = fb.bump(0, p)
        R = fb.project(field, p)
        assert float(R[0]) > 0


class TestWaveletBasis:
    def test_project_positive(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        wb = WaveletBasis(grid, 10, jax.random.PRNGKey(0), sigma=0.4, amp=1.0)
        p = wb.init_params()
        field = wb.bump(0, p)
        R = wb.project(field, p)
        assert float(R[0]) > 0


# -- TokenFieldEncoder tests ---------------------------------------------

class TestTokenFieldEncoder:
    def test_source_shape(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        gb = GaussianBasis(grid, 10, jax.random.PRNGKey(0))
        enc = TokenFieldEncoder(gb, EncodingConfig())
        p = gb.init_params()
        src = enc.source(0, p)
        assert src.shape == grid.shape

    def test_initial_condition_normalized(self):
        grid = Grid3D(GridConfig(n1=8, n2=8, n3=8))
        gb = GaussianBasis(grid, 10, jax.random.PRNGKey(0))
        enc = TokenFieldEncoder(gb, EncodingConfig())
        p = gb.init_params()
        ids = jnp.array([0, 1, 2], dtype=jnp.int32)
        phi0 = enc.initial_condition(ids, p)
        assert jnp.max(jnp.abs(phi0)) <= 1.0 + 1e-6
