"""Chappie facade — the single public class for the model (spec §21).

Example::

    model = Chappie(config)
    text  = model.generate("Hello", max_new_tokens=50)
    loss  = model.train_step(token_ids)
    model.save("ckpt.npz")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, random

from chappie.core.config import ChappieConfig, default_config
from chappie.core.field import Grid3D
from chappie.core.state import ChappieState, init_rest, reset_activity
from chappie.encoding.tokenizer import (
    CharTokenizer, BytePairTokenizer, build_tokenizer,
)
from chappie.encoding.basis import (
    BasisParams, GaussianBasis, FourierBasis, WaveletBasis,
    build_basis, BASIS_REGISTRY,
)
from chappie.encoding.token_field import TokenFieldEncoder
from chappie.decoding.field_decoder import FieldDecoder, DecoderParams
from chappie.decoding.sampler import sample as sample_token
from chappie.solver.spectral import SpectralOps
from chappie.solver.integrators import evolve
from chappie.training.loss import nll, accuracy
from chappie.training.force import learning_force
from chappie.training.optimizer import adam_init, adam_update
from chappie.training.dataset import SyntheticCurriculum, TextDataset
from chappie.inference.generate import generate as generate_fn

log = logging.getLogger(__name__)

# -- Params container ----------------------------------------------------

class ChappieParams:
    """All learnable parameters (basis + decoder)."""
    __slots__ = ("basis", "decoder")
    def __init__(self, basis: BasisParams, decoder: DecoderParams):
        self.basis = basis
        self.decoder = decoder
    def as_tuple(self):
        return (self.basis, self.decoder)


# -- Chappie class -------------------------------------------------------

class Chappie:
    """Chappie CLEFPE field language model (spec §21)."""

    def __init__(
        self,
        config: ChappieConfig | None = None,
        key: Array | None = None,
        tokenizer=None,
    ) -> None:
        self.config = config or default_config()
        self.grid   = Grid3D(self.config.grid)
        self.spectral = SpectralOps(self.grid)

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = build_tokenizer(
                vocab_size=self.config.training.vocab_size,
                method=self.config.data.tokenizer,
            )
        V = len(self.tokenizer)

        # Basis
        key = key if key is not None else random.PRNGKey(0)
        k_embed, k_basis = random.split(key)
        self.basis = build_basis(
            self.config.encoding.basis, self.grid, V, k_basis,
            sigma=self.config.encoding.sigma,
            amp=self.config.encoding.amp,
        )

        # Encoder / decoder
        self.encoder = TokenFieldEncoder(self.basis, self.config.encoding)
        self.decoder = FieldDecoder(self.basis, self.config)

        # Params
        basis_p = self.basis.init_params()
        dec_p   = self.decoder.init_params(V)
        self.params = ChappieParams(basis_p, dec_p)

        # State (activity only — rest state)
        self.state = init_rest(self.grid, self.config)

        # Optimizer state (for tiny-param updates)
        self._m, self._v = adam_init(self.params.as_tuple())
        self.step = 0

        # Dataset
        self.dataset = self._build_dataset()

        # Build jitted inner functions (for train_step API)
        self._build_jitted()

    # -----------------------------------------------------------------
    def _build_dataset(self):
        cfg = self.config
        if cfg.data.source == "text":
            path = Path(cfg.data.text_path)
            if path.exists():
                return TextDataset(str(path), self.tokenizer, cfg.training.seq_len)
            log.warning("Text file %s not found; falling back to synthetic.", path)
        return SyntheticCurriculum(self.tokenizer, cfg.data)

    def _build_jitted(self) -> None:
        cfg = self.config
        grid = self.grid
        spectral = self.spectral
        encoder = self.encoder
        decoder = self.decoder
        n_steps = cfg.solver.n_steps_per_token
        lr = cfg.training.lr
        n_unroll = cfg.training.n_unroll
        force_gain = cfg.training.force_gain

        def _rollout(params, seq):
            basis_p, dec_p = params
            T = seq.shape[0]
            s0 = init_rest(grid, cfg)

            def body(carry, t):
                s, acc = carry
                src = encoder.basis.bump(seq[t], basis_p)
                target = seq[t + 1]
                if force_gain > 0:
                    probs = decoder.probs_from_phi(s.phi, basis_p, dec_p)
                    fce = learning_force(s.phi, target, probs, encoder.basis,
                                         basis_p, grid, cfg)
                else:
                    fce = jnp.zeros_like(s.phi)
                s2 = evolve(s, src, fce, cfg, spectral, n_steps=n_steps)
                logits = decoder.logits(s2, basis_p, dec_p)
                return (s2, acc + nll(logits, target)), None

            (_, loss), _ = jax.lax.scan(body, (s0, 0.0), jnp.arange(T - 1))
            return loss / jnp.maximum(T - 1, 1)

        def _short_unroll(params, seq):
            basis_p, dec_p = params
            T = seq.shape[0]
            n = min(n_unroll, int(T) - 1)
            s0 = init_rest(grid, cfg)

            def body(carry, t):
                s, acc = carry
                src = encoder.basis.bump(seq[t], basis_p)
                s2 = evolve(s, src, 0.0, cfg, spectral, n_steps=n_steps)
                logits = decoder.logits(s2, basis_p, dec_p)
                return (s2, acc + nll(logits, seq[t + 1])), None

            (_, acc), _ = jax.lax.scan(body, (s0, 0.0), jnp.arange(n))
            return acc / jnp.maximum(n, 1)

        @jax.jit
        def jit_rollout(params, seq):
            return _rollout(params, seq)

        @jax.jit
        def jit_param_update(params, seq, m, v, step):
            grads = jax.grad(_short_unroll)(params, seq)
            return adam_update(params, grads, m, v, step, lr=lr)

        def _eval_one(params, seq):
            basis_p, dec_p = params
            T = seq.shape[0]
            s = init_rest(grid, cfg)
            acc_nll = 0.0; acc_ok = 0.0
            for t in range(T - 1):
                src = encoder.basis.bump(seq[t], basis_p)
                s = evolve(s, src, 0.0, cfg, spectral, n_steps=n_steps)
                logits = decoder.logits(s, basis_p, dec_p)
                acc_nll = acc_nll + nll(logits, seq[t + 1])
                acc_ok  = acc_ok  + accuracy(logits, seq[t + 1])
            n = T - 1
            return acc_nll / n, acc_ok / n

        @jax.jit
        def jit_eval_batch(params, batch):
            return jax.vmap(lambda s: _eval_one(params, s))(batch)

        self._jit_rollout = jit_rollout
        self._jit_param_update = jit_param_update
        self._jit_eval_batch = jit_eval_batch

    # -----------------------------------------------------------------
    # Public API (spec §21)
    # -----------------------------------------------------------------

    def encode(self, text: str) -> Array:
        """Token IDs from text."""
        return self.tokenizer.encode(text)

    def decode(self, ids: Array) -> str:
        """Text from token IDs."""
        return self.tokenizer.decode(ids)

    def initial_condition(self, text: str) -> Array:
        """Φ(x,0) from text — sum of token bumps, L∞-normalized."""
        ids = self.tokenizer.encode(text)
        return self.encoder.initial_condition(ids, self.params.basis)

    def predict(self, state: ChappieState) -> Array:
        """Probability distribution over vocabulary (V,)."""
        return self.decoder.probs(state, self.params.basis,
                                  self.params.decoder)

    def logits(self, state: ChappieState) -> Array:
        """Logits (V,) from a field state."""
        return self.decoder.logits(state, self.params.basis,
                                   self.params.decoder)

    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 64,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        key: Array | None = None,
    ) -> str:
        """Autoregressive text generation (spec §19)."""
        return generate_fn(
            self, prompt=prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p, key=key,
        )

    def train_step(self, token_ids: Array) -> float:
        """One training step (force rollout + optional param update). Returns NLL."""
        params_t = self.params.as_tuple()
        loss = float(self._jit_rollout(params_t, token_ids))
        if self.step % self.config.training.param_every == 0:
            new_p, new_m, new_v = self._jit_param_update(
                params_t, token_ids, self._m, self._v, self.step
            )
            self.params = ChappieParams(new_p[0], new_p[1])
            self._m, self._v = new_m, new_v
        self.step += 1
        return loss

    def nll(self, text: str) -> float:
        """Average next-char NLL over a text (no learning)."""
        ids = self.tokenizer.encode(text)
        if ids.shape[0] < 2:
            return float("nan")
        return float(self._jit_rollout(self.params.as_tuple(), ids))

    def param_count(self) -> int:
        """Total number of stored scalar parameters."""
        basis_p = self.params.basis
        dec_p   = self.params.decoder
        return int(sum(x.size for x in [basis_p.widths, basis_p.amps, dec_p.bias]))

    # -----------------------------------------------------------------
    # Checkpointing (spec §17)
    # -----------------------------------------------------------------

    def save(self, path: str, step: int | None = None, metrics: dict | None = None) -> None:
        """Save model state to .npz (spec §17)."""
        basis_p = self.params.basis
        dec_p   = self.params.decoder
        arrs = {
            "format_version":    np.asarray(1),
            "config_yaml":       np.asarray(self.config.to_yaml(), dtype=object),
            "tokenizer_name":    np.asarray(self.tokenizer.name, dtype=object),
            "tokenizer_vocab":   np.asarray(self.tokenizer.to_list(), dtype=object),
            "basis_widths":      np.asarray(basis_p.widths),
            "basis_amps":        np.asarray(basis_p.amps),
            "decoder_bias":      np.asarray(dec_p.bias),
            "opt_m_widths":      np.asarray(self._m[0].widths),
            "opt_m_amps":        np.asarray(self._m[0].amps),
            "opt_m_bias":        np.asarray(self._m[1].bias),
            "opt_v_widths":      np.asarray(self._v[0].widths),
            "opt_v_amps":        np.asarray(self._v[0].amps),
            "opt_v_bias":        np.asarray(self._v[1].bias),
            "step":              np.asarray(self.step if step is None else step),
            "metrics_json":      np.asarray(json.dumps(metrics or {}), dtype=object),
        }
        np.savez(path, allow_pickle=True, **arrs)
        log.info("Saved checkpoint to %s", path)

    @classmethod
    def load(cls, path: str) -> "Chappie":
        """Load a checkpoint from .npz."""
        data = np.load(path, allow_pickle=True)
        cfg_yaml = str(data["config_yaml"])
        cfg = ChappieConfig.from_yaml_text(cfg_yaml)
        tok_name = str(data["tokenizer_name"])
        tok_vocab = list(data["tokenizer_vocab"])
        if tok_name == "bpe":
            tokenizer = CharTokenizer(alphabet="".join(tok_vocab))
        else:
            tokenizer = CharTokenizer(alphabet="".join(tok_vocab))
        model = cls(config=cfg, tokenizer=tokenizer)
        model.params = ChappieParams(
            basis=BasisParams(
                widths=jnp.asarray(data["basis_widths"]),
                amps=jnp.asarray(data["basis_amps"]),
            ),
            decoder=DecoderParams(
                bias=jnp.asarray(data["decoder_bias"]),
            ),
        )
        model._m = (BasisParams(jnp.asarray(data["opt_m_widths"]),
                                jnp.asarray(data["opt_m_amps"])),
                    DecoderParams(jnp.asarray(data["opt_m_bias"])))
        model._v = (BasisParams(jnp.asarray(data["opt_v_widths"]),
                                jnp.asarray(data["opt_v_amps"])),
                    DecoderParams(jnp.asarray(data["opt_v_bias"])))
        model.step = int(data["step"])
        return model
