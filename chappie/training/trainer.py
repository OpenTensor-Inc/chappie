"""ChappieTrainer: orchestrates teacher-forced training with CLEFPE learning force.

The core step (jitted):
  1. Teacher-forced rollout with analytic learning force
  2. Optional tiny-param gradient update (truncated BPTT, short unroll)
  3. Metrics logging and checkpointing
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import Array, random

from chappie.core.config import ChappieConfig
from chappie.core.field import Grid3D
from chappie.core.state import ChappieState, init_rest
from chappie.solver.spectral import SpectralOps
from chappie.solver.integrators import evolve
from chappie.training.loss import nll, accuracy
from chappie.training.force import learning_force
from chappie.training.optimizer import adam_init, adam_update

log = logging.getLogger(__name__)

def _key_to_int(key) -> int:
    """Convert a JAX PRNG key or int to a plain Python int for the dataset."""
    if isinstance(key, int):
        return key
    if hasattr(key, '__getitem__'):
        return int(key[0])
    return int(key)


class ChappieTrainer:
    """Manages the hybrid training loop (force + tiny-param gradient).

    The jitted functions are built once in ``__init__`` and re-used across
    training steps.  Static hyperparameters (grid size, dt, n_steps, ...)
    are bound via closures so the compiled shapes are stable.
    """

    def __init__(self, model, cfg: ChappieConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.grid = model.grid
        self.spectral = SpectralOps(self.grid)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.history: list[dict] = []
        self._build_jitted()

    def _build_jitted(self) -> None:
        cfg = self.cfg
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
                    fce = learning_force(s.phi, target, probs, encoder.basis, basis_p, grid, cfg)
                else:
                    fce = jnp.zeros_like(s.phi)
                s2 = evolve(s, src, fce, cfg, spectral, n_steps=n_steps)
                logits = decoder.logits(s2, basis_p, dec_p)
                return (s2, acc + nll(logits, target)), None

            (s_final, acc), _ = jax.lax.scan(body, (s0, 0.0), jnp.arange(T - 1))
            return s_final, acc / jnp.maximum(T - 1, 1)

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
            _, loss = _rollout(params, seq)
            return loss

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

    def _level_for_step(self, step: int, steps: int) -> int:
        if not self.cfg.data.curriculum:
            return min(4, self.cfg.data.max_level)
        frac = step / max(steps - 1, 1)
        return min(self.cfg.data.max_level, 1 + int(frac * self.cfg.data.max_level))

    def train(self, steps: int | None = None) -> dict:
        cfg = self.cfg
        steps = steps or cfg.training.steps
        seed = cfg.training.seed
        key = random.PRNGKey(seed)
        model = self.model
        dataset = model.dataset
        seq_len = cfg.training.seq_len
        batch_size = cfg.training.batch_size
        ckpt_dir = Path(cfg.training.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        param_every = cfg.training.param_every
        val_every   = cfg.training.val_every
        ckpt_every  = cfg.training.checkpoint_every

        # Work with a local params tuple
        params_t = model.params.as_tuple()
        m, v = model._m, model._v

        log.info("Starting training: %d steps, seq_len=%d, grid=%d^3",
                 steps, seq_len, cfg.grid.n1)
        t0 = time.time()
        for step in range(steps):
            level = self._level_for_step(step, steps)
            k1, key = random.split(key)
            batch = dataset.sample_batch(batch_size, seq_len, level, key=int(k1[0]))
            losses = []
            for i in range(batch_size):
                seq = batch[i]
                loss = float(self._jit_rollout(params_t, seq))
                losses.append(loss)
                if step % param_every == 0:
                    params_t, m, v = self._jit_param_update(
                        params_t, seq, m, v, step)

            # Write params back to model
            from chappie.model import ChappieParams
            from chappie.encoding.basis import BasisParams
            from chappie.decoding.field_decoder import DecoderParams
            model.params = ChappieParams(
                basis=BasisParams(widths=params_t[0].widths, amps=params_t[0].amps),
                decoder=DecoderParams(bias=params_t[1].bias),
            )
            model._m, model._v = m, v
            model.step = step + 1

            import numpy as np
            mean_loss = float(np.mean(losses))
            self.history.append({"step": step, "train_nll": mean_loss, "level": level})
            if (step + 1) % val_every == 0:
                val = self.evaluate(n_seqs=4, level=level, key=key)
                self.history[-1].update(val)
                elapsed = time.time() - t0
                log.info(
                    "step %d/%d  train_nll=%.4f  val_nll=%.4f  val_ppl=%.3f  "
                    "val_acc=%.3f  [%.1fs]",
                    step + 1, steps, mean_loss,
                    val["val_nll"], val["val_ppl"], val["val_acc"], elapsed,
                )
            if (step + 1) % ckpt_every == 0:
                model.save(str(ckpt_dir / f"step-{step+1:05d}.npz"),
                           step=step, metrics=self.history[-1])
        final = self.evaluate(n_seqs=8, level=self.cfg.data.max_level, key=key)
        model.save(str(ckpt_dir / "final.npz"), step=steps, metrics=final)
        return final

    def evaluate(self, n_seqs: int = 8, level: int = 2,
                 key: int | None = None) -> dict:
        cfg = self.cfg
        dataset = self.model.dataset
        key_int = _key_to_int(key) if key is not None else 0
        batch = dataset.sample_batch(n_seqs, cfg.training.seq_len, level, key=key_int)
        params_t = self.model.params.as_tuple()
        nlls, accs = self._jit_eval_batch(params_t, batch)
        mean_nll = float(jnp.mean(nlls))
        mean_acc = float(jnp.mean(accs))
        return {
            "val_nll": mean_nll,
            "val_ppl": float(jnp.exp(mean_nll)),
            "val_acc": mean_acc,
        }
