"""Autoregressive text generation (spec §19).

prompt → encode → PDE evolution → probability distribution →
sample token → inject back → evolve again → repeat.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from chappie.core.config import ChappieConfig
from chappie.core.state import ChappieState, init_rest
from chappie.encoding.basis import BasisParams
from chappie.decoding.field_decoder import FieldDecoder, DecoderParams
from chappie.decoding.sampler import sample as sample_token
from chappie.solver.spectral import SpectralOps
from chappie.solver.integrators import evolve


def generate(
    model,  # Chappie instance
    prompt: str = "",
    max_new_tokens: int = 64,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    key: Array | None = None,
) -> str:
    """Generate text autoregressively from a prompt.

    Args:
        model:            Chappie instance with fitted params
        prompt:           starting text
        max_new_tokens:   number of tokens to generate
        temperature:      softmax temperature (None → use config)
        top_k:            top-k truncation (None → use config)
        top_p:            nucleus threshold (None → use config)
        key:              PRNG key (None → PRNGKey(0))

    Returns:
        generated string (prompt + new tokens)
    """
    cfg  = model.config
    grid = model.grid
    spectral = SpectralOps(grid)
    basis = model.basis
    encoder = model.encoder
    decoder = model.decoder
    basis_p, dec_p = model.params.as_tuple()

    if temperature is None: temperature = cfg.sampling.temperature
    if top_k is None:       top_k = cfg.sampling.top_k
    if top_p is None:       top_p = cfg.sampling.top_p
    if key is None:         key = jax.random.PRNGKey(0)

    # Encode the prompt
    ids = model.tokenizer.encode(prompt)

    # Evolve the prompt into the field (no force)
    state = init_rest(grid, cfg)
    for tok_id in ids:
        src = basis.bump(int(tok_id), basis_p)
        state = evolve(state, src, jnp.zeros_like(state.phi), cfg, spectral,
                       n_steps=cfg.solver.n_steps_per_token)

    # Autoregressive generation loop
    out = list(prompt)
    for _ in range(max_new_tokens):
        key, sub = jax.random.split(key)
        logits = decoder.logits(state, basis_p, dec_p)
        nid = int(sample_token(logits, sub, temperature, top_k, top_p))
        out.append(model.tokenizer.itos[nid])
        src = basis.bump(nid, basis_p)
        state = evolve(state, src, jnp.zeros_like(state.phi), cfg, spectral,
                       n_steps=cfg.solver.n_steps_per_token)
    return "".join(out)
