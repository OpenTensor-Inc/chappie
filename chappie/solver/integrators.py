"""Integrators: RK4, IMEX semi-implicit, and CFL guard (spec §12/§14).

The default is IMEX: diffusion/wave solved implicitly (per-mode in
Fourier space), nonlinear terms (HH, Φ³, coupling) explicit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, tree_util

from chappie.core.config import ChappieConfig
from chappie.core.field import Grid3D
from chappie.core.hh import hh_rhs
from chappie.core.memory import memory_rhs
from chappie.core.clefpe import clefpe_rhs
from chappie.core.state import ChappieState
from chappie.solver.boundary import absorbing_mask
from chappie.solver.spectral import SpectralOps


# -- tree helpers ----------------------------------------------------------

def _scale_state(s: ChappieState, a) -> ChappieState:
    return tree_util.tree_map(lambda x: x * a, s)


def _add_state(a: ChappieState, b: ChappieState) -> ChappieState:
    return tree_util.tree_map(lambda x, y: x + y, a, b)


def _clip_state(state: ChappieState, cfg: ChappieConfig) -> ChappieState:
    c = cfg.solver.clip_phi
    if c and c > 0:
        return state._replace(phi=jnp.clip(state.phi, -c, c))
    return state


# -- coupled RHS (used by RK4) -------------------------------------------

def coupled_rhs(
    state: ChappieState,
    source: Array,
    force: Array,
    cfg: ChappieConfig,
    lap,
    sponge: Array | None = None,
) -> ChappieState:
    """Full RHS of the coupled CLEFPE + HH + memory system."""
    phi   = state.phi
    phi_t = state.phi_t

    dphi, dphi_t = clefpe_rhs(state, force, cfg, lap, sponge)

    # Memory fields
    dmem = memory_rhs(state.mem, phi, lap, cfg.memory)
    if sponge is not None:
        dmem = dmem - sponge[None, ...] * state.mem

    # HH (explicit)
    i_in = source + cfg.hh.xi * phi
    dv, dm, dh, dn = hh_rhs(state.v, state.m, state.h, state.n, i_in, cfg.hh)

    return ChappieState(dphi, dphi_t, dmem, dv, dm, dh, dn)


# -- RK4 step -------------------------------------------------------------

def rk4_step(
    state: ChappieState,
    source: Array,
    force: Array,
    cfg: ChappieConfig,
    spectral: SpectralOps,
) -> ChappieState:
    dt = cfg.solver.dt
    lap = spectral.laplacian
    sponge = absorbing_mask(spectral.grid) if cfg.grid.boundary == "absorbing" else None

    k1 = coupled_rhs(state, source, force, cfg, lap, sponge)
    s2 = _add_state(state, _scale_state(k1, dt / 2))
    k2 = coupled_rhs(s2, source, force, cfg, lap, sponge)
    s3 = _add_state(state, _scale_state(k2, dt / 2))
    k3 = coupled_rhs(s3, source, force, cfg, lap, sponge)
    s4 = _add_state(state, _scale_state(k3, dt))
    k4 = coupled_rhs(s4, source, force, cfg, lap, sponge)

    new = _add_state(state,
        _scale_state(
            _add_state(
                _add_state(k1, _scale_state(k2, 2.0)),
                _add_state(_scale_state(k3, 2.0), k4),
            ),
            dt / 6.0,
        ))
    return _clip_state(new, cfg)


# -- IMEX semi-implicit step ----------------------------------------------

def imex_step(
    state: ChappieState,
    source: Array,
    force: Array,
    cfg: ChappieConfig,
    spectral: SpectralOps,
) -> ChappieState:
    dt = cfg.solver.dt
    phi, phi_t = state.phi, state.phi_t

    # Nonlinear acceleration for the W equation (real space)
    chi = jnp.asarray(cfg.memory.chi, dtype=jnp.float32)
    sponge = absorbing_mask(spectral.grid) if cfg.grid.boundary == "absorbing" else None

    nl_w = (cfg.clefpe.kappa * (state.v - cfg.clefpe.V0)
            + jnp.tensordot(chi, state.mem, axes=1)
            + force
            - cfg.clefpe.lam_R * (cfg.clefpe.a * phi + cfg.clefpe.b * phi**3))
    if sponge is not None:
        nl_w = nl_w - sponge * phi_t

    # Implicit update of (Φ, W) via per-mode backward Euler (§12)
    phi_hat = spectral.to_spectral(phi)
    w_hat   = spectral.to_spectral(phi_t)
    nl_w_hat = spectral.to_spectral(nl_w)
    new_phi_hat, new_w_hat = spectral.implicit_wave_update(
        phi_hat, w_hat, nl_w_hat, dt, cfg
    )
    phi_new   = spectral.from_spectral(new_phi_hat)
    phi_t_new = spectral.from_spectral(new_w_hat)

    # Memory: implicit diffusion/decay, explicit source
    mem_list = []
    for s_i in range(state.mem.shape[0]):
        m_hat = spectral.to_spectral(state.mem[s_i])
        src_hat = spectral.to_spectral(cfg.memory.eta[s_i] * phi)
        if sponge is not None:
            src_hat = src_hat - spectral.to_spectral(sponge * state.mem[s_i])
        mem_new_i = spectral.from_spectral(
            spectral.implicit_memory_update(
                m_hat, src_hat, dt, cfg.memory.rho[s_i], cfg.memory.mu[s_i]
            )
        )
        mem_list.append(mem_new_i)
    mem_new = jnp.stack(mem_list)

    # HH (explicit Euler — dt is well below the gating time constants)
    i_in = source + cfg.hh.xi * phi
    dv, dm, dh, dn = hh_rhs(state.v, state.m, state.h, state.n, i_in, cfg.hh)
    v_new = state.v + dt * dv
    m_new = state.m + dt * dm
    h_new = state.h + dt * dh
    n_new = state.n + dt * dn

    return _clip_state(
        ChappieState(phi_new, phi_t_new, mem_new, v_new, m_new, h_new, n_new),
        cfg,
    )


# -- CFL / safety ---------------------------------------------------------

def suggest_dt(cfg: ChappieConfig, grid: Grid3D) -> float:
    """Suggest a safe time step based on CFL and HH time constants."""
    import math
    safety = max(cfg.solver.cfl_safety, 1e-3)
    dx = grid.dx
    c  = cfg.clefpe.c
    dt_wave = dx / (c * math.sqrt(3.0) + 1e-12) * safety
    dt_hh   = 0.02 * safety   # HH gating time scale
    return float(min(dt_wave, dt_hh))


# -- evolve dispatch ------------------------------------------------------

def evolve(
    state: ChappieState,
    source: Array,
    force: Array,
    cfg: ChappieConfig,
    spectral: SpectralOps,
    *,
    n_steps: int | None = None,
    dt: float | None = None,
) -> ChappieState:
    """Evolve the state for n_steps of size dt.

    Dispatches to IMEX (default) or RK4 based on ``cfg.solver.type``.
    A smooth pulse envelope is applied to source/force over the window.
    """
    n_steps_ = n_steps if n_steps is not None else cfg.solver.n_steps_per_token
    dt_ = float(dt) if dt is not None else cfg.solver.dt
    # Cap dt by the explicit stability bound
    dt_ = min(dt_, suggest_dt(cfg, spectral.grid))

    # Pulse envelope: smooth on/off within the window
    active = max(1, int(round(cfg.encoding.pulse_width * n_steps_)))
    off_start = active - 1

    def envelope(i):
        x = i - off_start
        return jnp.where(
            x <= 0, 1.0,
            jnp.where(x >= 2, 0.0,
                       0.5 * (1.0 + jnp.cos(jnp.pi * x / 2.0))),
        )

    def body(i, s):
        env = envelope(i)
        src = source * env
        fce = force * env
        if cfg.solver.type == "imex":
            return imex_step(s, src, fce, cfg, spectral)
        elif cfg.solver.type == "rk4":
            return rk4_step(s, src, fce, cfg, spectral)
        raise ValueError(f"Unknown solver type: {cfg.solver.type!r}")

    return jax.lax.fori_loop(0, n_steps_, body, state)


def evolve_safe(
    state: ChappieState,
    source: Array,
    force: Array,
    cfg: ChappieConfig,
    spectral: SpectralOps,
    *,
    n_steps: int | None = None,
    dt: float | None = None,
) -> ChappieState:
    """Guarded evolve: retries with halved dt on NaN/Inf."""
    dt_ = dt if dt is not None else cfg.solver.dt
    max_h = cfg.solver.max_halvings
    halvings = 0
    while True:
        out = evolve(state, source, force, cfg, spectral,
                     n_steps=n_steps, dt=dt_)
        bad = jnp.isnan(jnp.sum(out.phi)) | jnp.isinf(jnp.sum(out.phi))
        if (not bad) or halvings >= max_h:
            return out
        dt_ = dt_ / 2.0
        halvings += 1
