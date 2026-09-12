"""Hodgkin-Huxley field equations (spec §4/§10).

Spatial Hodgkin-Huxley: V(x,t), m(x,t), h(x,t), n(x,t) are 3D fields
on the torus Ω. The gating variables use the classical rate functions,
guarded against division-by-zero near V = -40 and V = -55.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _x_over_1mexp(x: Array) -> Array:
    """x / (1 - exp(-x)), safe at x = 0 → 1."""
    return jnp.where(jnp.abs(x) < 1e-6, 1.0, x / (-jnp.expm1(-x)))


def hh_rates(v: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Rate functions (α_m, β_m, α_h, β_h, α_n, β_n) as functions of V.

    Note: these are standard HH rates with dimensionless currents and
    time constants in arbitrary field units (not necessarily milliseconds).
    """
    a = (v + 40.0) / 10.0
    alpha_m = _x_over_1mexp(a)
    beta_m = 4.0 * jnp.exp(-(v + 65.0) / 18.0)

    alpha_h = 0.07 * jnp.exp(-(v + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + jnp.exp(-(v + 35.0) / 10.0))

    b = (v + 55.0) / 10.0
    alpha_n = 0.1 * _x_over_1mexp(b)
    beta_n = 0.125 * jnp.exp(-(v + 65.0) / 80.0)

    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n


def gating_steady_state(v_rest: float) -> tuple[Array, Array, Array]:
    """Steady-state gating at a constant voltage (for initialization)."""
    v = jnp.array(v_rest, dtype=jnp.float32)
    am, bm, ah, bh, an, bn = hh_rates(v)
    m = am / (am + bm)
    h = ah / (ah + bh)
    n = an / (an + bn)
    return m, h, n


def hh_rhs(
    v: Array, m: Array, h: Array, n: Array,
    i_in: Array, cfg,
) -> tuple[Array, Array, Array, Array]:
    """Hodgkin-Huxley RHS: returns (dV/dt, dm/dt, dh/dt, dn/dt).

    I_input = source + ξΦ (token sources + coupling from the CLEFPE field).
    """
    am, bm, ah, bh, an, bn = hh_rates(v)
    dv = (i_in
          - cfg.g_Na * m**3 * h * (v - cfg.E_Na)
          - cfg.g_K  * n**4     * (v - cfg.E_K)
          - cfg.g_L  *            (v - cfg.E_L)) / cfg.C_m
    dm = am * (1.0 - m) - bm * m
    dh = ah * (1.0 - h) - bh * h
    dn = an * (1.0 - n) - bn * n
    return dv, dm, dh, dn
