"""CLEFPE learning force (spec §7/§10).

The analytic functional gradient:

  F_learning = (gain/τ) · (ψ̄_{y*} − Σ_y P(y) ψ̄_y)

This drives the field Φ toward the target token's basis function and
away from the current average prediction — a contrastive field force.
Only active during teacher-forced training; zero during inference.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from chappie.core.config import ChappieConfig
from chappie.core.field import Grid3D
from chappie.encoding.basis import BasisParams


def learning_force(
    phi: Array,
    target_id: Array,
    probs: Array,
    basis,
    p: BasisParams,
    grid: Grid3D,
    cfg: ChappieConfig,
) -> Array:
    """Compute the CLEFPE learning force field.

    Args:
        phi:       (n1, n2, n3) — current field
        target_id: scalar int32 — target token
        probs:     (V,) — current softmax distribution
        basis:     basis object with bump_norm and expectation_field
        p:         BasisParams
        grid:      Grid3D
        cfg:       ChappieConfig

    Returns:
        force field (n1, n2, n3), same shape as phi
    """
    if cfg.training.force_gain <= 0:
        return jnp.zeros_like(phi)
    psi_target = basis.bump_norm(target_id, p)
    exp_field = basis.expectation_field(probs, p)
    return cfg.training.force_gain * (psi_target - exp_field) / cfg.decoding.tau
