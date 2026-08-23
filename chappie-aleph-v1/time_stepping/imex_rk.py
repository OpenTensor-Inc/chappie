# chappie_aleph/time_stepping/imex_rk.py
import jax
import jax.numpy as jnp
from config import ModelConfig, SpatialConfig
# State import is now safe
from state import ChappieState
from physics.coupled_system import rhs_implicit, rhs_explicit
from numerics.linear_solvers import implicit_diffusion_step, solve_implicit_system

GAMMA = 1.0 - 1.0 / jnp.sqrt(2.0)

def imex_ars23_step(state: ChappieState, dt: float, config: ModelConfig) -> ChappieState:
    V = state.V
    gates = state.gates
    physics = config.physics
    spatial = config.spatial
    
    D_field = physics.D_base 
    Cm = physics.Cm
    
    # --- Stage 1 ---
    # Implicit Solve for V1
    V1 = implicit_diffusion_step(V, dt * GAMMA, D_field, spatial)
    
    # State1: V updated, Gates same, t/step updated
    # Use .replace() for functional update
    state1 = state.replace(V=V1, t=state.t + GAMMA * dt, step=state.step) # step usually increments at end
    
    # Explicit RHS at Stage 1 (using V_n for explicit part of stage 1? 
    # ARS Tableau: Explicit stage 1 is 0. So dV_exp_1 uses U_n (V))
    dV_exp_1, dgates_1 = rhs_explicit(V, state, config) 
    
    # Gates don't change in Stage 1 for Explicit part (a_E[0,0]=0)
    # So state1 gates remain same as state.gates

    # --- Stage 2 ---
    # RHS at Stage 1 (for Explicit part of Stage 2)
    dV_exp_1_s2, dgates_1_s2 = rhs_explicit(V1, state1, config) # Uses V1
    
    # RHS Implicit at Stage 1 (Diffusion)
    dV_imp_1 = rhs_implicit(V1, state1, config)
    
    # Predictor for V2 RHS
    rhs_V2 = V + dt * (1.0 - GAMMA) * (dV_imp_1 + dV_exp_1_s2)
    
    # Solve Implicit for V2
    V2 = solve_implicit_system(rhs_V2, dt * GAMMA, D_field, spatial)
    
    # Update Gates for Stage 2 (Explicit)
    # g2 = g_n + dt * (1-gamma) * dgates_1
    new_gates = {}
    for name in gates:
        new_gates[name] = gates[name] + dt * (1.0 - GAMMA) * dgates_1_s2[name]
    
    # Final State (Stiffly Accurate: Stage 2 = Solution)
    final_state = state.replace(
        V=V2, 
        gates=new_gates, 
        t=state.t + dt, 
        step=state.step + 1
    )
    
    return final_state

def solve_implicit_system(rhs: jax.Array, alpha: float, D_field: jax.Array, config: SpatialConfig) -> jax.Array:
    """Solves (I - alpha * L) V = rhs"""
    dx = config.dx
    N = rhs.shape[0]
    D = D_field
    
    main_diag = 1.0 + 2.0 * alpha * D / (dx*dx)
    off_diag = -alpha * D / (dx*dx)
    
    # BC Neumann
    main_diag = main_diag.at[0].set(1.0 + alpha * D[0] / (dx*dx))
    main_diag = main_diag.at[-1].set(1.0 + alpha * D[-1] / (dx*dx))
    lower_diag = off_diag.at[0].set(0.0)
    upper_diag = off_diag.at[-1].set(0.0)
    
    # Call PCR/Thomas
    from numerics.linear_solvers import pcr_tridiagonal_solve
    return pcr_tridiagonal_solve(lower_diag, main_diag, upper_diag, rhs)