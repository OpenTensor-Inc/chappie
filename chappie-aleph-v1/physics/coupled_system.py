# chappie_aleph/physics/coupled_system.py
import jax
import jax.numpy as jnp
from state import ChappieState
from config import ModelConfig
from physics.hodgkin_huxley import compute_ionic_currents
from numerics.spatial_ops import diffusion_rhs, wave_rhs_upwind

def split_state_for_imex(state: ChappieState, config: ModelConfig):
    """
    تفکیک حالت برای IMEX:
    Implicit Part (Y_imp): V (چون دیفیوژن روی V ضمنی است)
    Explicit Part (Y_exp): V, Gates (موج و HH صریح‌اند)
    
    در IMEX معمولاً سیستم را به صورت زیر می‌نویسیم:
    dY/dt = F_impl(Y) + F_exp(Y)
    """
    return state.V, state # Implicit gets V, Explicit gets full state

def rhs_implicit(V: jax.Array, state: ChappieState, config: ModelConfig) -> jax.Array:
    """
    بخش ضمنی: فقط دیفیوژن روی V.
    dV/dt = ∇·(D ∇V)
    """
    D_field = config.physics.D_base # یا state.D_field اگر یادگیری‌پذیر باشد
    return diffusion_rhs(V, D_field, config.spatial)

def rhs_explicit(V: jax.Array, state: ChappieState, config: ModelConfig) -> tuple[jax.Array, dict]:
    """
    بخش صریح: موج + دینامیک HH
    برمی‌گرداند: (dV/dt_explicit, dgates/dt)
    """
    physics = config.physics
    
    # 1. Wave Term
    c_field = physics.c_base # یا state.c_field
    dV_wave = wave_rhs_upwind(V, c_field, config.spatial)
    
    # 2. HH Terms
    I_ion, dgates = compute_ionic_currents(V, state.gates, physics)
    dV_hh = -I_ion / physics.Cm
    
    dV_exp = dV_wave + dV_hh
    
    return dV_exp, dgates

# ---------------------------------------------------------
# Full RHS for Non-IMEX solvers (e.g., Explicit RK4, Dopri5)
# ---------------------------------------------------------
def full_rhs(t: float, flat_state: jax.Array, config: ModelConfig) -> jax.Array:
    """Wrapper برای Solverهای عمومی (مثل diffrax یا خودمان)"""
    from state import ChappieState
    gate_names = tuple(sorted(config.physics.channel_configs[0].keys())) # Hack: need gate names list
    # بهتر است gate_names در config باشد
    N = config.spatial.n_nodes
    state = ChappieState.unflatten(flat_state, gate_names, N)
    
    V = state.V
    dV_imp = rhs_implicit(V, state, config)
    dV_exp, dgates = rhs_explicit(V, state, config)
    
    dV = dV_imp + dV_exp
    
    # Flatten derivatives
    derivs = [dV] + [dgates[k] for k in state.gate_names]
    return jnp.concatenate(derivs)