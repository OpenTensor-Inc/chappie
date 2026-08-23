# chappie_aleph/physics/hodgkin_huxley.py
from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from config import PhysicsConfig
from state import ChappieState

# ---------------------------------------------------------
# Gate Kinetics: Analytical Steady State & Time Constants
# تستخدم در Rush-Larsen / Exponential Integrator
# ---------------------------------------------------------
def gate_steady_state(V: jax.Array, V_half: float, k: float) -> jax.Array:
    """m_inf(V) = 1 / (1 + exp(-(V - V_half)/k))"""
    return 1.0 / (1.0 + jnp.exp(-(V - V_half) / k))

def gate_time_constant(V: jax.Array, tau_min: float, tau_max: float, V_tau: float, k_tau: float) -> jax.Array:
    """
    مدل پیشرفته τ(V) = tau_min + (tau_max - tau_min) * sech^2((V - V_tau)/k_tau)
    یا ساده: ثابت. اینجا انعطاف‌پذیر پیاده‌سازی شده.
    """
    # مثال: تابع پیچ 부르ه‌ای برای tau
    return tau_min + (tau_max - tau_min) / jnp.cosh((V - V_tau) / k_tau)**2

# ---------------------------------------------------------
# Ionic Currents Calculation (Vectorized over Channels)
# ---------------------------------------------------------
def compute_ionic_currents(V: jax.Array, gates: Dict[str, jax.Array], physics: PhysicsConfig) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    محاسبه جریان‌های ینی کل و مشتقات گیت‌ها.
    برمی‌گرداند: (I_ion_total, dgates_dict)
    """
    I_total = jnp.zeros_like(V)
    dgates = {}
    
    for ch_cfg in physics.channel_configs:
        name = ch_cfg["name"]
        g_max = ch_cfg["g_max"]
        E_rev = ch_cfg["E_rev"]
        
        # Conduction variable = product of gates^power
        # m^p * h^q ...
        cond = jnp.ones_like(V)
        
        # Update Gates (Rush-Larsen Style: Exact solution for linear gating ODE)
        # dg/dt = (g_inf(V) - g) / tau(V)
        # g_new = g_inf + (g_old - g_inf) * exp(-dt / tau)
        # در اینجا فقط RHS (dg/dt) را برمی‌گردانیم تا Solver عمومی استفاده کند
        # اما برای ثبات بالا، Solver ما از Rush-Larsen در Kernel صریح استفاده می‌کند.
        # این تابع برای محاسبه RHS عمومی (Explicit Part) است.
        
        for gate_key, gate_cfg in ch_cfg.items():
            if not gate_key.startswith("gate_"): continue
            
            g_name = gate_key.replace("gate_", "") # m, h, n
            full_name = f"{name}_{g_name}"
            g_val = gates[full_name]
            
            g_inf = gate_steady_state(V, gate_cfg["V_half"], gate_cfg["k"])
            tau = gate_time_constant(V, 
                                     gate_cfg["tau_min"], gate_cfg["tau_max"],
                                     gate_cfg.get("V_tau", 0.0), gate_cfg.get("k_tau", 1.0))
            
            # RHS for dg/dt
            dgates[full_name] = (g_inf - g_val) / tau
            
            # Contribution to conduction
            power = gate_cfg["power"]
            cond = cond * (g_val ** power)
            
        # Current for this channel
        I_ch = g_max * cond * (V - E_rev)
        I_total += I_ch
        
    return I_total, dgates

# ---------------------------------------------------------
# Rush-Larsen Exact Update Kernel (برای استفاده در Fused Explicit Kernel)
# این را در numerics/spatial_ops یا یک kernel جداگانه می‌ذاریم
# ---------------------------------------------------------
def rush_larsen_update(g_old: jax.Array, V: jax.Array, dt: float, gate_cfg: dict) -> jax.Array:
    """Analytical update for gate: g_new = g_inf + (g_old - g_inf) * exp(-dt/tau)"""
    g_inf = gate_steady_state(V, gate_cfg["V_half"], gate_cfg["k"])
    tau = gate_time_constant(V, 
                             gate_cfg["tau_min"], gate_cfg["tau_max"],
                             gate_cfg.get("V_tau", 0.0), gate_cfg.get("k_tau", 1.0))
    return g_inf + (g_old - g_inf) * jnp.exp(-dt / tau)