# chappie_aleph/config.py
from dataclasses import dataclass
from typing import Tuple, Literal
import jax.numpy as jnp

@dataclass(frozen=True)
class SpatialConfig:
    """تنظیمات فضایی - یک بعدی برای توالی (Sequence)"""
    domain_length: float = 1.0      # طول دامنه معمولی [0, 1]
    n_nodes: int = 8192             # تعداد نقاط فضایی (N) -> Context Length Proxy
    # dx محاسبه می‌شود
    @property
    def dx(self) -> float:
        return self.domain_length / (self.n_nodes - 1)

@dataclass(frozen=True)
class PhysicsConfig:
    """پارامترهای فیزیکی قابل یادگیری (Learnable Fields/Scalars)"""
    # --- Diffusion (Local Context) ---
    D_base: float = 0.01            # ضرایب دیفیوژن پایه (Scalar یا Field)
    D_learnable: bool = True        # آیا D فیلد فضایی یادگیری‌پذیر است؟
    
    # --- Wave/Transport (Global Context) ---
    c_base: float = 1.0             # سرعت موج پایه
    c_learnable: bool = True        # فیلد سرعت موج یادگیری‌پذیر؟
    wave_type: Literal["hyperbolic_first", "hyperbolic_second"] = "hyperbolic_first"
    # first:  dx/dt + c * grad(V) = 0  (Transport)
    # second: d2x/dt2 = c^2 * laplacian(V) (Wave Eq) -> نیاز به حالت قبلی V_{t-1}

    # --- Generalized Hodgkin-Huxley (Computation/Non-linearity) ---
    # پارامترهای اسکالر گلوبال (Shared across space) یا فیلد (Per-node)
    # اینها Weight های اصلی مدل هستند (تعداد بسیار کم)
    Cm: float = 1.0                 # خازن غشا
    
    # کانال‌ها: نام -> پارامترها
    # ما از یک دیکشنری dataclass برای انعطاف استفاده می‌کنیم
    # فرمت: g_max, E_rev, V_half_act, k_act, tau_base_act, V_half_inact, k_inact, tau_base_inact, power
    
    # تنظیمات پیش‌فرض برای ۳ کانال کلاسیک + ۱ کانال Leak
    channel_configs: Tuple[dict, ...] = (
        {   # Fast Sodium (Na) - Spike Initiation / Non-linear Switch
            "name": "Na", "g_max": 120.0, "E_rev": 50.0, 
            "gate_m": {"V_half": -40.0, "k": 5.0, "tau_min": 0.01, "tau_max": 0.1, "power": 3},
            "gate_h": {"V_half": -65.0, "k": -5.0, "tau_min": 1.0, "tau_max": 10.0, "power": 1},
        },
        {   # Delayed Rectifier Potassium (K) - Repolarization / Memory Decay
            "name": "K", "g_max": 36.0, "E_rev": -77.0,
            "gate_n": {"V_half": -55.0, "k": 5.0, "tau_min": 1.0, "tau_max": 5.0, "power": 4},
        },
        {   # Leak - Stabilization / Bias
            "name": "L", "g_max": 0.3, "E_rev": -54.3,
            # No gates, pure Ohmic
        },
        {   # Custom "Compute" Channel (e.g., Calcium or h-current) - Learnable Logic
            "name": "Comp", "g_max": 5.0, "E_rev": 120.0,
            "gate_m": {"V_half": -30.0, "k": 3.0, "tau_min": 5.0, "tau_max": 50.0, "power": 2},
            "gate_h": {"V_half": -70.0, "k": -4.0, "tau_min": 50.0, "tau_max": 200.0, "power": 1},
        }
    )

@dataclass(frozen=True)
class SolverConfig:
    """تنظیمات حل عددی"""
    dt: float = 0.01                # قدم زمانی (باید CFL برقرار باشد: dt < dx / c_max)
    n_steps: int = 100              # عمق محاسبه (Equivalent to Layers)
    imex_scheme: Literal["ARS23", "SSP33", "CN_ADI"] = "ARS23" # IMEX Runge-Kutta scheme
    
    # Implicit Solver Tolerance
    implicit_tol: float = 1e-6
    implicit_max_iter: int = 50     # برای PCR/Thomas معمولاً ۱.iter هست اما برای غیرخطی/متغیر ضریب.Iter می‌خواهد
    
    # Checkpointing for Adjoint
    checkpoint_every: int = 4       # ذخیره حالت هر K گام برای Backward Recompute

@dataclass(frozen=True)
class ModelConfig:
    spatial: SpatialConfig = SpatialConfig()
    physics: PhysicsConfig = PhysicsConfig()
    solver: SolverConfig = SolverConfig()
    
    # Vocab / I/O
    vocab_size: int = 256           # Byte-level (0-255)
    embed_dim: int = 256            # بعد امبدینگ (اگر از Linear Projection برای Injection استفاده کنیم)
    n_probe_points: int = 64        # نقاط خواندن خروجی (برای Logits)

# Singleton Instance
CONFIG = ModelConfig()