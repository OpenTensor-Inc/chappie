# chappie_aleph/adjoint/hybrid_vjp.py
import jax
from jax import custom_vjp
from solver import simulate_trajectory
from state import ChappieState
from config import ModelConfig

# ---------------------------------------------------------
# Custom VJP برای Hybrid Adjoint
# Forward: simulate_trajectory (با Checkpointing)
# Backward: 
#   1. Explicit Parts (Wave, HH) -> Recompute from Checkpoints + Reverse AD
#   2. Implicit Part (Diffusion)   -> Analytical Adjoint (Transpose Linear Solve)
# ---------------------------------------------------------

@custom_vjp
def chappie_forward(params, init_state: ChappieState, config: ModelConfig):
    # params: dict of learnable fields (D_field, c_field, HH_params)
    # در JAX پارامترها باید آرایه‌های مسطح باشند.
    return simulate_trajectory(init_state, config)

def chappie_forward_fwd(params, init_state, config):
    # Forward pass with Checkpointing
    # ما باید حالت‌های میانی را برای Recompute ذخیره کنیم.
    # jax.lax.scan با checkpointing داخلی (jax.checkpoint یا manual)
    
    # برای سادگی فعلاً بدون چک‌پوینتینگ (خطر OOM برای n_steps زیاد)
    final_state = simulate_trajectory(init_state, config)
    # ذخیره موارد مورد نیاز برای backward
    residuals = (params, init_state, config, final_state) 
    return final_state, residuals

def chappie_forward_bwd(residuals, grad_output):
    params, init_state, config, final_state = residuals
    # TODO: Implement Hybrid Adjoint Logic Here
    # 1. Backward pass through time (Reverse Scan)
    # 2. At each step:
    #    a. Explicit (Wave/HH): Standard Reverse AD (using saved checkpoints or recompute)
    #    b. Implicit (Diffusion): Solve Adjoint Linear System (Transpose of Thomas/PCR)
    # 3. Accumulate gradients for params (D_field, c_field, HH scalars)
    
    raise NotImplementedError("Hybrid Adjoint Implementation Required - Next Step")
    
chappie_forward.defvjp(chappie_forward_fwd, chappie_forward_bwd)