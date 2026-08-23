# chappie_aleph/solver.py
import jax
import jax.numpy as jnp
from state import ChappieState, init_state
from config import ModelConfig, CONFIG
from time_stepping.imex_rk import imex_ars23_step

@jax.jit
def simulate_step(state: ChappieState, config: ModelConfig = CONFIG) -> ChappieState:
    """یک گام کامل شبیه‌سازی (IMEX Step)"""
    return imex_ars23_step(state, config.solver.dt, config)

@jax.jit
def simulate_trajectory(init_state: ChappieState, config: ModelConfig = CONFIG) -> ChappieState:
    """اجرای تمام گام‌های زمانی (Scan Loop)"""
    def scan_fn(carry, _):
        new_state = simulate_step(carry, config)
        return new_state, new_state.V # یا ذخیره چک‌پوینت‌ها
    
    final_state, _ = jax.lax.scan(scan_fn, init_state, jnp.arange(config.solver.n_steps))
    return final_state

# ---------------------------------------------------------
# I/O Interface (Token <-> Field)
# ---------------------------------------------------------
def inject_tokens(state: ChappieState, token_ids: jax.Array, embed_matrix: jax.Array, positions: jax.Array) -> ChappieState:
    # ... logic ...
    V = state.V.at[positions].add(embed_matrix[token_ids].sum(axis=-1))
    return state.replace(V=V) # Use replace

def readout_logits(state: ChappieState, probe_matrix: jax.Array, probe_positions: jax.Array) -> jax.Array:
    """
    خواندن لاگیت‌ها از میدان.
    probe_matrix: (Vocab, N_Probes) - وزن‌های خطی خروجی
    probe_positions: (N_Probes,) - مکان‌های پروب
    """
    V_probes = state.V[probe_positions] # (N_Probes,)
    logits = probe_matrix @ V_probes    # (Vocab,)
    return logits