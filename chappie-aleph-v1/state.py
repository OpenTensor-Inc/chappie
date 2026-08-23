# chappie_aleph/state.py
import jax
import jax.numpy as jnp
from jax import tree_util
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from config import ModelConfig, PhysicsConfig

# ---------------------------------------------------------
# ChappieState as a JAX PyTree Dataclass
# ---------------------------------------------------------
@dataclass
class ChappieState:
    """
    حالت کامل سیستم.
    نکته کلیدی: فیلدهای داده (Arrays) باید در metadata(None) باشند
    و فیلدهای متادیتا (int, float, tuple) در metadata(static=True).
    """
    V: jax.Array                      # (N,) - Dynamic
    gates: Dict[str, jax.Array]       # Dict of Arrays - Dynamic
    
    # Static metadata (does not change during scan, not differentiated)
    t: float = field(default=0.0, metadata=dict(static=True))
    step: int = field(default=0, metadata=dict(static=True))
    # gate_names ذخیره نمی‌شود، از کلیدهای gates استخراج می‌شود

    # ---------------------------------------------------------
    # Properties / Helpers
    # ---------------------------------------------------------
    @property
    def N(self) -> int:
        return self.V.shape[0]
    
    @property
    def gate_names(self) -> Tuple[str, ...]:
        # سورت برای ترتیب ثابت (ضروری برای JAX)
        return tuple(sorted(self.gates.keys()))

    def get_gate(self, channel: str, gate: str) -> jax.Array:
        return self.gates[f"{channel}_{gate}"]

    def set_gate(self, channel: str, gate: str, value: jax.Array) -> 'ChappieState':
        # Immutable update for functional style
        new_gates = dict(self.gates)
        new_gates[f"{channel}_{gate}"] = value
        return ChappieState(V=self.V, gates=new_gates, t=self.t, step=self.step)

    def replace(self, **kwargs) -> 'ChappieState':
        """Helper for functional updates (like dataclasses.replace but works with JAX)."""
        return ChappieState(
            V=kwargs.get('V', self.V),
            gates=kwargs.get('gates', self.gates),
            t=kwargs.get('t', self.t),
            step=kwargs.get('step', self.step)
        )

# ---------------------------------------------------------
# Register as PyTree Node
# ---------------------------------------------------------
# fields: Dynamic arrays (V, gates dict values)
# static: t, step
tree_util.register_dataclass(
    ChappieState,
    data_fields=['V', 'gates'],
    meta_fields=['t', 'step']
)

# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------
def sigmoid_jax(x, V_half, k):
    return 1.0 / (1.0 + jnp.exp(-(x - V_half) / k))

def init_state(key: jax.Array, config: ModelConfig = None) -> ChappieState:
    """ایجاد حالت اولیه: V=Rest, Gates=SteadyState(V_rest)"""
    if config is None:
        from config import CONFIG
        config = CONFIG
        
    N = config.spatial.n_nodes
    physics = config.physics
    
    V_rest = jnp.full((N,), -65.0, dtype=jnp.float32)
    
    gates = {}
    for ch_cfg in physics.channel_configs:
        name = ch_cfg["name"]
        if "gate_m" in ch_cfg:
            m_inf = sigmoid_jax(V_rest, ch_cfg["gate_m"]["V_half"], ch_cfg["gate_m"]["k"])
            gates[f"{name}_m"] = m_inf
        if "gate_h" in ch_cfg:
            h_inf = sigmoid_jax(V_rest, ch_cfg["gate_h"]["V_half"], ch_cfg["gate_h"]["k"])
            gates[f"{name}_h"] = h_inf
        if "gate_n" in ch_cfg:
            n_inf = sigmoid_jax(V_rest, ch_cfg["gate_n"]["V_half"], ch_cfg["gate_n"]["k"])
            gates[f"{name}_n"] = n_inf
            
    return ChappieState(V=V_rest, gates=gates, t=0.0, step=0)