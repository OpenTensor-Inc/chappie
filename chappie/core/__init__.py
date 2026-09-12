"""chappie.core — CLEFPE field dynamics, HH, memory, grid, state."""

from chappie.core.config import (
    ChappieConfig, GridConfig, SolverConfig, ClefpeConfig,
    MemoryConfig, HHConfig, EncodingConfig, DecodingConfig,
    TrainingConfig, DataConfig, SamplingConfig,
    default_config, load_yaml_subset, dump_yaml_subset,
)
from chappie.core.field import Grid3D
from chappie.core.state import ChappieState, init_rest, reset_activity
from chappie.core.hh import hh_rates, gating_steady_state, hh_rhs
from chappie.core.memory import memory_rhs
from chappie.core.clefpe import clefpe_rhs, potential_force
from chappie.core.diffusion import diffusion_rhs
from chappie.core.wave import wave_terms
from chappie.core.energy import energy_components, total_energy
