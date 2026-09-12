"""chappie.config — configuration loading and default.yaml path."""

from __future__ import annotations
from pathlib import Path

from chappie.core.config import (
    ChappieConfig, GridConfig, SolverConfig, ClefpeConfig,
    MemoryConfig, HHConfig, EncodingConfig, DecodingConfig,
    TrainingConfig, DataConfig, SamplingConfig,
    default_config, load_yaml_subset, dump_yaml_subset, DEFAULT_YAML,
)

# Absolute path to the shipped default.yaml
DEFAULT_YAML_PATH = str(Path(__file__).parent / "default.yaml")
