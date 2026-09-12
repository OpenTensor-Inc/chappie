"""Chappie configuration: dataclasses + a tiny YAML-subset loader.

Rationale: PyYAML is not a dependency of this project, so ``default.yaml`` is
parsed with a small, documented YAML-subset reader (block maps, ``key: value``,
flow lists ``[a, b, c]``, scalars, ``#`` comments).  Everything in
``config/default.yaml`` maps 1:1 onto the dataclasses below.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, get_type_hints

# ---------------------------------------------------------------------------
# Tiny YAML-subset parser / dumper
# ---------------------------------------------------------------------------


def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if s in ("", "null", "~", "None"):
        return None
    if s in ("true", "True", "yes"):
        return True
    if s in ("false", "False", "no"):
        return False
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x) for x in inner.split(",")]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def load_yaml_subset(text: str) -> dict[str, Any]:
    """Parse a YAML subset into nested dicts.

    Supported: indentation nesting, ``key: value``, flow lists, comments.
    Unsupported (raises): anchors/aliases, ``- item`` block lists, multiline
    strings.  This is intentional — the shipped config avoids those features.
    """
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.strip()))
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for indent, content in lines:
        if content.startswith("- "):
            raise ValueError(
                "block lists ('- item') are not supported; use flow lists [a, b]"
            )
        if ":" not in content:
            raise ValueError(f"expected 'key: value', got {content!r}")
        key, _, val = content.partition(":")
        key = key.strip()
        val = val.strip()
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(val)
    return root


def _dump_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_dump_scalar(x) for x in v) + "]"
    return str(v)


def dump_yaml_subset(d: dict[str, Any], indent: int = 0) -> str:
    pad = "  " * indent
    out: list[str] = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(f"{pad}{k}:")
            out.append(dump_yaml_subset(v, indent + 1))
        else:
            out.append(f"{pad}{k}: {_dump_scalar(v)}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Config dataclasses  (every parameter of the spec §24 is configurable)
# ---------------------------------------------------------------------------


@dataclass
class GridConfig:
    n1: int = 16
    n2: int = 16
    n3: int = 16
    length: float = 1.0
    boundary: str = "periodic"  # periodic | dirichlet | neumann | absorbing
    dtype: str = "float32"      # float32 | float64 | bfloat16


@dataclass
class SolverConfig:
    type: str = "imex"          # imex | rk4
    dt: float = 0.01
    n_steps_per_token: int = 8
    cfl_safety: float = 0.5
    max_halvings: int = 6
    clip_phi: float = 1.0       # 0 disables clipping
    energy_normalize: bool = False


@dataclass
class ClefpeConfig:
    c: float = 1.2              # wave speed
    D: float = 0.05             # diffusion of Φ_t
    gamma: float = 0.8          # damping of the CLEFPE oscillator
    kappa: float = 0.5          # HH → CLEFPE coupling
    V0: float = -65.0           # HH rest potential
    lam_R: float = 0.1          # nonlinear potential strength
    a: float = -0.5             # quadratic potential coefficient
    b: float = 0.25             # quartic potential coefficient


@dataclass
class MemoryConfig:
    rho: tuple[float, ...] = (0.02, 0.005)   # memory diffusion per field
    mu: tuple[float, ...] = (0.3, 0.02)      # memory decay per field
    eta: tuple[float, ...] = (0.4, 0.2)      # Φ → memory coupling per field
    chi: tuple[float, ...] = (0.6, 0.4)      # memory → CLEFPE coupling per field

    @property
    def n_fields(self) -> int:
        return len(self.rho)


@dataclass
class HHConfig:
    C_m: float = 1.0
    xi: float = 0.05            # Φ → HH coupling (I_input += ξΦ)
    g_Na: float = 120.0
    g_K: float = 36.0
    g_L: float = 0.3
    E_Na: float = 50.0
    E_K: float = -77.0
    E_L: float = -54.4
    rest: float = -65.0


@dataclass
class EncodingConfig:
    basis: str = "gaussian"     # gaussian | fourier | wavelet
    sigma: float = 0.4          # Gaussian/wavelet width (grid units)
    amp: float = 1.0
    pulse_width: float = 0.5    # fraction of the per-token window the source is on
    init_mode: str = "sum_normalized"  # sum of bumps, L∞-normalized
    context_weights: str = "overlap"   # dynamics-derived weights (not attention)


@dataclass
class DecodingConfig:
    tau: float = 0.5            # softmax temperature
    read_modalities: tuple[str, ...] = ("phi",)  # phi | mem | v
    bias: bool = True           # learned per-token bias
    n_modes: int = 48           # FourierBasis modes / spectral readout size


@dataclass
class TrainingConfig:
    vocab_size: int = 96
    seq_len: int = 32
    batch_size: int = 1
    steps: int = 120
    lr: float = 2e-3
    n_unroll: int = 4           # short-unroll length for the tiny-param gradients
    teacher_forcing: bool = True
    force_gain: float = 0.5     # CLEFPE learning-force strength (0 disables)
    param_every: int = 2        # update tiny params every N steps
    val_every: int = 20
    checkpoint_every: int = 50
    ckpt_dir: str = "artifacts/ckpt"
    seed: int = 0


@dataclass
class DataConfig:
    source: str = "synthetic"   # synthetic | text
    tokenizer: str = "char"     # char | bpe
    text_path: str = "data/tiny_corpus.txt"
    curriculum: bool = True
    max_level: int = 4


@dataclass
class SamplingConfig:
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 1.0


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class ChappieConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    clefpe: ClefpeConfig = field(default_factory=ClefpeConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    hh: HHConfig = field(default_factory=HHConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # -- conversion --------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChappieConfig":
        return _from_dict(cls, d)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    # -- YAML --------------------------------------------------------------

    @classmethod
    def from_yaml_text(cls, text: str) -> "ChappieConfig":
        return cls.from_dict(load_yaml_subset(text))

    @classmethod
    def from_yaml(cls, path: str) -> "ChappieConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_yaml_text(f.read())

    def to_yaml(self) -> str:
        return dump_yaml_subset(self.to_dict())

    # -- overrides ---------------------------------------------------------

    def with_overrides(self, overrides: dict[str, Any]) -> "ChappieConfig":
        """Apply dotted-key overrides, e.g. ``{"training.lr": 1e-3}``."""
        import copy

        new = copy.deepcopy(self)
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            node = new
            for p in parts[:-1]:
                node = getattr(node, p)
            setattr(node, parts[-1], value)
        return new


def _from_dict(cls: type, d: dict[str, Any]):
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        t = hints.get(f.name)
        if is_dataclass(t):
            kwargs[f.name] = _from_dict(t, d[f.name])
        else:
            kwargs[f.name] = d[f.name]
    return cls(**kwargs)


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    return obj


# The embedded default (also written to config/default.yaml).
DEFAULT_YAML = """\
# Chappie v1 default configuration (CLEFPE field language model)
grid:
  n1: 16
  n2: 16
  n3: 16
  length: 1.0
  boundary: periodic
  dtype: float32
solver:
  type: imex
  dt: 0.01
  n_steps_per_token: 8
  cfl_safety: 0.5
  max_halvings: 6
  clip_phi: 1.0
  energy_normalize: false
clefpe:
  c: 1.2
  D: 0.05
  gamma: 0.8
  kappa: 0.5
  V0: -65.0
  lam_R: 0.1
  a: -0.5
  b: 0.25
memory:
  rho: [0.02, 0.005]
  mu: [0.3, 0.02]
  eta: [0.4, 0.2]
  chi: [0.6, 0.4]
hh:
  C_m: 1.0
  xi: 0.05
  g_Na: 120.0
  g_K: 36.0
  g_L: 0.3
  E_Na: 50.0
  E_K: -77.0
  E_L: -54.4
  rest: -65.0
encoding:
  basis: gaussian
  sigma: 0.4
  amp: 1.0
  pulse_width: 0.5
  init_mode: sum_normalized
  context_weights: overlap
decoding:
  tau: 0.5
  read_modalities: [phi]
  bias: true
  n_modes: 48
training:
  vocab_size: 96
  seq_len: 32
  batch_size: 1
  steps: 120
  lr: 0.002
  n_unroll: 4
  teacher_forcing: true
  force_gain: 0.5
  param_every: 2
  val_every: 20
  checkpoint_every: 50
  ckpt_dir: artifacts/ckpt
  seed: 0
data:
  source: synthetic
  tokenizer: char
  text_path: data/tiny_corpus.txt
  curriculum: true
  max_level: 4
sampling:
  temperature: 0.8
  top_k: 0
  top_p: 1.0
"""


def default_config() -> ChappieConfig:
    return ChappieConfig.from_yaml_text(DEFAULT_YAML)
