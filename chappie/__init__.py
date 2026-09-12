"""Chappie — Continuous Non-linear Tensor-Free Architecture (spec §27).

A language model using 3D field dynamics (CLEFPE + HH + memory)
instead of Transformer attention and stored weight matrices.
"""

__version__ = "1.0.0"

from chappie.model import Chappie, ChappieParams
from chappie.training.trainer import ChappieTrainer
from chappie.core.config import ChappieConfig, default_config
from chappie.core.field import Grid3D
from chappie.core.state import ChappieState, init_rest
from chappie.inference.generate import generate
