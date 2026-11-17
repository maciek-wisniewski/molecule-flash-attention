"""Flash attention with structural biases, masks, multipliers and gates computed in the kernel.

See ``docs/guide.md`` for how the pieces fit together and ``docs/recipes.md`` for worked
examples.
"""

from . import layers, nn, reference, terms
from .checks import MolflashDebugError, debug
from .ops import attention, attention_varlen
from .primitives import (
    EdgeBias,
    EdgeMask,
    NodeBias,
    PairBias,
    PairFunction,
    PairGate,
    PairMask,
    PairScale,
    pair_function,
)
from .sampling import BackwardSampling

__version__ = "0.1.0"

__all__ = [
    "BackwardSampling",
    "EdgeBias",
    "EdgeMask",
    "NodeBias",
    "PairBias",
    "PairFunction",
    "PairGate",
    "PairMask",
    "PairScale",
    "MolflashDebugError",
    "attention",
    "attention_varlen",
    "debug",
    "layers",
    "nn",
    "pair_function",
    "reference",
    "terms",
]
