"""Canonical algorithm adapter registry."""

from .base import AlgorithmAdapter
from .bp import BPAdapter
from .cafo import CaFoAdapter
from .cafo_math import aggregate_predictor_outputs, predictor_cross_entropy
from .ff import FFAdapter
from .ff_math import aggregate_goodness, generate_hinton_inputs, linear_cooldown_lr
from .mf import MFAdapter
from .mf_math import local_cross_entropy, projection_logits
from .registry import (
    ALGORITHM_REGISTRY,
    AlgorithmRegistry,
    build_algorithm,
    get_algorithm,
)

__all__ = [
    "ALGORITHM_REGISTRY",
    "AlgorithmAdapter",
    "AlgorithmRegistry",
    "BPAdapter",
    "CaFoAdapter",
    "FFAdapter",
    "MFAdapter",
    "aggregate_goodness",
    "aggregate_predictor_outputs",
    "build_algorithm",
    "generate_hinton_inputs",
    "get_algorithm",
    "linear_cooldown_lr",
    "local_cross_entropy",
    "predictor_cross_entropy",
    "projection_logits",
]
