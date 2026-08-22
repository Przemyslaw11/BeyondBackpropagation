"""Canonical algorithm adapter registry."""

from .base import AlgorithmAdapter
from .bp import BPAdapter
from .cafo import CaFoAdapter
from .ff import FFAdapter
from .mf import MFAdapter
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
    "build_algorithm",
    "get_algorithm",
]
