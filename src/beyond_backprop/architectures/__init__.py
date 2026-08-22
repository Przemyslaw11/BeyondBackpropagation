"""Architecture registry and fair BP baseline construction."""

from .factory import (
    ARCHITECTURE_REGISTRY,
    ArchitectureRegistry,
    build_fair_bp_baseline,
    build_model,
)

__all__ = [
    "ARCHITECTURE_REGISTRY",
    "ArchitectureRegistry",
    "build_fair_bp_baseline",
    "build_model",
]
