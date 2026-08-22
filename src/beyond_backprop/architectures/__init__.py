"""Architecture registry and fair BP baseline construction."""

from .factory import ArchitectureRegistry, build_fair_bp_baseline, build_model

__all__ = ["ArchitectureRegistry", "build_fair_bp_baseline", "build_model"]
