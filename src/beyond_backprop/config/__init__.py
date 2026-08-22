"""Typed configuration loading and validation."""

from .loader import (
    ConfigValidationError,
    apply_overrides,
    experiment_config_from_mapping,
    load_experiment_config,
    load_mapping,
    normalize_legacy_config,
    resolved_config_hash,
    save_resolved_config,
)
from .models import (
    AlgorithmName,
    ArchitectureName,
    BackendName,
    DatasetName,
    ExperimentConfig,
)

__all__ = [
    "AlgorithmName",
    "ArchitectureName",
    "BackendName",
    "ConfigValidationError",
    "apply_overrides",
    "DatasetName",
    "ExperimentConfig",
    "experiment_config_from_mapping",
    "load_experiment_config",
    "load_mapping",
    "normalize_legacy_config",
    "resolved_config_hash",
    "save_resolved_config",
]
