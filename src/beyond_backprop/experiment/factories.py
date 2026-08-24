"""Factories used by experiment and tuning entry points."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..algorithms import build_algorithm
from ..architectures import build_model
from ..config import ExperimentConfig, load_experiment_config
from ..training import ExperimentRunner


def load_config(
    path: str | Path,
    *,
    base_config: str | Path = "configs/base.yaml",
    overrides: tuple[str, ...] = (),
) -> ExperimentConfig:
    return load_experiment_config(path, base_config, overrides)


def build_algorithm_factory(name: str):
    return lambda: build_algorithm(name)


def build_runner(**kwargs: Any) -> ExperimentRunner:
    return ExperimentRunner(**kwargs)


def build_model_for_config(
    config: ExperimentConfig | Mapping[str, Any], device: Any
) -> Any:
    return build_model(config, device)


__all__ = [
    "build_algorithm_factory",
    "build_model_for_config",
    "build_runner",
    "load_config",
]
