"""Lazy construction of experiment tracking services."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config.models import ExperimentConfig
from .local import LocalFileTracker
from .noop import NoOpTracker
from .wandb import WandbTracker


def build_tracker(
    config: ExperimentConfig | Mapping[str, Any],
    *,
    directory: str | Path,
):
    """Select no-op, local, or W&B tracking without importing W&B eagerly."""

    mapping = (
        config.to_mapping() if isinstance(config, ExperimentConfig) else dict(config)
    )
    tracking = mapping.get("tracking", {})
    logging_wandb = mapping.get("logging", {}).get("wandb", {})
    enabled = tracking.get("enabled")
    if enabled is None:
        enabled = bool(logging_wandb.get("use_wandb", False))
    if not enabled:
        return NoOpTracker()

    mode = str(tracking.get("mode", logging_wandb.get("mode", "local"))).lower()
    if mode == "wandb" or tracking.get(
        "use_wandb", logging_wandb.get("use_wandb", False)
    ):
        settings = dict(logging_wandb)
        settings.update(dict(tracking))
        settings.pop("enabled", None)
        return WandbTracker(settings)
    return LocalFileTracker(directory)


__all__ = ["build_tracker"]
