"""Compatibility shim: canonical implementations live in beyond_backprop."""

from __future__ import annotations

from beyond_backprop.utils.training_support import (
    log_metrics,
    setup_logging,
    setup_wandb,
)

__all__ = [
    "log_metrics",
    "setup_logging",
    "setup_wandb",
]
