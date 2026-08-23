"""Compatibility shim: canonical implementations live in beyond_backprop."""

from __future__ import annotations

from beyond_backprop.runtime import set_seed
from beyond_backprop.utils.training_support import (
    create_directory_if_not_exists,
    format_time,
    save_checkpoint,
)

__all__ = [
    "create_directory_if_not_exists",
    "format_time",
    "save_checkpoint",
    "set_seed",
]
