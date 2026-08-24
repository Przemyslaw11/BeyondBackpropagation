"""Device resolution with the legacy local/SLURM policy preserved."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..config.models import BackendName
from .backend_policy import get_execution_backend


def resolve_device(config: Mapping[str, Any]) -> torch.device:
    general = config.get("general", {})
    backend = get_execution_backend(dict(config))
    return backend.resolve_device(str(general.get("device", "auto")))


def backend_name(config: Mapping[str, Any]) -> BackendName:
    return BackendName.parse(config.get("general", {}).get("backend", "slurm"))
