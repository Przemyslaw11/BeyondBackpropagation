"""Checkpoint manager with atomic replacement and compatibility metadata."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from ..contracts import CheckpointMetadata


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, corrupt, or incompatible."""


class CheckpointManager:
    FORMAT_VERSION = 1

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        filename: str,
        *,
        model_state: dict[str, Any],
        epoch: int,
        algorithm: str,
        optimizer_state: dict[str, Any] | None = None,
        best_metric_name: str | None = None,
        best_metric_value: float | None = None,
        config_hash: str | None = None,
    ) -> Path:
        metadata = CheckpointMetadata(
            format_version=self.FORMAT_VERSION,
            algorithm=algorithm,
            epoch=epoch,
            best_metric_name=best_metric_name,
            best_metric_value=best_metric_value,
            config_hash=config_hash,
        )
        payload = {
            "metadata": metadata.__dict__,
            "state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
        }
        target = self.directory / filename
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=self.directory
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise
        return target

    def load(self, filename: str, *, map_location: Any = "cpu") -> dict[str, Any]:
        path = self.directory / filename
        if not path.exists():
            raise CheckpointError(f"Checkpoint not found: {path}")
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except Exception as exc:
            raise CheckpointError(f"Could not load checkpoint {path}: {exc}") from exc
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise CheckpointError(f"Invalid checkpoint payload: {path}")
        metadata = payload.get("metadata", {})
        if metadata.get("format_version") != self.FORMAT_VERSION:
            raise CheckpointError(
                f"Unsupported checkpoint version {metadata.get('format_version')!r} in {path}"
            )
        return payload
