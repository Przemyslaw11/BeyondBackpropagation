"""Checkpoint manager with atomic replacement and compatibility metadata."""

from __future__ import annotations

import contextlib
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
    LEGACY_FORMAT_VERSION = 0

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
            with contextlib.suppress(OSError):
                os.unlink(temporary_name)
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
        if not isinstance(payload, dict):
            raise CheckpointError(f"Invalid checkpoint payload: {path}")

        # Historical writers saved either a raw state_dict or a wrapper using
        # ``model_state_dict``/``optimizer``.  Normalize those payloads at the
        # boundary so all canonical callers consume one shape.
        if "state_dict" not in payload:
            legacy_state = payload.get("model_state_dict", payload.get("model_state"))
            if (
                legacy_state is None
                and payload
                and all(isinstance(value, torch.Tensor) for value in payload.values())
            ):
                legacy_state = payload
            if legacy_state is None:
                raise CheckpointError(f"Invalid checkpoint payload: {path}")
            payload = {
                "metadata": {
                    "format_version": self.LEGACY_FORMAT_VERSION,
                    "algorithm": path.name.split("_", 1)[0],
                    "epoch": int(payload.get("epoch", 0))
                    if isinstance(payload.get("epoch", 0), int)
                    else 0,
                },
                "state_dict": legacy_state,
                "optimizer_state_dict": payload.get(
                    "optimizer_state_dict", payload.get("optimizer")
                ),
            }
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = getattr(metadata, "__dict__", {})
        version = metadata.get("format_version", self.LEGACY_FORMAT_VERSION)
        if version not in {self.FORMAT_VERSION, self.LEGACY_FORMAT_VERSION}:
            raise CheckpointError(
                f"Unsupported checkpoint version {version!r} in {path}"
            )
        payload["metadata"] = metadata
        return payload
