"""Dependency-free local tracking for reproducible CPU and batch runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..contracts import MetricValue, RunStatus


def _metric_payload(value: MetricValue | float) -> Any:
    if isinstance(value, MetricValue):
        return value.to_dict()
    return {"value": float(value)}


class LocalFileTracker:
    """Persist config, metric events, artifacts, and final status as JSON."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.directory / "metrics.jsonl"
        self.artifacts_path = self.directory / "artifacts.json"
        self._artifacts: list[str] = []

    def log_metrics(self, metrics: Mapping[str, MetricValue | float]) -> None:
        event = {name: _metric_payload(value) for name, value in metrics.items()}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def log_config(self, config: Mapping[str, Any]) -> None:
        (self.directory / "config.json").write_text(
            json.dumps(dict(config), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def log_artifact(self, path: str) -> None:
        self._artifacts.append(path)
        self.artifacts_path.write_text(
            json.dumps(self._artifacts, indent=2), encoding="utf-8"
        )

    def finish(self, status: RunStatus) -> None:
        (self.directory / "status.json").write_text(
            json.dumps({"status": status.value}, indent=2), encoding="utf-8"
        )


__all__ = ["LocalFileTracker"]
