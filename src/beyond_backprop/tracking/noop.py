"""No-op tracking implementation that keeps local runs dependency-light."""

from collections.abc import Mapping
from typing import Any

from ..contracts import MetricValue, RunStatus


class NoOpTracker:
    def log_metrics(self, metrics: Mapping[str, MetricValue | float]) -> None:
        return None

    def log_config(self, config: Mapping[str, Any]) -> None:
        return None

    def log_artifact(self, path: str) -> None:
        return None

    def finish(self, status: RunStatus) -> None:
        return None
