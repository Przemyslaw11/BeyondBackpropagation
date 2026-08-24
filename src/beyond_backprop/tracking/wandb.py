"""Optional W&B tracker with a lazy dependency boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import MetricValue, RunStatus


class WandbTracker:
    """Adapt the W&B run API without importing W&B during package import."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "W&B tracking was requested, but the tracking extra is not installed"
            ) from exc

        settings = dict(config)
        self._run = wandb.init(
            project=settings.get("project"),
            entity=settings.get("entity"),
            mode=settings.get("mode"),
            name=settings.get("run_name", settings.get("name")),
            config=settings,
        )

    def log_metrics(self, metrics: Mapping[str, MetricValue | float]) -> None:
        payload: dict[str, float] = {}
        for name, value in metrics.items():
            payload[name] = float(
                value.value if isinstance(value, MetricValue) else value
            )
        self._run.log(payload)

    def log_config(self, config: Mapping[str, Any]) -> None:
        self._run.config.update(dict(config), allow_val_change=True)

    def log_artifact(self, path: str) -> None:
        import wandb

        artifact = wandb.Artifact(name="run-artifacts", type="results")
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def finish(self, status: RunStatus) -> None:
        self._run.finish(exit_code=0 if status is RunStatus.SUCCEEDED else 1)


__all__ = ["WandbTracker"]
