"""Shared adapter helpers for canonical algorithm lifecycles."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..contracts import (
    EvaluationResult,
    MetricProvenance,
    MetricValue,
    RunStatus,
    TrainingContext,
    TrainingResult,
)


def context_mapping(context: TrainingContext) -> dict[str, Any]:
    """Return the mutable configuration shape expected by legacy routines."""
    config = context.config
    if isinstance(config, Mapping):
        return dict(config)
    return config.to_mapping()


def metric(
    value: float,
    *,
    unit: str = "unitless",
    source: str = "algorithm",
    measured: bool = True,
) -> MetricValue:
    return MetricValue(
        float(value), unit, MetricProvenance(source=source, measured=measured)
    )


def scalar_metrics(values: Mapping[str, Any]) -> dict[str, MetricValue]:
    """Convert legacy scalar summaries into explicit canonical metrics."""
    result: dict[str, MetricValue] = {}
    for name, value in values.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        result[name] = metric(numeric)
    return result


def flatten_if_needed(
    context: TrainingContext,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Return the MLP input adapter used by FF/MF and BP MLP baselines."""
    config = context_mapping(context)
    architecture = str(config.get("model", {}).get("name", "")).lower()
    if "mlp" not in architecture:
        return None

    def flatten(images: torch.Tensor) -> torch.Tensor:
        return images.view(images.shape[0], -1)

    return flatten


def result_from_peak_memory(
    algorithm: str,
    peak_memory: Any,
    diagnostics: Mapping[str, float] | None = None,
) -> TrainingResult:
    """Create a successful result from the legacy trainers' return value."""
    metrics: dict[str, MetricValue] = {}
    if peak_memory is not None:
        with contextlib.suppress(TypeError, ValueError):
            metrics["peak_memory_mib"] = metric(float(peak_memory), unit="MiB")
    for name, value in (diagnostics or {}).items():
        # RUN-002: unmeasured provenance keeps counters out of scientific
        # measurements while still persisting them in artifacts.
        metrics[name] = metric(float(value), source="trainer", measured=False)
    return TrainingResult(status=RunStatus.SUCCEEDED, metrics=metrics)


def evaluation_result(values: Mapping[str, Any]) -> EvaluationResult:
    """Normalize legacy evaluation dictionaries to canonical percentage units."""
    loss = float(values.get("eval_loss", values.get("loss", float("nan"))))
    accuracy = float(
        values.get(
            "eval_accuracy", values.get("accuracy_percent", values.get("accuracy", 0.0))
        )
    )
    return EvaluationResult(
        loss=loss,
        accuracy_percent=accuracy,
        metrics={
            "loss": metric(loss, unit="loss"),
            "accuracy_percent": metric(accuracy, unit="percentage_points"),
        },
    )


class AlgorithmAdapter:
    """Base class implementing the canonical fit/evaluate contract."""

    name = "unknown"

    def __init__(self) -> None:
        self.lifecycle: list[str] = []
        self._best_state: dict[str, Any] | None = None

    def fit(self, context: TrainingContext) -> TrainingResult:
        raise NotImplementedError

    def evaluate(
        self, model: Any, loader: Any, context: TrainingContext
    ) -> EvaluationResult:
        raise NotImplementedError

    def restore_best_state(
        self, context: TrainingContext, result: TrainingResult | None = None
    ) -> None:
        """Restore an in-memory best snapshot when an adapter captured one."""
        if self._best_state is not None and hasattr(context.model, "load_state_dict"):
            context.model.load_state_dict(self._best_state)
            return

        manager = context.checkpoint_manager
        if manager is None or not hasattr(context.model, "load_state_dict"):
            return
        experiment_name = context_mapping(context).get("experiment_name", "model")
        for filename in (
            f"{self.name}_{experiment_name}_best.pth",
            f"{self.name}_best.pth",
        ):
            try:
                payload = manager.load(filename, map_location=context.device)
            except Exception:
                # Legacy checkpoint writers may have already restored a raw
                # state_dict; canonical payloads are the manager's concern.
                continue
            state_dict = payload.get("state_dict")
            if isinstance(state_dict, Mapping):
                context.model.load_state_dict(state_dict)
                return

    def _snapshot(self, model: Any) -> None:
        if hasattr(model, "state_dict"):
            self._best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }


__all__ = [
    "AlgorithmAdapter",
    "context_mapping",
    "evaluation_result",
    "flatten_if_needed",
    "metric",
    "result_from_peak_memory",
    "scalar_metrics",
]
