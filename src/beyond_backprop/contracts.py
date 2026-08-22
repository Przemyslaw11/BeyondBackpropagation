"""Small, dependency-light contracts shared by experiment components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .config.models import ExperimentConfig


class RunStatus(str, Enum):
    """Terminal and in-progress experiment states."""

    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class MetricProvenance:
    """How a metric was obtained."""

    source: str
    measured: bool
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"source": self.source, "measured": self.measured}
        if self.note:
            result["note"] = self.note
        return result


@dataclass(frozen=True)
class MetricValue:
    """A scalar metric with an explicit unit and provenance."""

    value: float
    unit: str
    provenance: MetricProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            **self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Standard evaluation output using percentage-point accuracy."""

    loss: float
    accuracy_percent: float
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    """Standard training output independent of a particular algorithm."""

    status: RunStatus
    best_epoch: int | None = None
    best_metric: MetricValue | None = None
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    checkpoint_path: str | None = None
    error: str | None = None
    evaluation: EvaluationResult | None = None


@dataclass(frozen=True)
class ResourceSnapshot:
    """Optional resource measurements for the canonical training region."""

    duration_sec: float | None = None
    energy_wh: float | None = None
    peak_memory_mib: float | None = None
    co2e_g: float | None = None
    measured: bool = False
    source: str = "none"

    def to_metrics(self) -> dict[str, MetricValue]:
        values: dict[str, MetricValue] = {}
        for name, value, unit in (
            ("duration_sec", self.duration_sec, "s"),
            ("energy_wh", self.energy_wh, "Wh"),
            ("peak_memory_mib", self.peak_memory_mib, "MiB"),
            ("co2e_g", self.co2e_g, "gCO2e"),
        ):
            if value is not None:
                values[name] = MetricValue(
                    float(value),
                    unit,
                    MetricProvenance(self.source, self.measured),
                )
        return values


@dataclass(frozen=True)
class RunMetadata:
    """Reproducibility metadata persisted with every canonical run."""

    run_id: str
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    command_line: str = ""
    git_commit: str | None = None
    git_dirty: bool | None = None
    python_version: str = ""
    torch_version: str | None = None
    device: str = "cpu"
    seed: int | None = None
    hostname: str = ""


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata needed to reason about a checkpoint without loading weights."""

    format_version: int
    algorithm: str
    epoch: int
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    config_hash: str | None = None


@dataclass(frozen=True)
class TrainingContext:
    """Inputs and shared services supplied to an algorithm trainer."""

    config: ExperimentConfig | Mapping[str, Any]
    model: Any
    train_loader: Any
    val_loader: Any | None = None
    device: Any = "cpu"
    evaluator: Evaluator | None = None
    tracker: ExperimentTracker | None = None
    resource_monitor: ResourceMonitor | None = None
    metadata: RunMetadata | None = None
    checkpoint_manager: Any | None = None


class TrainingAlgorithm(Protocol):
    """Minimal lifecycle required by the experiment runner."""

    def fit(self, context: TrainingContext) -> TrainingResult: ...


class Evaluator(Protocol):
    """Algorithm-aware evaluation boundary."""

    def evaluate(
        self, model: Any, loader: Any, context: TrainingContext
    ) -> EvaluationResult: ...


class ExperimentTracker(Protocol):
    """Optional tracking service used by the runner."""

    def log_metrics(self, metrics: Mapping[str, MetricValue | float]) -> None: ...

    def log_config(self, config: Mapping[str, Any]) -> None: ...

    def log_artifact(self, path: str) -> None: ...

    def finish(self, status: RunStatus) -> None: ...


class ResourceMonitor(Protocol):
    """Optional measurement service scoped to training."""

    def start(self) -> None: ...

    def stop(self) -> ResourceSnapshot: ...
