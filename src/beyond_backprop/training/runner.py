"""Shared experiment runner for canonical algorithm adapters."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..algorithms import ALGORITHM_REGISTRY, AlgorithmAdapter
from ..architectures import build_model
from ..checkpointing import CheckpointManager
from ..config.loader import experiment_config_from_mapping
from ..config.models import ExperimentConfig
from ..contracts import (
    EvaluationResult,
    ExperimentTracker,
    MetricProvenance,
    MetricValue,
    ResourceMonitor,
    ResourceSnapshot,
    RunMetadata,
    RunStatus,
    TrainingContext,
    TrainingResult,
)
from ..data import build_dataloaders
from ..monitoring import NoOpResourceMonitor
from ..runtime import collect_run_metadata, resolve_device, set_seed
from ..tracking import NoOpTracker


@dataclass(frozen=True)
class ExperimentResult:
    """Terminal result of one runner invocation."""

    status: RunStatus
    training: TrainingResult
    evaluation: EvaluationResult | None = None
    resources: ResourceSnapshot = field(default_factory=ResourceSnapshot)
    artifacts: tuple[str, ...] = ()
    error: str | None = None

    @property
    def training_result(self) -> TrainingResult:
        """Compatibility spelling for callers that name the phase explicitly."""
        return self.training


TrackerFactory = Callable[[ExperimentConfig], ExperimentTracker]
MonitorFactory = Callable[[ExperimentConfig], ResourceMonitor]


class ExperimentRunner:
    """Orchestrate shared infrastructure without flattening algorithm lifecycles."""

    def __init__(
        self,
        *,
        data_builder: Callable[
            [ExperimentConfig | dict[str, Any]], tuple[Any, Any | None, Any]
        ] = build_dataloaders,
        model_builder: Callable[
            [ExperimentConfig | dict[str, Any], torch.device], Any
        ] = build_model,
        algorithm_registry: Any = ALGORITHM_REGISTRY,
        tracker: ExperimentTracker | None = None,
        resource_monitor: ResourceMonitor | None = None,
        tracker_factory: TrackerFactory | None = None,
        monitor_factory: MonitorFactory | None = None,
        artifact_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.data_builder = data_builder
        self.model_builder = model_builder
        self.algorithm_registry = algorithm_registry
        self.tracker = tracker
        self.resource_monitor = resource_monitor
        self.tracker_factory = tracker_factory
        self.monitor_factory = monitor_factory
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None

    def run(self, config: ExperimentConfig | dict[str, Any]) -> ExperimentResult:
        """Execute the canonical lifecycle and finalize services on every path."""
        typed_config = self._coerce_config(config)
        mapping = typed_config.to_mapping()
        tracker: ExperimentTracker | None = None
        monitor: ResourceMonitor | None = None

        status = RunStatus.CREATED
        resource_snapshot = ResourceSnapshot(source="not_started")
        monitor_started = False
        training_result = TrainingResult(status=RunStatus.CREATED)
        evaluation: EvaluationResult | None = None
        artifacts: tuple[str, ...] = ()
        error: str | None = None
        metadata: RunMetadata | None = None

        try:
            status = RunStatus.RUNNING
            self._prepare_backend_environment(mapping)
            device = resolve_device(mapping)
            set_seed(typed_config.seed)
            train_loader, val_loader, test_loader = self.data_builder(typed_config)
            model = self.model_builder(typed_config, device)
            algorithm: AlgorithmAdapter = self.algorithm_registry.build(
                typed_config.algorithm
            )
            tracker = self.tracker or (
                self.tracker_factory(typed_config)
                if self.tracker_factory is not None
                else NoOpTracker()
            )
            monitor = self.resource_monitor or (
                self.monitor_factory(typed_config)
                if self.monitor_factory is not None
                else NoOpResourceMonitor()
            )
            checkpoint_manager = self._checkpoint_manager(mapping)
            metadata = collect_run_metadata(
                typed_config.experiment_name,
                device=str(device),
                seed=typed_config.seed,
            )
            context = TrainingContext(
                config=typed_config,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                tracker=tracker,
                resource_monitor=monitor,
                metadata=metadata,
                checkpoint_manager=checkpoint_manager,
            )

            tracker.log_config(mapping)
            monitor.start()
            monitor_started = True
            training_result = algorithm.fit(context)
            if training_result.status is RunStatus.FAILED:
                raise RuntimeError(training_result.error or "Algorithm training failed")
            algorithm.restore_best_state(context, training_result)
            evaluation = algorithm.evaluate(model, test_loader, context)
            training_result = TrainingResult(
                status=training_result.status,
                best_epoch=training_result.best_epoch,
                best_metric=training_result.best_metric,
                metrics=training_result.metrics,
                checkpoint_path=training_result.checkpoint_path,
                error=training_result.error,
                evaluation=evaluation,
            )
            tracker.log_metrics(
                {
                    "evaluation_loss": evaluation.loss,
                    "evaluation_accuracy_percent": evaluation.accuracy_percent,
                }
            )
            status = RunStatus.SUCCEEDED
        except Exception as exc:
            status = RunStatus.FAILED
            error = f"{type(exc).__name__}: {exc}"
            training_result = TrainingResult(
                status=RunStatus.FAILED,
                metrics=training_result.metrics,
                error=error,
            )
        finally:
            if monitor_started and monitor is not None:
                try:
                    resource_snapshot = monitor.stop()
                    if tracker is not None:
                        tracker.log_metrics(resource_snapshot.to_metrics())
                except Exception as exc:
                    if status is RunStatus.SUCCEEDED:
                        status = RunStatus.FAILED
                        error = f"{type(exc).__name__}: {exc}"
                        training_result = TrainingResult(
                            status=RunStatus.FAILED,
                            metrics=training_result.metrics,
                            evaluation=evaluation,
                            error=error,
                        )
            if metadata is not None:
                try:
                    artifacts = self._persist_artifacts(
                        typed_config,
                        training_result,
                        evaluation,
                        resource_snapshot,
                        metadata,
                        status,
                    )
                    if tracker is not None:
                        for artifact in artifacts:
                            tracker.log_artifact(artifact)
                except Exception as exc:
                    if status is RunStatus.SUCCEEDED:
                        status = RunStatus.FAILED
                        error = f"{type(exc).__name__}: {exc}"
                        training_result = TrainingResult(
                            status=RunStatus.FAILED,
                            metrics=training_result.metrics,
                            evaluation=evaluation,
                            error=error,
                        )
            if tracker is not None:
                try:
                    tracker.finish(status)
                except Exception as exc:
                    if status is RunStatus.SUCCEEDED:
                        status = RunStatus.FAILED
                        error = f"{type(exc).__name__}: {exc}"
                        training_result = TrainingResult(
                            status=RunStatus.FAILED,
                            metrics=training_result.metrics,
                            evaluation=evaluation,
                            error=error,
                        )

        return ExperimentResult(
            status=status,
            training=training_result,
            evaluation=evaluation,
            resources=resource_snapshot,
            artifacts=artifacts,
            error=error,
        )

    @staticmethod
    def _coerce_config(config: ExperimentConfig | dict[str, Any]) -> ExperimentConfig:
        if isinstance(config, ExperimentConfig):
            return config
        # Mapping callers should use the same deterministic loader defaults and
        # immutable contract as YAML callers without touching the filesystem.
        return experiment_config_from_mapping(config)

    @staticmethod
    def _prepare_backend_environment(mapping: dict[str, Any]) -> None:
        from ..runtime.backend_policy import get_execution_backend

        backend = get_execution_backend(mapping)
        for key, value in backend.prepare_environment(mapping).items():
            os.environ.setdefault(key, value)

    @staticmethod
    def _checkpoint_manager(mapping: dict[str, Any]) -> CheckpointManager | None:
        checkpoint_dir = mapping.get("checkpointing", {}).get("checkpoint_dir")
        return CheckpointManager(checkpoint_dir) if checkpoint_dir else None

    def _persist_artifacts(
        self,
        config: ExperimentConfig,
        training: TrainingResult,
        evaluation: EvaluationResult | None,
        resources: ResourceSnapshot,
        metadata: RunMetadata,
        status: RunStatus,
    ) -> tuple[str, ...]:
        target = self.artifact_dir
        if target is None:
            checkpoint_dir = (
                config.to_mapping().get("checkpointing", {}).get("checkpoint_dir")
            )
            target = Path(checkpoint_dir) if checkpoint_dir else None
        if target is None:
            return ()
        target.mkdir(parents=True, exist_ok=True)
        for directory_name in ("checkpoints", "logs", "profiling"):
            (target / directory_name).mkdir(exist_ok=True)

        from ..config.loader import save_resolved_config

        canonical_config_path = target / "config.resolved.yaml"
        legacy_config_path = target / "resolved_config.yaml"
        save_resolved_config(config, canonical_config_path)
        save_resolved_config(config, legacy_config_path)

        metadata_payload = metadata.to_dict()
        metadata_payload["config_hash"] = config.config_hash
        metadata_path = target / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8"
        )

        metrics = self._artifact_metrics(training, evaluation, resources)
        metrics_path = target / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {name: value.to_dict() for name, value in metrics.items()},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        history_path = target / "history.csv"
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("phase", "metric", "value", "unit", "source", "measured"),
            )
            writer.writeheader()
            for name, value in metrics.items():
                writer.writerow(
                    {
                        "phase": "evaluation" if name.startswith("eval_") else "run",
                        "metric": name,
                        **value.to_dict(),
                    }
                )

        summary = {
            "status": status.value,
            "config_hash": config.config_hash,
            "training": training.to_dict(),
            "evaluation": evaluation.to_dict() if evaluation is not None else None,
            "resources": resources.to_dict(),
            "metadata": metadata_payload,
        }
        canonical_summary_path = target / "summary.json"
        legacy_summary_path = target / "run_summary.json"
        summary_text = json.dumps(summary, indent=2, sort_keys=True, default=str)
        canonical_summary_path.write_text(summary_text, encoding="utf-8")
        legacy_summary_path.write_text(summary_text, encoding="utf-8")
        return tuple(
            str(path)
            for path in (
                canonical_config_path,
                legacy_config_path,
                metadata_path,
                metrics_path,
                history_path,
                canonical_summary_path,
                legacy_summary_path,
            )
        )

    @staticmethod
    def _artifact_metrics(
        training: TrainingResult,
        evaluation: EvaluationResult | None,
        resources: ResourceSnapshot,
    ) -> dict[str, MetricValue]:
        metrics: dict[str, MetricValue] = {}
        for name, value in training.metrics.items():
            if isinstance(value, MetricValue):
                metrics[f"train_{name}"] = value
            else:
                metrics[f"train_{name}"] = MetricValue(
                    float(value), "", MetricProvenance("training", True)
                )
        if evaluation is not None:
            metrics["eval_loss"] = MetricValue(
                evaluation.loss, "", MetricProvenance("evaluator", True)
            )
            metrics["eval_accuracy_percent"] = MetricValue(
                evaluation.accuracy_percent,
                "percentage_points",
                MetricProvenance("evaluator", True),
            )
            metrics.update(
                {f"eval_{name}": value for name, value in evaluation.metrics.items()}
            )
        metrics.update(resources.to_metrics())
        return metrics


def run_experiment(
    config: ExperimentConfig | dict[str, Any], **runner_kwargs: Any
) -> ExperimentResult:
    """Functional entry point for the shared runner."""
    return ExperimentRunner(**runner_kwargs).run(config)


__all__ = ["ExperimentResult", "ExperimentRunner", "run_experiment"]
