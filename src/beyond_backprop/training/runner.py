"""Shared experiment runner for canonical algorithm adapters."""

from __future__ import annotations

import csv
import json
import logging
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
from ..monitoring import profile_model
from ..runtime import collect_run_metadata, resolve_device, set_seed


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

logger = logging.getLogger(__name__)


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
        # CFG-004: every artifact shows the hyperparameters that actually ran.
        logger.info(
            "Resolved algorithm_params for %s: %s",
            typed_config.algorithm,
            mapping.get("algorithm_params", {}),
        )
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
        profiling: dict[str, Any] = {}

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
            artifact_target = self.artifact_dir or self._artifact_target(typed_config)
            if self.tracker is not None:
                tracker = self.tracker
            elif self.tracker_factory is not None:
                tracker = self.tracker_factory(typed_config)
            else:
                from ..tracking import build_tracker

                tracker = build_tracker(typed_config, directory=artifact_target)
            if self.resource_monitor is not None:
                monitor = self.resource_monitor
            elif self.monitor_factory is not None:
                monitor = self.monitor_factory(typed_config)
            else:
                from ..monitoring import build_resource_monitor

                monitor = build_resource_monitor(typed_config)
            checkpoint_manager = self._checkpoint_manager(mapping, artifact_target)
            metadata = collect_run_metadata(
                typed_config.experiment_name,
                device=str(device),
                seed=typed_config.seed,
                config_hash=typed_config.config_hash,
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

            if typed_config.to_mapping().get("profiling", {}).get("enabled", False):
                profiling = profile_model(model, mapping, device)

            tracker.log_config(mapping)
            monitor.start()
            monitor_started = True
            training_result = algorithm.fit(context)
            # Evaluation is deliberately outside the canonical training
            # measurement region.
            monitor_started = False
            resource_snapshot = monitor.stop()
            tracker.log_metrics(resource_snapshot.to_metrics())
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
                    monitor_started = False
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
                        profiling,
                    )
                    if tracker is not None:
                        for artifact in artifacts:
                            try:
                                tracker.log_artifact(artifact)
                            except Exception:
                                self._rewrite_summary_status(
                                    typed_config,
                                    RunStatus.FAILED,
                                    "tracker artifact logging failed",
                                )
                                raise
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
                        self._rewrite_summary_status(typed_config, status, error)
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
                        self._rewrite_summary_status(typed_config, status, error)

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
    def _checkpoint_manager(
        mapping: dict[str, Any], default_directory: Path | None = None
    ) -> CheckpointManager | None:
        checkpoint_dir = mapping.get("checkpointing", {}).get("checkpoint_dir")
        directory = checkpoint_dir or default_directory
        return CheckpointManager(directory) if directory is not None else None

    @staticmethod
    def _artifact_target(config: ExperimentConfig) -> Path:
        from ..runtime import get_execution_backend

        mapping = config.to_mapping()
        backend = get_execution_backend(mapping)
        return Path(backend.resolve_results_dir(mapping)) / config.experiment_name

    def _persist_artifacts(
        self,
        config: ExperimentConfig,
        training: TrainingResult,
        evaluation: EvaluationResult | None,
        resources: ResourceSnapshot,
        metadata: RunMetadata,
        status: RunStatus,
        profiling: dict[str, Any] | None = None,
    ) -> tuple[str, ...]:
        target = self.artifact_dir or self._artifact_target(config)
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

        profile_path = target / "profiling" / "profile.json"
        profile_path.write_text(
            json.dumps(profiling or {}, indent=2, sort_keys=True), encoding="utf-8"
        )
        log_path = target / "logs" / "run.log"
        log_path.write_text(
            f"status={status.value}\nexperiment_name={config.experiment_name}\n",
            encoding="utf-8",
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
                profile_path,
                log_path,
            )
        )

    def _rewrite_summary_status(
        self, config: ExperimentConfig, status: RunStatus, error: str | None
    ) -> None:
        """Keep already-written summaries consistent with late service failures."""

        target = self.artifact_dir or self._artifact_target(config)
        for filename in ("summary.json", "run_summary.json"):
            path = target / filename
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["status"] = status.value
                if error:
                    payload["error"] = error
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True, default=str),
                    encoding="utf-8",
                )
            except (OSError, json.JSONDecodeError):
                continue

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
