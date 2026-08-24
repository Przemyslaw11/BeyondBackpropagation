from __future__ import annotations

import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from beyond_backprop.algorithms.base import AlgorithmAdapter
from beyond_backprop.config import experiment_config_from_mapping
from beyond_backprop.contracts import EvaluationResult, RunStatus, TrainingResult
from beyond_backprop.monitoring import (
    NoOpResourceMonitor,
    WallClockResourceMonitor,
    build_resource_monitor,
)
from beyond_backprop.tracking import NoOpTracker, build_tracker
from beyond_backprop.training import ExperimentRunner


def _config(*, enabled: bool = True):
    return experiment_config_from_mapping(
        {
            "experiment_name": "runner_services",
            "general": {"seed": 1, "device": "cpu", "backend": "local"},
            "algorithm": {"name": "BP"},
            "model": {"name": "MF_MLP", "params": {"hidden_dims": [3]}},
            "data": {
                "name": "MNIST",
                "root": "/tmp/offline",
                "download": False,
                "val_split": 0.2,
                "num_classes": 2,
                "input_channels": 1,
                "image_size": 2,
            },
            "data_loader": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
            "optimizer": {"type": "AdamW", "lr": 0.01, "weight_decay": 0.0},
            "training": {"epochs": 1, "early_stopping_enabled": False},
            "monitoring": {"enabled": enabled, "energy_enabled": False},
            "tracking": {"enabled": enabled, "mode": "local"},
            "profiling": {"enabled": True},
        }
    )


class _Adapter(AlgorithmAdapter):
    name = "bp"

    def fit(self, context):
        self._snapshot(context.model)
        return TrainingResult(status=RunStatus.SUCCEEDED)

    def evaluate(self, model, loader, context):
        return EvaluationResult(loss=0.25, accuracy_percent=75.0)


class _Registry:
    def build(self, name):
        return _Adapter()


def test_factories_select_noop_when_disabled(tmp_path) -> None:
    config = _config(enabled=False)
    assert isinstance(build_resource_monitor(config), NoOpResourceMonitor)
    assert isinstance(build_tracker(config, directory=tmp_path), NoOpTracker)


def test_enabled_local_run_persists_complete_artifacts(tmp_path) -> None:
    config = _config(enabled=True)
    dataset = TensorDataset(torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    result = ExperimentRunner(
        data_builder=lambda cfg: (loader, loader, loader),
        model_builder=lambda cfg, device: nn.Linear(4, 2),
        algorithm_registry=_Registry(),
        artifact_dir=tmp_path,
    ).run(config)
    assert result.status is RunStatus.SUCCEEDED
    assert result.resources.duration_sec is not None
    assert isinstance(result.resources.source, str)
    for relative in (
        "config.resolved.yaml",
        "metadata.json",
        "metrics.json",
        "history.csv",
        "summary.json",
        "logs/run.log",
        "profiling/profile.json",
        "status.json",
        "metrics.jsonl",
    ):
        assert (tmp_path / relative).exists(), relative
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["config_hash"] == config.config_hash
    profile = json.loads((tmp_path / "profiling/profile.json").read_text())
    assert profile["forward_gflops"]["measured"] is False


def test_evaluation_occurs_after_monitor_stop(tmp_path) -> None:
    events: list[str] = []

    class Monitor(WallClockResourceMonitor):
        def stop(self):
            events.append("stop")
            return super().stop()

    class Adapter(_Adapter):
        def evaluate(self, model, loader, context):
            events.append("evaluate")
            return super().evaluate(model, loader, context)

    class Registry:
        def build(self, name):
            return Adapter()

    config = _config(enabled=False)
    dataset = TensorDataset(torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    result = ExperimentRunner(
        data_builder=lambda cfg: (loader, loader, loader),
        model_builder=lambda cfg, device: nn.Linear(4, 2),
        algorithm_registry=Registry(),
        monitor_factory=lambda cfg: Monitor(),
        artifact_dir=tmp_path,
    ).run(config)
    assert result.status is RunStatus.SUCCEEDED
    assert events == ["stop", "evaluate"]
