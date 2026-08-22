from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from beyond_backprop.algorithms import (
    ALGORITHM_REGISTRY,
    BPAdapter,
    CaFoAdapter,
    FFAdapter,
    MFAdapter,
    build_algorithm,
)
from beyond_backprop.algorithms.base import AlgorithmAdapter
from beyond_backprop.config.loader import experiment_config_from_mapping
from beyond_backprop.contracts import (
    EvaluationResult,
    ResourceSnapshot,
    RunStatus,
    TrainingContext,
    TrainingResult,
)
from beyond_backprop.training.runner import ExperimentRunner


def _config(algorithm: str = "BP"):
    normalized_algorithm = algorithm.lower()
    model_name = {
        "ff": "FF_MLP",
        "cafo": "CaFo_CNN",
    }.get(normalized_algorithm, "MF_MLP")
    return experiment_config_from_mapping(
        {
            "experiment_name": "tiny",
            "general": {"seed": 3, "device": "cpu", "backend": "local"},
            "algorithm": {"name": algorithm},
            "model": {"name": model_name, "params": {"hidden_dims": [3]}},
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
            "algorithm_params": {},
            "checkpointing": {},
            "monitoring": {},
            "tracking": {},
        }
    )


def _context(config, model):
    dataset = TensorDataset(
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]], [[[0.0, 1.0], [1.0, 0.0]]]]),
        torch.tensor([0, 1]),
    )
    loader = DataLoader(dataset, batch_size=2)
    return TrainingContext(config, model, loader, loader, torch.device("cpu"))


def test_algorithm_registry_lookup_and_adapter_construction():
    assert set(ALGORITHM_REGISTRY._factories) == {"bp", "ff", "cafo", "mf"}
    assert isinstance(build_algorithm("bp"), BPAdapter)
    assert isinstance(build_algorithm("ff"), FFAdapter)
    assert isinstance(build_algorithm("cafo"), CaFoAdapter)
    assert isinstance(build_algorithm("mf"), MFAdapter)


def test_bp_adapter_trains_global_cross_entropy_on_cpu():
    config = _config()
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    context = _context(config, model)
    result = BPAdapter().fit(context)
    assert result.status is RunStatus.SUCCEEDED
    assert "peak_memory_mib" in result.metrics


def test_ff_lifecycle_invocation_uses_local_and_downstream_stages(monkeypatch):
    class Legacy:
        @staticmethod
        def train_ff_model(**kwargs):
            return 0.0

        @staticmethod
        def evaluate_ff_model(*args):
            return {"eval_loss": 0.5, "eval_accuracy": 50.0}

    monkeypatch.setattr("beyond_backprop.algorithms.ff._legacy_module", lambda: Legacy)
    adapter = FFAdapter()
    context = _context(_config("FF"), nn.Linear(4, 2))
    result = adapter.fit(context)
    evaluation = adapter.evaluate(
        context.model,
        context.test_loader
        if hasattr(context, "test_loader")
        else context.train_loader,
        context,
    )
    assert result.status is RunStatus.SUCCEEDED
    assert adapter.lifecycle[:2] == ["local_goodness_updates", "downstream_classifier"]
    assert evaluation.accuracy_percent == 50.0


def test_cafo_component_lifecycle_records_frozen_or_trainable_blocks(monkeypatch):
    class Legacy:
        @staticmethod
        def train_cafo_model(**kwargs):
            return 0.0

        @staticmethod
        def evaluate_cafo_model(*args, **kwargs):
            return {"eval_loss": 0.5, "eval_accuracy": 50.0}

    monkeypatch.setattr(
        "beyond_backprop.algorithms.cafo._legacy_module", lambda: Legacy
    )
    adapter = CaFoAdapter()
    context = _context(_config("CaFo"), nn.Linear(4, 2))
    result = adapter.fit(context)
    assert result.status is RunStatus.SUCCEEDED
    assert adapter.lifecycle == ["blocks_frozen", "predictor_stages"]


def test_mf_adapter_preserves_layer_isolation_boundary(monkeypatch):
    class Legacy:
        @staticmethod
        def train_mf_model(**kwargs):
            assert kwargs["input_adapter"] is not None
            return 0.0

        @staticmethod
        def evaluate_mf_model(*args):
            return {"eval_loss": 0.5, "eval_accuracy": 50.0}

    monkeypatch.setattr("beyond_backprop.algorithms.mf._legacy_module", lambda: Legacy)
    model = nn.Module()
    model.num_hidden_layers = 2
    adapter = MFAdapter()
    context = _context(_config("MF"), model)
    result = adapter.fit(context)
    assert result.status is RunStatus.SUCCEEDED
    assert adapter.lifecycle == ["M0", "W1_M1", "W2_M2"]


class _TrackingAdapter(AlgorithmAdapter):
    name = "bp"

    def fit(self, context):
        self._snapshot(context.model)
        with torch.no_grad():
            next(context.model.parameters()).add_(10.0)
        return TrainingResult(status=RunStatus.SUCCEEDED)

    def restore_best_state(self, context, result=None):
        super().restore_best_state(context, result)

    def evaluate(self, model, loader, context):
        return EvaluationResult(
            float(next(model.parameters()).detach().abs().sum()), 50.0
        )


class _FailingAdapter(AlgorithmAdapter):
    name = "bp"

    def fit(self, context):
        raise RuntimeError("synthetic failure")

    def evaluate(self, model, loader, context):
        raise AssertionError("evaluation must not run")


class _Registry:
    def __init__(self, adapter):
        self.adapter = adapter

    def build(self, name):
        return self.adapter


@dataclass
class _Tracker:
    finished: RunStatus | None = None
    configs: int = 0

    def log_config(self, config):
        self.configs += 1

    def log_metrics(self, metrics):
        pass

    def log_artifact(self, path):
        pass

    def finish(self, status):
        self.finished = status


@dataclass
class _Monitor:
    started: bool = False
    stopped: bool = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        return ResourceSnapshot(source="synthetic")


def test_runner_restores_best_state_and_finalizes_disabled_services(tmp_path):
    config = _config()
    dataset = TensorDataset(torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    model = nn.Linear(4, 2)
    tracker = _Tracker()
    monitor = _Monitor()
    result = ExperimentRunner(
        data_builder=lambda cfg: (loader, loader, loader),
        model_builder=lambda cfg, device: model,
        algorithm_registry=_Registry(_TrackingAdapter()),
        tracker=tracker,
        resource_monitor=monitor,
        artifact_dir=tmp_path,
    ).run(config)
    assert result.status is RunStatus.SUCCEEDED
    assert result.evaluation is not None
    assert result.evaluation.loss < 10.0
    assert tracker.finished is RunStatus.SUCCEEDED
    assert tracker.configs == 1
    assert monitor.started and monitor.stopped
    assert (tmp_path / "config.resolved.yaml").exists()
    assert (tmp_path / "resolved_config.yaml").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "run_summary.json").exists()


def test_runner_failure_cleanup_does_not_require_tracking_or_gpu():
    config = _config()
    dataset = TensorDataset(torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    tracker = _Tracker()
    monitor = _Monitor()
    result = ExperimentRunner(
        data_builder=lambda cfg: (loader, loader, loader),
        model_builder=lambda cfg, device: nn.Linear(4, 2),
        algorithm_registry=_Registry(_FailingAdapter()),
        tracker=tracker,
        resource_monitor=monitor,
    ).run(config)
    assert result.status is RunStatus.FAILED
    assert "synthetic failure" in (result.error or "")
    assert tracker.finished is RunStatus.FAILED
    assert monitor.started and monitor.stopped


def test_runner_uses_noop_tracking_and_monitoring_by_default():
    config = _config()
    dataset = TensorDataset(torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    result = ExperimentRunner(
        data_builder=lambda cfg: (loader, loader, loader),
        model_builder=lambda cfg, device: nn.Linear(4, 2),
        algorithm_registry=_Registry(_TrackingAdapter()),
    ).run(config)
    assert result.status is RunStatus.SUCCEEDED
    assert result.resources.source == "disabled"
