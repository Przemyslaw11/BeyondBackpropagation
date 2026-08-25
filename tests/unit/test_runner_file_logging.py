"""Runner artifact-file logging (OBS-002 contract)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from beyond_backprop.algorithms.base import AlgorithmAdapter
from beyond_backprop.config.loader import experiment_config_from_mapping
from beyond_backprop.contracts import ResourceSnapshot, RunStatus
from beyond_backprop.training.runner import ExperimentRunner
from beyond_backprop.utils.training_support import attach_artifact_log_handler


def _config() -> dict:
    return experiment_config_from_mapping(
        {
            "experiment_name": "file-logging",
            "general": {"seed": 3, "device": "cpu", "backend": "local"},
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
            "algorithm_params": {},
            "checkpointing": {},
            "monitoring": {},
            "tracking": {},
        }
    )


@dataclass
class _Tracker:
    def log_config(self, config):
        pass

    def log_metrics(self, metrics):
        pass

    def log_artifact(self, path):
        pass

    def finish(self, status):
        pass


class _Monitor:
    def start(self):
        pass

    def stop(self):
        return ResourceSnapshot(source="synthetic")


class _Registry:
    def __init__(self, adapter: AlgorithmAdapter) -> None:
        self.adapter = adapter

    def build(self, name: str) -> AlgorithmAdapter:
        return self.adapter


class _FailingFitAdapter(AlgorithmAdapter):
    name = "bp"

    def fit(self, context):
        raise RuntimeError("synthetic file-logging failure")

    def evaluate(self, model, loader, context):  # pragma: no cover - not reached
        raise AssertionError("evaluation must not run")


def _file_handlers() -> list[logging.FileHandler]:
    return [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, logging.FileHandler)
    ]


def test_failing_run_writes_traceback_to_artifact_log(tmp_path: Path):
    dataset = TensorDataset(torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    baseline_handlers = list(logging.getLogger().handlers)

    result = ExperimentRunner(
        data_builder=lambda cfg: (loader, loader, loader),
        model_builder=lambda cfg, device: nn.Linear(4, 2),
        algorithm_registry=_Registry(_FailingFitAdapter()),
        tracker=_Tracker(),
        resource_monitor=_Monitor(),
        artifact_dir=tmp_path,
    ).run(_config())

    assert result.status is RunStatus.FAILED
    log_file = tmp_path / "logs" / "run.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "synthetic file-logging failure" in content
    assert "RuntimeError" in content
    # The two-line run summary is still appended with its stable schema
    # (same file as the artifact log, per the legacy layout).
    assert "status=failed" in content
    # No FileHandler targeting the artifact log leaks after run().
    assert all(
        getattr(handler, "baseFilename", "") != str(log_file)
        for handler in _file_handlers()
    )
    assert len(logging.getLogger().handlers) == len(baseline_handlers)


def test_artifact_log_accumulates_across_attachments(tmp_path: Path):
    log_path = tmp_path / "logs" / "run.log"

    first = attach_artifact_log_handler(log_path)
    logging.getLogger().warning("first-pass message")
    logging.getLogger().removeHandler(first)
    first.close()

    second = attach_artifact_log_handler(log_path)
    logging.getLogger().warning("second-pass message")
    logging.getLogger().removeHandler(second)
    second.close()

    content = log_path.read_text(encoding="utf-8")
    assert "first-pass message" in content
    assert "second-pass message" in content


def test_run_skips_handler_when_target_already_attached(tmp_path: Path, monkeypatch):
    dataset = TensorDataset(torch.zeros(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)

    calls: list[Path] = []
    real_attach = attach_artifact_log_handler

    def spy(path):
        calls.append(Path(path))
        return real_attach(path)

    monkeypatch.setattr(
        "beyond_backprop.training.runner.attach_artifact_log_handler", spy
    )

    pre = attach_artifact_log_handler(tmp_path / "logs" / "run.log")
    try:
        ExperimentRunner(
            data_builder=lambda cfg: (loader, loader, loader),
            model_builder=lambda cfg, device: nn.Linear(4, 2),
            algorithm_registry=_Registry(_FailingFitAdapter()),
            tracker=_Tracker(),
            resource_monitor=_Monitor(),
            artifact_dir=tmp_path,
        ).run(_config())
    finally:
        logging.getLogger().removeHandler(pre)
        pre.close()

    assert calls == []  # dedupe: same-file handler was already attached
