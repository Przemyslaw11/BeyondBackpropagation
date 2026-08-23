from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from beyond_backprop.algorithms.base import evaluation_result
from beyond_backprop.algorithms.mf_math import local_cross_entropy
from beyond_backprop.architectures import build_model
from beyond_backprop.config import experiment_config_from_mapping
from beyond_backprop.contracts import RunStatus, TrainingResult
from beyond_backprop.monitoring.profiling import profile_model
from beyond_backprop.runtime import set_seed
from beyond_backprop.tuning.runner import _objective_from_runner


def _base_config() -> dict:
    return {
        "experiment_name": "scientific",
        "general": {"seed": 4, "device": "cpu", "backend": "local"},
        "algorithm": {"name": "BP"},
        "model": {"name": "MF_MLP", "params": {"hidden_dims": [4, 3]}},
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
        "training": {"epochs": 1},
    }


def test_identical_seeds_produce_identical_initialization() -> None:
    config = experiment_config_from_mapping(_base_config())
    set_seed(config.seed)
    first = build_model(config, torch.device("cpu"))
    first_state = copy.deepcopy(first.state_dict())
    set_seed(config.seed)
    second = build_model(config, torch.device("cpu"))
    assert all(
        torch.equal(first_state[name], value)
        for name, value in second.state_dict().items()
    )


def test_bp_baseline_has_no_mf_projection_parameters() -> None:
    config = experiment_config_from_mapping(_base_config())
    model = build_model(config, torch.device("cpu"))
    assert not any("projection" in name.lower() for name, _ in model.named_parameters())


def test_mf_preceding_layers_are_frozen_and_activations_detached() -> None:
    config = experiment_config_from_mapping(_base_config())
    model = build_model(config, torch.device("cpu"), for_bp_baseline=False)
    model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    active = model.layers[2]
    for parameter in active.parameters():
        parameter.requires_grad_(True)
    inputs = torch.randn(2, 4)
    with torch.no_grad():
        preceding = model.layers[1](model.layers[0](inputs))
    assert not preceding.requires_grad
    projection = model.projection_matrices[2]
    projection.requires_grad_(True)
    loss = local_cross_entropy(
        active(preceding.detach()), projection, torch.tensor([0, 1])
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in model.layers[0].parameters())
    assert all(parameter.grad is not None for parameter in active.parameters())


def test_percentage_point_metrics_do_not_get_converted_again() -> None:
    result = evaluation_result({"eval_loss": 0.5, "eval_accuracy": 62.5})
    assert result.accuracy_percent == 62.5
    assert result.metrics["accuracy_percent"].unit == "percentage_points"


def test_profiler_marks_estimates_as_unmeasured() -> None:
    config = experiment_config_from_mapping(_base_config())
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    profile = profile_model(model, config.to_mapping(), torch.device("cpu"))
    assert profile["forward_gflops"]["measured"] is False
    assert profile["estimated_bp_update_gflops"]["source"].startswith("estimated")


def test_tuning_objective_never_falls_back_to_test_evaluation(monkeypatch) -> None:
    class FakeRunner:
        def run(self, config):
            return type(
                "Result",
                (),
                {
                    "status": RunStatus.SUCCEEDED,
                    "error": None,
                    "training": TrainingResult(status=RunStatus.SUCCEEDED),
                },
            )()

    monkeypatch.setattr("beyond_backprop.tuning.runner.ExperimentRunner", FakeRunner)
    with pytest.raises(RuntimeError, match="validation best metric"):
        _objective_from_runner(_base_config())
