"""Direct tests for experiment.factories (currently at 0% coverage)."""

from __future__ import annotations

import torch

from beyond_backprop.algorithms import BPAdapter
from beyond_backprop.architectures.mf_mlp import MF_MLP
from beyond_backprop.experiment.factories import (
    build_algorithm_factory,
    build_model_for_config,
    build_runner,
    load_config,
)
from beyond_backprop.training import ExperimentRunner


def _mapping() -> dict:
    return {
        "experiment_name": "factories-test",
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


def test_build_algorithm_factory_constructs_registered_adapter():
    adapter = build_algorithm_factory("bp")()
    assert isinstance(adapter, BPAdapter)


def test_build_runner_returns_runner_instance():
    assert isinstance(build_runner(), ExperimentRunner)


def test_build_model_for_config_builds_native_and_baseline_from_mapping():
    # algorithm BP selects the fair BP baseline for the configured architecture.
    baseline = build_model_for_config(_mapping(), torch.device("cpu"))
    assert isinstance(baseline, torch.nn.Sequential)

    # A non-BP algorithm selects the native model named in model.name.
    native = build_model_for_config(
        {**_mapping(), "algorithm": {"name": "FF"}}, torch.device("cpu")
    )
    assert isinstance(native, MF_MLP)


def test_load_config_reads_yaml_with_base_defaults(tmp_path):
    experiment_yaml = tmp_path / "experiment.yaml"
    experiment_yaml.write_text(
        "\n".join(
            [
                "experiment_name: factories-file-test",
                "algorithm:",
                "  name: BP",
                "training:",
                "  epochs: 1",
                "  early_stopping_enabled: false",
            ]
        )
    )
    config = load_config(experiment_yaml, base_config="configs/base.yaml")
    mapping = config.to_mapping()
    assert mapping["experiment_name"] == "factories-file-test"
    assert mapping["algorithm"]["name"] == "BP"
    # Values inherited from configs/base.yaml.
    assert mapping["training"]["log_interval"] == 100
