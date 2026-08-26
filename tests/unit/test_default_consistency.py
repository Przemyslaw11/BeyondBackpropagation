"""Single-sourced defaults (WP8): minimal mapping resolves to the constants."""

from __future__ import annotations

from beyond_backprop.config.loader import experiment_config_from_mapping
from beyond_backprop.config.models import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_SIZE_FF,
)
from beyond_backprop.training.early_stopping import DEFAULT_MIN_DELTA, DEFAULT_PATIENCE


def _mapping(algorithm: str) -> dict:
    return {
        "experiment_name": "defaults",
        "general": {"seed": 3, "device": "cpu", "backend": "local"},
        "algorithm": {"name": algorithm},
        "model": {"name": {"FF": "FF_MLP", "MF": "MF_MLP"}[algorithm], "params": {}},
        "data": {
            "name": "MNIST",
            "root": "/tmp/offline",
            "download": False,
            "val_split": 0.2,
            "num_classes": 2,
            "input_channels": 1,
            "image_size": 2,
        },
        "optimizer": {"type": "AdamW", "lr": 0.01, "weight_decay": 0.0},
        "training": {"epochs": 1, "early_stopping_enabled": False},
        "algorithm_params": {},
    }


def test_minimal_ff_mapping_uses_ff_batch_size_constant():
    config = experiment_config_from_mapping(_mapping("FF"))
    assert config.batch_size == DEFAULT_BATCH_SIZE_FF == 100


def test_minimal_mf_mapping_uses_general_batch_size_constant():
    config = experiment_config_from_mapping(_mapping("MF"))
    assert config.batch_size == DEFAULT_BATCH_SIZE == 128


def test_constants_are_distinct_and_match_documented_values():
    assert DEFAULT_BATCH_SIZE_FF == 100
    assert DEFAULT_BATCH_SIZE == 128
    assert DEFAULT_PATIENCE == 10
    assert DEFAULT_MIN_DELTA == 0.0
