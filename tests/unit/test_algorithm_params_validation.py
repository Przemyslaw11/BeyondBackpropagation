"""Per-algorithm strict validation of ``algorithm_params`` keys (CFG-004)."""

from __future__ import annotations

from pathlib import Path

import pytest

from beyond_backprop.config import ConfigValidationError, load_mapping, validate_mapping

CONFIG_ROOT = Path(__file__).parents[2] / "configs"


def _mapping(algorithm: str = "FF") -> dict:
    return {
        "experiment_name": "strict",
        "general": {"seed": 3, "device": "cpu", "backend": "local"},
        "algorithm": {"name": algorithm},
        "model": {
            "name": {"FF": "FF_MLP", "MF": "MF_MLP", "CaFo": "CaFo_CNN"}[algorithm]
        },
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


def test_typo_key_raises_naming_bad_key():
    mapping = _mapping("FF")
    mapping["algorithm_params"] = {"predictor_lerning_rate": 0.001}
    with pytest.raises(ConfigValidationError, match="predictor_lerning_rate"):
        validate_mapping(mapping)


def test_foreign_algorithm_key_is_rejected():
    mapping = _mapping("MF")
    mapping["algorithm_params"] = {"ff_learning_rate": 1e-3}
    with pytest.raises(ConfigValidationError, match="ff_learning_rate"):
        validate_mapping(mapping)


def test_unresolved_algorithm_name_falls_back_to_union():
    mapping = _mapping("MF")
    del mapping["algorithm"]  # base templates carry shared defaults without a name
    mapping["algorithm_params"] = {"mf_early_stopping_patience": 5}
    validate_mapping(mapping)  # must not raise


@pytest.mark.parametrize(
    "path",
    sorted(p for p in CONFIG_ROOT.rglob("*.yaml")),
    ids=lambda p: str(p.relative_to(CONFIG_ROOT)),
)
def test_every_repository_yaml_passes_strict_validation(path: Path):
    resolved = load_mapping(path)  # merges base.yaml, normalizes, validates
    assert isinstance(resolved, dict)
