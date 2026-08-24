"""Tests for the shared Optuna-to-YAML config updater in scripts/tuning_utils.

These are offline tests: Optuna is an optional dependency, so the study
lookup is monkeypatched and only the YAML read/modify/write contract is
exercised. The per-algorithm key maps are regression-guarded because they
define which published hyperparameters land back in the configs.
"""

import importlib
import sys
from pathlib import Path

import pytest
import yaml

TUNING_UTILS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "tuning_utils"
sys.path.insert(0, str(TUNING_UTILS_DIR))

common = importlib.import_module("_common")


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text(
        "general:\n  seed: 0\nalgorithm_params:\n  ff_learning_rate: 0.1\n",
        encoding="utf-8",
    )
    return path


def test_mapped_and_unknown_params_are_applied_and_skipped(config_file: Path) -> None:
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        common,
        "load_best_params",
        lambda db, name: {"ff_lr": 0.01, "ds_wd": 1e-4, "untuned": 5},
    )
    try:
        assert common.update_config_with_best_params(
            "unused.db",
            str(config_file),
            None,
            {"ff_lr": "ff_learning_rate", "ds_wd": "downstream_weight_decay"},
            label="FF",
            create_backup=False,
        )
    finally:
        monkey.undo()

    data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert data["algorithm_params"]["ff_learning_rate"] == 0.01
    assert data["algorithm_params"]["downstream_weight_decay"] == 1e-4
    assert "untuned" not in data["algorithm_params"]
    assert data["general"] == {"seed": 0}  # untouched sections survive


def test_missing_required_section_fails_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "bp.yaml"
    path.write_text("general:\n  seed: 0\n", encoding="utf-8")
    monkey = pytest.MonkeyPatch()
    monkey.setattr(common, "load_best_params", lambda db, name: {"lr": 0.5})
    try:
        assert not common.update_config_with_best_params(
            "unused.db",
            str(path),
            None,
            {"lr": "lr"},
            section="optimizer",
            create_missing_section=False,
            fail_on_empty_params=True,
            label="BP",
            create_backup=False,
        )
    finally:
        monkey.undo()
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"general": {"seed": 0}}


def test_missing_section_is_created_when_allowed(tmp_path: Path) -> None:
    path = tmp_path / "mf.yaml"
    path.write_text("general:\n  seed: 0\n", encoding="utf-8")
    monkey = pytest.MonkeyPatch()
    monkey.setattr(common, "load_best_params", lambda db, name: {"lr": 0.5})
    try:
        assert common.update_config_with_best_params(
            "unused.db",
            str(path),
            None,
            {"lr": "lr"},
            label="MF",
            create_backup=False,
        )
    finally:
        monkey.undo()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["algorithm_params"]["lr"] == 0.5


def test_backup_created_alongside_config(config_file: Path) -> None:
    monkey = pytest.MonkeyPatch()
    monkey.setattr(common, "load_best_params", lambda db, name: {"ff_lr": 0.02})
    try:
        assert common.update_config_with_best_params(
            "unused.db",
            str(config_file),
            None,
            {"ff_lr": "ff_learning_rate"},
            label="FF",
        )
    finally:
        monkey.undo()
    backups = list(config_file.parent.glob(f"{config_file.name}.bak_*"))
    assert len(backups) == 1
    old = yaml.safe_load(backups[0].read_text(encoding="utf-8"))
    assert old["algorithm_params"]["ff_learning_rate"] == 0.1


@pytest.mark.parametrize("fail_flag,expected", [(False, True), (True, False)])
def test_empty_best_params_exit_policy(
    config_file: Path, fail_flag: bool, expected: bool
) -> None:
    before = config_file.read_text(encoding="utf-8")
    monkey = pytest.MonkeyPatch()
    monkey.setattr(common, "load_best_params", lambda db, name: {})
    try:
        result = common.update_config_with_best_params(
            "unused.db",
            str(config_file),
            None,
            {"lr": "lr"},
            fail_on_empty_params=fail_flag,
            label="FF",
            create_backup=False,
        )
    finally:
        monkey.undo()
    assert result is expected
    assert config_file.read_text(encoding="utf-8") == before


def test_load_best_params_missing_db_returns_none(tmp_path: Path) -> None:
    assert common.load_best_params(str(tmp_path / "nope.db"), None) is None


@pytest.mark.parametrize(
    "module_name,expected_map",
    [
        (
            "update_ff_configs",
            {
                "ff_lr": "ff_learning_rate",
                "ff_wd": "ff_weight_decay",
                "ds_lr": "downstream_learning_rate",
                "ds_wd": "downstream_weight_decay",
            },
        ),
        (
            "update_mf_configs",
            {
                "lr": "lr",
                "epochs_per_layer": "epochs_per_layer",
                "wd": "weight_decay",
            },
        ),
        (
            "update_cafo_configs",
            {
                "pred_lr": "predictor_lr",
                "pred_wd": "predictor_weight_decay",
                "epochs_per_block": "num_epochs_per_block",
                "block_lr": "block_lr",
                "block_wd": "block_weight_decay",
                "block_epochs": "block_training_epochs",
            },
        ),
        (
            "update_bp_configs",
            {
                "lr": "lr",
                "wd": "weight_decay",
                "momentum": "momentum",
            },
        ),
    ],
)
def test_per_algorithm_key_maps_are_preserved(
    module_name: str, expected_map: dict
) -> None:
    module = importlib.import_module(module_name)
    assert expected_map == module.KEY_MAP
