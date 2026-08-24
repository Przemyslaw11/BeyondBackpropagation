from pathlib import Path

import pytest

from beyond_backprop.config import (
    ConfigValidationError,
    load_experiment_config,
    load_mapping,
)

CONFIG_ROOT = Path(__file__).parents[2] / "configs"


def test_all_repository_yaml_configs_load_after_normalization() -> None:
    paths = sorted(CONFIG_ROOT.rglob("*.yaml"))
    assert len(paths) == 44

    for path in paths:
        resolved = load_mapping(path)
        assert isinstance(resolved, dict)


def test_legacy_cafo_fields_are_normalized_and_lists_replace() -> None:
    config = load_mapping("configs/cafo/fashion_mnist_cnn_3block.yaml")

    data = config["data"]
    params = config["algorithm_params"]
    assert "input_channels'" not in data
    assert data["input_channels"] == 1
    assert params["num_epochs_per_block"] == 145
    assert "epochs_per_block" not in params
    assert params["predictor_early_stopping_patience"] == 8

    tuning = load_mapping("configs/tuning/cafo_cifar10_cnn_3block_tune.yaml")
    assert tuning["model"]["params"]["block_channels"] == [32, 128, 512]


def test_config_hash_and_typed_config_are_reproducible_and_mutable_at_boundary() -> (
    None
):
    first = load_experiment_config("configs/mf/mnist_mlp_2x1000.yaml")
    second = load_experiment_config("configs/mf/mnist_mlp_2x1000.yaml")

    assert first.config_hash == second.config_hash
    assert first.algorithm.value == "mf"
    assert first.to_mapping()["model"]["params"]["hidden_dims"] == [1000, 1000]
    with pytest.raises(TypeError):
        first.model_params["hidden_dims"] = [3]  # type: ignore[index]


def test_explicit_cli_overrides_have_highest_precedence() -> None:
    config = load_experiment_config(
        "configs/mf/mnist_mlp_2x1000.yaml",
        overrides=[
            "general.device=cpu",
            "data.download=false",
            "data_loader.batch_size=17",
        ],
    )
    assert config.device == "cpu"
    assert not config.download
    assert config.batch_size == 17


def test_invalid_cli_override_is_actionable() -> None:
    with pytest.raises(ConfigValidationError, match="section.key=value"):
        load_mapping("configs/mf/mnist_mlp_2x1000.yaml", overrides=["bad-override"])


def test_unknown_keys_fail_before_runtime_resolution() -> None:
    with pytest.raises(ConfigValidationError, match="Unknown keys in 'model.params'"):
        from beyond_backprop.config.loader import validate_mapping

        validate_mapping(
            {
                "algorithm": {"name": "BP"},
                "model": {"name": "MF_MLP", "params": {"not_a_field": 1}},
                "data": {"name": "MNIST"},
            }
        )


def test_incompatible_early_stopping_metric_fails_actionably() -> None:
    with pytest.raises(ConfigValidationError, match="Accuracy early-stopping"):
        from beyond_backprop.config.loader import validate_mapping

        validate_mapping(
            {
                "algorithm": {"name": "BP"},
                "model": {"name": "MF_MLP"},
                "data": {"name": "MNIST"},
                "training": {
                    "early_stopping_metric": "val_accuracy",
                    "early_stopping_mode": "min",
                },
            }
        )
