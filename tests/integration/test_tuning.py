from __future__ import annotations

import json

from beyond_backprop.tuning import generate_trial_config, run_study, search_space


def _config() -> dict:
    return {
        "experiment_name": "synthetic_tune",
        "general": {"seed": 10, "device": "cpu", "backend": "local"},
        "algorithm": {"name": "FF"},
        "model": {"name": "FF_MLP", "params": {"hidden_dims": [3]}},
        "data": {"name": "MNIST", "root": "/tmp/offline", "download": False},
        "tuning": {
            "enabled": True,
            "n_trials": 3,
            "direction": "maximize",
            "metric": "val_accuracy",
            "sampler": "RANDOM",
            "pruner": "None",
            "ff_lr_range": [0.001, 0.01],
            "ff_wd_range": [0.0001, 0.001],
            "ds_lr_range": [0.002, 0.02],
            "ds_wd_range": [0.001, 0.01],
        },
    }


def test_search_space_preserves_legacy_parameter_names_and_trial_seeds() -> None:
    config = _config()
    assert [parameter.name for parameter in search_space(config)] == [
        "ff_lr",
        "ff_wd",
        "ds_lr",
        "ds_wd",
    ]
    first = generate_trial_config(
        config,
        {parameter.name: parameter.low for parameter in search_space(config)},
        2,
    )
    second = generate_trial_config(
        config,
        {parameter.name: parameter.low for parameter in search_space(config)},
        2,
    )
    assert first["general"]["seed"] == 12
    assert first == second
    assert first["monitoring"]["enabled"] is False
    assert first["tracking"]["enabled"] is False


def test_synthetic_study_is_bounded_deterministic_and_serialized(tmp_path) -> None:
    config = _config()

    def objective(trial_config):
        return float(trial_config["algorithm_params"]["ff_learning_rate"])

    result = run_study(
        config,
        output_dir=tmp_path,
        study_name="synthetic",
        n_trials=3,
        objective=objective,
    )
    assert len(result.trials) == 3
    assert result.best_value == max(trial.value for trial in result.trials)
    assert result.best_params
    payload = json.loads((tmp_path / "synthetic.json").read_text())
    assert payload["best_params"] == result.best_params
    assert [trial.seed for trial in result.trials] == [10, 11, 12]
