"""Wiring test for the Seq-Global tuning objective (W1, PREREG-phase2 §5b).

The objective must search the MF-family breadth (lr, weight decay), train with
``train_mf_global_model`` rather than any BP/joint trainer, pass the reduced HPO
epoch budget through ``early_stopping.max_epochs``, and score with the shared MF
evaluator. Anything else would tune a different arm than the one that runs.
"""

import optuna
import torch

from src.tuning import optuna_objective_mfglobal as mod


def test_objective_mfglobal_wiring(monkeypatch):
    calls = {}

    def fake_loaders(**kwargs):
        return object(), object(), object()

    def fake_model(config, device):
        return torch.nn.Linear(2, 2), (lambda x: x)

    def fake_train(**kwargs):
        calls["lr"] = kwargs["config"]["algorithm_params"]["lr"]
        calls["wd"] = kwargs["config"]["algorithm_params"]["weight_decay"]
        calls["max_epochs"] = kwargs["config"]["early_stopping"]["max_epochs"]
        return 0.0

    def fake_eval(**kwargs):
        return {"eval_accuracy": 77.5}

    monkeypatch.setattr(mod, "get_dataloaders", fake_loaders)
    monkeypatch.setattr(mod, "get_model_and_adapter", fake_model)
    monkeypatch.setattr(mod, "train_mf_global_model", fake_train)
    monkeypatch.setattr(mod, "evaluate_mf_model", fake_eval)

    base_config = {
        "general": {"seed": 42, "backend": "local", "device": "cpu"},
        "algorithm_params": {"optimizer_type": "Adam"},
        "early_stopping": {"enabled": False, "max_epochs": 100},
        "tuning": {
            "enabled": True,
            "direction": "maximize",
            "metric": "val_accuracy",
            "max_epochs": 2,
            "lr_range": [1.0e-5, 1.0e-2],
            "wd_range": [1.0e-6, 1.0e-3],
        },
    }

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: mod.objective_mfglobal(trial, base_config), n_trials=1)

    assert study.best_value == 77.5
    assert 1.0e-5 <= study.best_params["lr"] <= 1.0e-2
    assert 1.0e-6 <= study.best_params["wd"] <= 1.0e-3
    assert calls["lr"] == study.best_params["lr"]
    assert calls["wd"] == study.best_params["wd"]
    assert calls["max_epochs"] == 2
