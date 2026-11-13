"""Guards for the W1 control code paths.

These are wiring tests, not accuracy tests: BP-bestval must restore the
best-validation state without touching disk, and BP-sched tuning must step the
same cosine schedule as the confirmatory arm.
"""

import optuna
import torch
import torch.nn as nn

from src.baselines import bp as bp_mod
from src.tuning import optuna_objective as tune_mod


def test_bp_bestval_restores_best_in_memory(monkeypatch):
    model = nn.Linear(1, 1, bias=False)
    seen = {"epoch": 0}
    accuracies = [10.0, 90.0, 20.0]

    def fake_train_epoch(*args, **kwargs):
        seen["epoch"] += 1
        with torch.no_grad():
            model.weight.fill_(float(seen["epoch"]))
        return 1.0, 50.0, 0.0

    def fake_eval(*args, **kwargs):
        acc = accuracies[seen["epoch"] - 1]
        return 1.0, acc

    monkeypatch.setattr(bp_mod, "train_bp_epoch", fake_train_epoch)
    monkeypatch.setattr(bp_mod, "evaluate_bp_model", fake_eval)

    cfg = {
        "experiment_name": "unit_bp_bestval",
        "training": {"criterion": "CrossEntropyLoss", "log_interval": 999},
        "optimizer": {"type": "AdamW", "lr": 1.0e-3, "weight_decay": 0.0},
        "early_stopping": {
            "enabled": True,
            "metric": "val_loss",
            "mode": "min",
            "min_delta": 0.0,
            "patience": 10,
            "max_epochs": 3,
        },
        "checkpointing": {
            "checkpoint_dir": None,
            "restore_best_in_memory": True,
            "save_best_metric": "bp_val_accuracy",
        },
        "logging": {"wandb": {"use_wandb": False}},
    }

    bp_mod.train_bp_model(
        model,
        train_loader=object(),
        val_loader=object(),
        config=cfg,
        device=torch.device("cpu"),
        input_adapter=None,
    )

    assert cfg["_run_stats"]["epochs_completed"] == 3
    assert model.weight.item() == 2.0, "best-accuracy state was not restored"


def test_bp_tuning_steps_configured_scheduler(monkeypatch):
    captured = {}

    def fake_loaders(**kwargs):
        return object(), object(), object()

    def fake_model(config, device):
        return nn.Linear(2, 2), (lambda x: x)

    def fake_train(**kwargs):
        return None

    def fake_eval(**kwargs):
        return 1.0, 50.0

    original_optimizer = tune_mod._create_bp_optimizer

    def capture_optimizer(model, trial_params, optimizer_cfg):
        optimizer = original_optimizer(model, trial_params, optimizer_cfg)
        captured["optimizer"] = optimizer
        captured["initial_lr"] = optimizer.param_groups[0]["lr"]
        return optimizer

    monkeypatch.setattr(tune_mod, "get_dataloaders", fake_loaders)
    monkeypatch.setattr(tune_mod, "get_model_and_adapter", fake_model)
    monkeypatch.setattr(tune_mod, "train_bp_epoch", fake_train)
    monkeypatch.setattr(tune_mod, "evaluate_bp_model", fake_eval)
    monkeypatch.setattr(tune_mod, "_create_bp_optimizer", capture_optimizer)

    cfg = {
        "general": {"seed": 42, "backend": "local", "device": "cpu"},
        "algorithm": {"name": "BP"},
        "data": {"name": "CIFAR10", "root": "./data"},
        "optimizer": {"type": "AdamW"},
        "training": {"scheduler": "CosineAnnealingLR", "scheduler_params": {}},
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
    study.optimize(lambda trial: tune_mod.objective(trial, cfg), n_trials=1)

    final_lr = captured["optimizer"].param_groups[0]["lr"]
    assert final_lr < captured["initial_lr"] * 0.6
