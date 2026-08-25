"""Characterization of CaFo trainer logging/metric surface (WP9 guard).

Pins the observable log-line formats and per-epoch metric-dict keys of
``train_cafo_model`` before migrating its optimizer construction to
``loop_support.build_optimizer``, so the refactor can be verified
behavior-preserving. Run against the pre-migration code first.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

import beyond_backprop.algorithms.cafo as cafo_module
from beyond_backprop.algorithms.cafo import train_cafo_model
from beyond_backprop.architectures.cafo_cnn import CaFo_CNN

CPU = torch.device("cpu")


def test_cafo_trainer_log_lines_and_metric_keys_are_stable(monkeypatch, caplog):
    torch.manual_seed(0)
    config = {
        "experiment_name": "cafo-loop",
        "model": {"name": "synthetic", "params": {}},
        "data": {"num_classes": 2, "input_channels": 1, "image_size": 2},
        "training": {"epochs": 1, "log_interval": 1},
        "algorithm_params": {
            "train_blocks": True,
            "block_training_epochs": 1,
            "num_epochs_per_block": 1,
            "log_interval": 1,
            "block_lr": 0.001,
            "predictor_lr": 0.001,
        },
        "checkpointing": {},
    }
    model = CaFo_CNN(input_channels=1, block_channels=[2], image_size=2, num_classes=2)
    dataset = TensorDataset(torch.rand(4, 1, 2, 2), torch.zeros(4, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)

    logged: list[dict] = []
    real_log_metrics = cafo_module.log_metrics
    monkeypatch.setattr(
        cafo_module,
        "log_metrics",
        lambda metrics, wandb_run=None, commit=False: (
            logged.append(dict(metrics)),
            real_log_metrics(metrics, wandb_run=wandb_run, commit=commit),
        )[1],
    )

    with caplog.at_level("INFO", logger="beyond_backprop.algorithms.cafo"):
        peak = train_cafo_model(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            device=CPU,
        )

    assert isinstance(peak, float)
    messages = [record.getMessage() for record in caplog.records]
    assert "--- Starting CaFo Block Training (DFA) Phase ---" in messages
    assert "--- Starting CaFo Predictor Training Phase for 1 blocks ---" in messages
    assert "Starting CaFo training for Predictor_1 (Block 1 frozen)" in messages
    assert any("Skipping block training phase." in m for m in messages) or any(
        "Block Training" in m for m in messages
    )

    # Every metric dict carries global_step; epoch summaries expose the
    # legacy Train Loss / Val keys.
    assert logged and all("global_step" in d for d in logged)
    assert any(
        any("Loss_Epoch" in key or "Train_Loss" in key for key in d) for d in logged
    )
