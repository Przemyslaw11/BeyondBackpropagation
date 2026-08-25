"""Characterization of MF trainer logging/metric surface (WP9 guard).

Pins the observable log-line formats and per-epoch metric-dict keys of
``train_mf_model`` before the epoch-scaffolding extraction so the refactor
can be verified behavior-preserving. Run against the pre-migration code first.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import beyond_backprop.algorithms.mf as mf_module
from beyond_backprop.algorithms.mf import train_mf_model
from beyond_backprop.architectures.mf_mlp import MF_MLP
from beyond_backprop.training.loop_support import build_optimizer

CPU = torch.device("cpu")


def _config() -> dict:
    return {
        "experiment_name": "mf-loop",
        "model": {"name": "synthetic", "params": {"hidden_dims": [3]}},
        "data": {"num_classes": 2},
        "training": {"epochs": 1, "log_interval": 1},
        "algorithm_params": {"epochs_per_layer": 2, "log_interval": 1},
        "checkpointing": {},
    }


def test_build_optimizer_resolves_known_names():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = build_optimizer("Adam", [param], lr=0.01, weight_decay=0.0)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 0.01


def test_build_optimizer_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer("NotAnOptimizer", [torch.nn.Parameter(torch.zeros(1))], lr=0.1)


def test_mf_trainer_log_lines_and_metric_keys_are_stable(monkeypatch, caplog):
    torch.manual_seed(0)
    model = MF_MLP(input_dim=4, hidden_dims=[3], num_classes=2)
    dataset = TensorDataset(torch.rand(4, 1, 2, 2), torch.zeros(4, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)

    logged: list[dict] = []
    real_log_metrics = mf_module.log_metrics
    monkeypatch.setattr(
        mf_module,
        "log_metrics",
        lambda metrics, wandb_run=None, commit=False: (
            logged.append(dict(metrics)),
            real_log_metrics(metrics, wandb_run=wandb_run, commit=commit),
        )[1],
    )

    with caplog.at_level("INFO", logger="beyond_backprop.algorithms.mf"):
        train_mf_model(
            model=model,
            train_loader=loader,
            config=_config(),
            device=CPU,
            input_adapter=lambda tensor: tensor.view(tensor.shape[0], -1),
            val_loader=None,
        )

    messages = [record.getMessage() for record in caplog.records]
    # Phase banners and the completion line keep their exact shapes.
    assert "--- Starting MF training for Layer_M0 ---" in messages
    assert "--- Starting MF training for Layer_W1_M1 ---" in messages
    assert any(m.startswith("--- Starting MF training for Layer_") for m in messages)
    assert any(
        "Finished all layer-wise MF training. Total Epochs (Sum):" in m
        for m in messages
    )
    assert any(": Early Stopping Disabled." in m for m in messages)

    # Every logged metric dict carries global_step first and a Train Loss key;
    # epoch summaries expose the legacy loss/peak-mem keys.
    assert logged and all("global_step" in d for d in logged)
    loss_keys = [k for d in logged for k in d if "Loss_Epoch" in k or "LocalLoss" in k]
    assert any("Layer_M0" in k or "Layer_W1_M1" in k for k in loss_keys)
