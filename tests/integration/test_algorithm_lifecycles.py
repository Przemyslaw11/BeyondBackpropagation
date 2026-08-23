from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from beyond_backprop.algorithms import BPAdapter
from beyond_backprop.checkpointing import CheckpointManager
from beyond_backprop.config import experiment_config_from_mapping
from beyond_backprop.contracts import RunStatus, TrainingContext


def _config() -> object:
    return experiment_config_from_mapping(
        {
            "experiment_name": "bp_lifecycle",
            "general": {"seed": 1, "device": "cpu", "backend": "local"},
            "algorithm": {"name": "BP"},
            "model": {"name": "MF_MLP", "params": {"hidden_dims": [4]}},
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
            "training": {
                "epochs": 2,
                "early_stopping_enabled": True,
                "early_stopping_metric": "bp_val_loss",
                "early_stopping_patience": 2,
                "early_stopping_mode": "min",
            },
            "checkpointing": {},
        }
    )


def test_bp_lifecycle_writes_legacy_names_and_restores_best(tmp_path) -> None:
    config = _config()
    data = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]], [[[0.0, 1.0], [1.0, 0.0]]]])
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(data, labels), batch_size=2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    context = TrainingContext(
        config,
        model,
        loader,
        loader,
        torch.device("cpu"),
        checkpoint_manager=CheckpointManager(tmp_path),
    )
    adapter = BPAdapter()
    result = adapter.fit(context)
    assert result.status is RunStatus.SUCCEEDED
    assert (tmp_path / "bp_checkpoint_epoch_1.pth").exists()
    assert (tmp_path / "bp_bp_lifecycle_best.pth").exists()
    with torch.no_grad():
        next(model.parameters()).add_(100.0)
    adapter.restore_best_state(context, result)
    assert torch.max(next(model.parameters()).detach()) < 100.0
