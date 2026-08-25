"""Characterization tests: FF final-eval weights vs checkpoint configuration.

These pin the PRE-WP2 divergence: which weights produce the final evaluation
depends on whether ``checkpointing.checkpoint_dir`` is configured
(decision BEST-001 in docs/refactoring-decisions.md supersedes this).

Validation accuracies are scripted per epoch (monkeypatching the module-level
``evaluate_ff_model`` used by the trainer's validation hook) so EarlyStopping
follows a deterministic trajectory: epoch 1 scores 90.0 (best), epochs 2-3
score 50.0. Per-epoch weight snapshots are captured inside the validation hook,
so each assertion compares exact tensors.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

import beyond_backprop.algorithms.ff as ff_module
from beyond_backprop.algorithms.ff import train_ff_model
from beyond_backprop.architectures.ff_mlp import FF_MLP

DEVICE = torch.device("cpu")


def _config(checkpoint_dir: str | None) -> dict[str, Any]:
    return {
        "experiment_name": "synthetic",
        "model": {"name": "synthetic", "params": {"hidden_dims": [3]}},
        "data": {"num_classes": 2, "input_channels": 1, "image_size": 2},
        "data_loader": {"batch_size": 2},
        "training": {
            "epochs": 3,
            "log_interval": 1,
            "early_stopping_enabled": True,
            "early_stopping_patience": 3,
        },
        "algorithm_params": {},
        "checkpointing": ({"checkpoint_dir": checkpoint_dir} if checkpoint_dir else {}),
    }


def _loaders() -> tuple[DataLoader, DataLoader]:
    dataset = TensorDataset(torch.rand(4, 1, 2, 2), torch.tensor([0, 1, 0, 1]))
    return DataLoader(dataset, batch_size=2), DataLoader(dataset, batch_size=4)


def _state(model: FF_MLP) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _states_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return set(a) == set(b) and all(torch.equal(a[key], b[key]) for key in a)


def _run(
    monkeypatch, checkpoint_dir: str | None
) -> tuple[FF_MLP, list[dict[str, torch.Tensor]]]:
    """Train the synthetic FF model; snapshot weights at each validation."""
    torch.manual_seed(0)
    config = _config(checkpoint_dir)
    model = FF_MLP(config, DEVICE)
    train_loader, val_loader = _loaders()

    scripted_val_accuracy = iter([90.0, 50.0, 50.0])
    epoch_snapshots: list[dict[str, torch.Tensor]] = []

    def fake_validate(m, loader, device):
        epoch_snapshots.append(_state(m))
        return {
            "eval_accuracy": next(scripted_val_accuracy),
            "eval_loss": float("nan"),
        }

    monkeypatch.setattr(ff_module, "evaluate_ff_model", fake_validate)

    train_ff_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=DEVICE,
    )
    return model, epoch_snapshots


def test_pre_wp2_without_checkpoint_dir_final_eval_uses_last_epoch_weights(
    tmp_path, monkeypatch
):
    model, snapshots = _run(monkeypatch, None)
    assert len(snapshots) == 3
    assert _states_equal(_state(model), snapshots[-1])


def test_pre_wp2_with_checkpoint_dir_final_eval_uses_best_val_weights(
    tmp_path, monkeypatch
):
    model, snapshots = _run(monkeypatch, str(tmp_path / "ckpt"))
    assert len(snapshots) == 3
    # Best validation accuracy (90.0) was epoch 1.
    assert _states_equal(_state(model), snapshots[0])
