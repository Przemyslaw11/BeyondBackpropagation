"""Synthetic end-to-end tests driving the real per-algorithm training loops.

Fast CPU loops over tiny random tensors that execute the verbatim trainer
bodies in ``algorithms/{ff,mf,cafo}.py`` (no monkeypatching of the trainer
functions) plus their algorithm-specific evaluation paths. These protect the
fairness-critical loop code and unblock future refactors of it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import ModuleList
from torch.utils.data import DataLoader, TensorDataset

from beyond_backprop.algorithms.cafo import evaluate_cafo_model, train_cafo_model
from beyond_backprop.algorithms.ff import evaluate_ff_model, train_ff_model
from beyond_backprop.algorithms.mf import evaluate_mf_model, train_mf_model
from beyond_backprop.architectures.cafo_cnn import CaFo_CNN
from beyond_backprop.architectures.ff_mlp import FF_MLP
from beyond_backprop.architectures.mf_mlp import MF_MLP

DEVICE = torch.device("cpu")


def _flatten(images: torch.Tensor) -> torch.Tensor:
    return images.view(images.shape[0], -1)


def _base_config(**algorithm_params: Any) -> dict[str, Any]:
    """Minimal legacy-shape config accepted by the trainer functions."""
    return {
        "experiment_name": "synthetic",
        "model": {"name": "synthetic", "params": {"hidden_dims": [3]}},
        "data": {"num_classes": 2, "input_channels": 1, "image_size": 2},
        "data_loader": {"batch_size": 2},
        "training": {
            "epochs": 2,
            "log_interval": 1,
            "early_stopping_enabled": False,
        },
        "algorithm_params": algorithm_params,
        "checkpointing": {},
    }


def _loaders() -> tuple[DataLoader, DataLoader]:
    images = torch.tensor(
        [
            [[[1.0, 0.5], [0.25, 0.0]]],
            [[[0.0, 1.0], [0.5, 0.25]]],
            [[[0.9, 0.1], [0.8, 0.2]]],
            [[[0.2, 0.7], [0.4, 0.6]]],
        ]
    )
    labels = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=2), DataLoader(dataset, batch_size=4)


def test_train_ff_model_runs_two_epochs_on_cpu_and_evaluates():
    torch.manual_seed(0)
    config = _base_config()
    model = FF_MLP(config, DEVICE)
    train_loader, val_loader = _loaders()
    step_ref = [-1]

    peak_memory = train_ff_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=DEVICE,
        input_adapter=_flatten,
        step_ref=step_ref,
    )

    assert isinstance(peak_memory, float)
    assert peak_memory >= 0.0
    # 2 epochs x 2 batches per epoch.
    assert step_ref[0] == 3

    results = evaluate_ff_model(model, val_loader, DEVICE)
    assert 0.0 <= results["eval_accuracy"] <= 100.0


def test_train_ff_model_early_stopping_triggers_on_stagnant_metric():
    torch.manual_seed(0)
    config = _base_config()
    config["training"] = {
        "epochs": 6,
        "log_interval": 1,
        "early_stopping_enabled": True,
        "early_stopping_patience": 1,
        "early_stopping_metric": "FF_Hinton/Val_Acc_Epoch",
        "early_stopping_mode": "max",
    }
    model = FF_MLP(config, DEVICE)
    train_loader, val_loader = _loaders()
    step_ref = [-1]

    train_ff_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=DEVICE,
        input_adapter=_flatten,
        step_ref=step_ref,
    )

    # Identical validation data each epoch cannot strictly improve after the
    # first epoch, so patience-1 early stopping must halt before epoch 6
    # (which would need 11 steps).
    assert step_ref[0] < 11


def test_train_mf_model_runs_layerwise_phases_on_cpu_and_evaluates():
    torch.manual_seed(0)
    config = _base_config(
        epochs_per_layer=1,
        log_interval=1,
        mf_early_stopping_enabled=False,
        lr=0.01,
    )
    model = MF_MLP(input_dim=4, hidden_dims=[3], num_classes=2)
    train_loader, val_loader = _loaders()
    step_ref = [-1]

    peak_memory = train_mf_model(
        model=model,
        train_loader=train_loader,
        config=config,
        device=DEVICE,
        input_adapter=_flatten,
        val_loader=val_loader,
        step_ref=step_ref,
    )

    assert isinstance(peak_memory, float)
    # Two matrix-only phases (M0, W1M1) x 1 epoch x 2 batches -> 4 steps.
    assert step_ref[0] >= 3

    values = evaluate_mf_model(model, val_loader, DEVICE, _flatten)
    assert 0.0 <= values["eval_accuracy"] <= 100.0


def test_train_cafo_model_trains_blocks_and_predictors_then_evaluates():
    torch.manual_seed(0)
    config = _base_config(
        train_blocks=True,
        block_training_epochs=1,
        num_epochs_per_block=1,
        log_interval=1,
        block_lr=0.001,
        predictor_lr=0.001,
    )
    model = CaFo_CNN(
        input_channels=1, block_channels=[2], image_size=2, num_classes=2
    )
    train_loader, val_loader = _loaders()
    step_ref = [-1]

    peak_memory = train_cafo_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=DEVICE,
        step_ref=step_ref,
    )

    assert isinstance(peak_memory, float)
    trained_predictors = getattr(model, "trained_predictors", None)
    assert isinstance(trained_predictors, ModuleList)
    assert len(trained_predictors) == len(model.blocks) == 1

    values = evaluate_cafo_model(model, val_loader, DEVICE)
    assert 0.0 <= values["eval_accuracy"] <= 100.0


def test_train_ff_model_writes_checkpoints_through_legacy_saver(tmp_path):
    torch.manual_seed(0)
    config = _base_config()
    config["checkpointing"] = {"checkpoint_dir": str(tmp_path)}
    model = FF_MLP(config, DEVICE)
    train_loader, val_loader = _loaders()

    train_ff_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=DEVICE,
        input_adapter=_flatten,
    )

    assert (tmp_path / "ff_checkpoint_epoch_2.pth").exists()
    # The epoch file keeps the full legacy wrapper payload...
    payload = torch.load(tmp_path / "ff_checkpoint_epoch_2.pth", weights_only=False)
    assert set(payload) >= {"epoch", "state_dict", "optimizer", "best_metric_value"}
    # ...while the best file remains a raw state_dict (legacy restart contract).
    best_files = list(tmp_path.glob("ff_synthetic_best.pth"))
    assert len(best_files) == 1
    best = torch.load(best_files[0], weights_only=False)
    assert best and all(isinstance(value, torch.Tensor) for value in best.values())
    # No atomic-write temp files left behind.
    assert not [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
