"""Characterization of CaFo trainer logging/metric surface (WP9 guard).

Pins the observable log-line formats and per-epoch metric-dict keys of
``train_cafo_model`` before migrating its optimizer construction to
``loop_support.build_optimizer``, so the refactor can be verified
behavior-preserving. Run against the pre-migration code first.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, TensorDataset

import beyond_backprop.algorithms.cafo as cafo_module
from beyond_backprop.algorithms.cafo import (
    train_cafo_model,
    train_cafo_predictor_only,
)
from beyond_backprop.architectures.cafo_cnn import CaFo_CNN, CaFoPredictor
from beyond_backprop.training import loop_support

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


def test_cafo_batch_metrics_boundary_is_pinned_at_loop_support(monkeypatch):
    """Pin the skeleton-emitted per-batch metric dict (WP9 migration guard).

    The ``{prefix}/Train_Loss_Batch`` AND ``{prefix}/Train_Acc_Batch``
    emission moved from the cafo.py epoch loop into ``run_epochs``' cadence
    boundary via ``on_log_boundary``, so it is captured via loop_support's
    log_metrics (the cafo-module patch above cannot see it). Shape:
    ``global_step`` first, ``commit=True`` always, both keys present.
    """
    torch.manual_seed(0)
    model = CaFo_CNN(input_channels=1, block_channels=[2], image_size=2, num_classes=2)
    dataset = TensorDataset(torch.rand(4, 1, 2, 2), torch.zeros(4, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    predictor = CaFoPredictor(model.get_predictor_input_dim(0), model.num_classes)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    emitted: list[tuple[dict, bool]] = []
    monkeypatch.setattr(
        loop_support,
        "log_metrics",
        lambda metrics, wandb_run=None, commit=False: emitted.append(
            (dict(metrics), commit)
        ),
    )

    avg_loss, avg_acc, peak_mem, epochs_trained = train_cafo_predictor_only(
        block=model.blocks[0],
        predictor=predictor,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        train_loader=loader,
        val_loader=None,
        epochs=1,
        device=CPU,
        get_block_input_fn=lambda img: img,
        early_stopping_config={},
        block_index=0,
        step_ref=[-1],
    )

    assert emitted, "skeleton must emit per-batch metrics via loop_support"
    keys = {key for metrics, _ in emitted for key in metrics}
    assert "Predictor_1/Train_Loss_Batch" in keys
    assert "Predictor_1/Train_Acc_Batch" in keys
    assert all(list(metrics)[0] == "global_step" for metrics, _ in emitted)
    assert all(commit is True for _, commit in emitted)
    # Return contract: peak mem never sampled (0.0); started-epoch counting.
    assert peak_mem == 0.0
    assert epochs_trained == 1
    assert isinstance(avg_loss, float) and isinstance(avg_acc, float)


def test_cafo_predictor_loop_has_no_nan_abort():
    """Legacy CaFo has no RUN-004 guard: NaN losses train through."""
    torch.manual_seed(0)
    model = CaFo_CNN(input_channels=1, block_channels=[2], image_size=2, num_classes=2)
    dataset = TensorDataset(torch.rand(4, 1, 2, 2), torch.zeros(4, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    predictor = CaFoPredictor(model.get_predictor_input_dim(0), model.num_classes)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    class NaNCriterion(torch.nn.Module):
        def forward(self, predictions, labels):
            del labels
            # Scalar NaN value but keep the autograd graph attached, like a
            # real diverged forward pass (a grad-less or non-scalar tensor
            # would fail backward for reasons unrelated to NaN accounting).
            return predictions.sum() * float("nan")

    avg_loss, _, peak_mem, epochs_trained = train_cafo_predictor_only(
        block=model.blocks[0],
        predictor=predictor,
        optimizer=optimizer,
        criterion=NaNCriterion(),
        train_loader=loader,
        val_loader=None,
        epochs=2,
        device=CPU,
        get_block_input_fn=lambda img: img,
        early_stopping_config={},
        block_index=0,
        step_ref=[-1],
    )

    # No abort and no raise: both started epochs ran despite every batch
    # being NaN (the skeleton's RUN-004 accounting is off for CaFo).
    assert epochs_trained == 2
    assert math.isnan(avg_loss)
    assert peak_mem == 0.0
