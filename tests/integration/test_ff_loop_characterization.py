"""Characterization of the FF trainer logging/metric surface (WP9 guard).

Pins the observable log-line formats, per-batch and per-epoch metric-dict
key sets, the RUN-002 skip-threshold abort message, and the checkpoint
contract of ``train_ff_model`` before migrating its epoch loop onto
``run_epochs``, so the refactor can be verified behavior-preserving.
Run against the pre-migration code first.

Both ``ff``-module and ``loop_support`` ``log_metrics`` references are
captured into one list: the per-batch emission moves from ff.py into the
skeleton during the WP9 migration, and these pins must hold unchanged on
both sides of that move.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import beyond_backprop.algorithms.ff as ff_module
import beyond_backprop.training.loop_support as loop_support
from beyond_backprop.algorithms.ff import train_ff_model
from beyond_backprop.architectures.ff_mlp import FF_MLP

CPU = torch.device("cpu")

BATCH_KEYS = {
    "global_step",
    "FF_Hinton/Train_Loss_Batch",
    "FF_Hinton/FF_Loss_Batch",
    "FF_Hinton/PeerNorm_Loss_Batch",
    "FF_Hinton/Cls_Loss_Batch",
    "FF_Hinton/Cls_Acc_Batch",
    "Layer_1/FF_Acc_Batch",
}

EPOCH_KEYS = {
    "global_step",
    "FF_Hinton/Train_Loss_Epoch",
    "FF_Hinton/FF_Loss_Epoch",
    "FF_Hinton/PeerNorm_Loss_Epoch",
    "FF_Hinton/Cls_Loss_Epoch",
    "FF_Hinton/Cls_Acc_Epoch",
    "FF_Hinton/Val_Acc_Epoch",
    "FF_Hinton/Epoch_Duration_Sec",
    "FF_Hinton/LR_FF_Layers",
    "FF_Hinton/LR_Downstream",
    "FF_Hinton/Epoch",
    "FF_Hinton/Peak_GPU_Mem_Epoch_MiB",
    "Layer_1/FF_Acc_EpochAvg",
}


def _config(**training_overrides: Any) -> dict[str, Any]:
    training: dict[str, Any] = {"epochs": 2, "log_interval": 1}
    training.update(training_overrides)
    return {
        "experiment_name": "synthetic",
        "model": {"name": "synthetic", "params": {"hidden_dims": [3]}},
        "data": {"num_classes": 2, "input_channels": 1, "image_size": 2},
        "training": training,
        "algorithm_params": {},
        "checkpointing": {},
    }


def _loaders() -> tuple[DataLoader, DataLoader]:
    dataset = TensorDataset(torch.rand(4, 1, 2, 2), torch.zeros(4, dtype=torch.long))
    return DataLoader(dataset, batch_size=2), DataLoader(dataset, batch_size=4)


def _capture_log_metrics(monkeypatch: pytest.MonkeyPatch) -> list[tuple[dict, bool]]:
    """Capture metric dicts emitted via ff.py AND via the loop skeleton."""
    emitted: list[tuple[dict, bool]] = []

    def record(metrics: dict, wandb_run: Any = None, commit: bool = False) -> None:
        del wandb_run
        emitted.append((dict(metrics), commit))

    monkeypatch.setattr(ff_module, "log_metrics", record)
    monkeypatch.setattr(loop_support, "log_metrics", record)
    return emitted


def test_ff_trainer_log_lines_and_metric_keys_are_stable(monkeypatch, caplog):
    """ES-disabled run: pin log shapes plus exact batch/epoch metric keys."""
    torch.manual_seed(0)
    config = _config(early_stopping_enabled=False)
    model = FF_MLP(config, CPU)
    train_loader, val_loader = _loaders()
    emitted = _capture_log_metrics(monkeypatch)

    with caplog.at_level("DEBUG", logger="beyond_backprop.algorithms.ff"):
        peak_memory = train_ff_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=CPU,
            step_ref=[-1],
        )

    assert isinstance(peak_memory, float)
    messages = [record.getMessage() for record in caplog.records]
    assert (
        "Starting Forward-Forward (Hinton style) training using modified FF_MLP."
        in messages
    )
    assert "Early stopping disabled." in messages
    # Two-epoch run: per-epoch LR cooldown DEBUG lines and validation INFO.
    assert any("Epoch 1/2: LR Update - FF=" in m and ", DS=" in m for m in messages)
    assert any(m.startswith("FF Validation Epoch 1/2 - Accuracy: ") for m in messages)
    assert any(
        m.startswith("FF Epoch 1/2 | Train Loss: ")
        and "| Peak Mem: 0.0 MiB | Duration: " in m
        for m in messages
    )
    assert any(
        m.startswith("Finished Forward-Forward (Hinton) training loop. Total time: ")
        for m in messages
    )
    assert any(
        m.startswith("NOTE: Reference implementation used PyTorch 1.11.")
        for m in messages
    )

    # Metric-dict shapes: global_step first, commit=True, exact key sets
    # (single hidden layer -> exactly the Layer_1 keys).
    assert emitted
    assert all(list(metrics)[0] == "global_step" for metrics, _ in emitted)
    assert all(commit is True for _, commit in emitted)
    batch_dicts = [m for m, _ in emitted if "FF_Hinton/Train_Loss_Batch" in m]
    epoch_dicts = [m for m, _ in emitted if "FF_Hinton/Train_Loss_Epoch" in m]
    assert batch_dicts and all(set(m) == BATCH_KEYS for m in batch_dicts)
    assert epoch_dicts and all(set(m) == EPOCH_KEYS for m in epoch_dicts)


def test_ff_early_stopping_message_and_stop_epoch_checkpoint_suppression(
    monkeypatch, caplog, tmp_path
):
    """ES trigger pins the three-line stop message and the stop-timing
    checkpoint boundary: the triggering epoch writes NO checkpoint."""
    torch.manual_seed(0)
    config = _config(
        early_stopping_enabled=True,
        early_stopping_patience=1,
        early_stopping_metric="FF_Hinton/Val_Acc_Epoch",
        early_stopping_mode="max",
    )
    config["checkpointing"] = {"checkpoint_dir": str(tmp_path)}
    model = FF_MLP(config, CPU)
    train_loader, val_loader = _loaders()
    _capture_log_metrics(monkeypatch)
    step_ref = [-1]

    with caplog.at_level("INFO", logger="beyond_backprop.algorithms.ff"):
        train_ff_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=CPU,
            step_ref=step_ref,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        m.startswith(
            "Early stopping enabled: Metric Key='ff_hinton/val_acc_epoch', "
            "Patience=1, Mode='max'"
        )
        for m in messages
    )
    assert any(
        m.startswith("Epoch 1: Early stopping metric improved to ") for m in messages
    )
    assert any(
        m.startswith("Epoch 2: Early stopping metric did not improve. Patience: ")
        for m in messages
    )
    # Identical validation data each epoch: epoch 1 improves, epoch 2 trips
    # the three-line early-stopping sequence verbatim.
    assert "--- Early Stopping Triggered ---" in messages
    assert any(
        m.startswith(
            "Metric 'ff_hinton/val_acc_epoch' did not improve for 1 epochs (Best: "
        )
        and m.endswith(").")
        for m in messages
    )
    assert "Stopping training at epoch 2." in messages
    # Training stopped at epoch 2 of a 2-batches-per-epoch loader.
    assert step_ref[0] == 3

    # Epoch 1 checkpointed; the ES-triggering epoch 2 did NOT.
    assert (tmp_path / "ff_checkpoint_epoch_1.pth").exists()
    assert not (tmp_path / "ff_checkpoint_epoch_2.pth").exists()
    payload = torch.load(tmp_path / "ff_checkpoint_epoch_1.pth", weights_only=False)
    assert set(payload) >= {
        "epoch",
        "state_dict",
        "optimizer",
        "best_metric_value",
        "val_accuracy",
    }
    best_files = list(tmp_path.glob("ff_synthetic_best.pth"))
    assert len(best_files) == 1


def test_ff_skip_threshold_aborts_with_pinned_message(monkeypatch):
    """RUN-002: exception-driven skips past max(1, len(loader)//100) abort;
    step_ref counts skipped batches too."""
    torch.manual_seed(0)
    config = _config(early_stopping_enabled=False)
    config["training"]["epochs"] = 1
    model = FF_MLP(config, CPU)
    dataset = TensorDataset(torch.rand(4, 1, 2, 2), torch.zeros(4, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise ValueError("input generation exploded")

    monkeypatch.setattr(ff_module, "generate_hinton_inputs", boom)
    step_ref = [-1]

    with pytest.raises(
        RuntimeError,
        match=r"FF training skipped 2 batches \(threshold 1\); aborting run\.",
    ):
        train_ff_model(
            model=model,
            train_loader=loader,
            val_loader=None,
            config=config,
            device=CPU,
            step_ref=step_ref,
        )

    # Both batches were attempted (and counted) before the abort
    # (step_ref starts at -1, so two attempts land on 1).
    assert step_ref[0] == 1
