"""NaN/Inf batch accounting in trainers (RUN-002 contract)."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import beyond_backprop.algorithms.ff as ff_module
import beyond_backprop.algorithms.mf as mf_module
from beyond_backprop.algorithms.ff import FFAdapter, train_ff_model
from beyond_backprop.algorithms.mf import train_mf_matrix_only, train_mf_model
from beyond_backprop.architectures.ff_mlp import FF_MLP
from beyond_backprop.architectures.mf_mlp import MF_MLP
from beyond_backprop.contracts import TrainingContext

CPU = torch.device("cpu")


def _ff_config(epochs: int = 2) -> dict[str, Any]:
    return {
        "experiment_name": "diagnostics",
        "model": {"name": "synthetic", "params": {"hidden_dims": [3]}},
        "data": {"num_classes": 2, "input_channels": 1, "image_size": 2},
        "data_loader": {"batch_size": 2},
        "training": {
            "epochs": epochs,
            "log_interval": 1,
            "early_stopping_enabled": False,
        },
        "algorithm_params": {},
        "checkpointing": {},
    }


def _mf_config() -> dict[str, Any]:
    return {
        "experiment_name": "diagnostics",
        "model": {"name": "synthetic", "params": {"hidden_dims": [3]}},
        "data": {"num_classes": 2},
        "training": {"epochs": 1, "log_interval": 1},
        "algorithm_params": {"epochs_per_layer": 2},
        "checkpointing": {},
    }


def _loaders():
    dataset = TensorDataset(torch.rand(8, 1, 2, 2), torch.zeros(8, dtype=torch.long))
    return (
        DataLoader(dataset, batch_size=2),
        DataLoader(dataset, batch_size=4),
    )


def test_ff_exception_skips_raise_past_threshold(monkeypatch):
    torch.manual_seed(0)
    model = FF_MLP(_ff_config(), CPU)
    train_loader, val_loader = _loaders()

    def poison(*args, **kwargs):
        raise RuntimeError("poisoned input generation")

    monkeypatch.setattr(ff_module, "generate_hinton_inputs", poison)

    # 4 batches -> threshold max(1, 4 // 100) = 1; the second skip raises.
    with pytest.raises(RuntimeError, match="threshold"):
        train_ff_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=_ff_config(),
            device=CPU,
        )


def test_ff_single_poisoned_batch_is_counted_not_fatal(monkeypatch):
    torch.manual_seed(0)
    model = FF_MLP(_ff_config(), CPU)
    train_loader, val_loader = _loaders()

    real_generate = ff_module.generate_hinton_inputs
    calls = {"n": 0}

    def poison_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("poisoned once")
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(ff_module, "generate_hinton_inputs", poison_once)

    diagnostics: dict[str, float] = {}
    peak = train_ff_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=_ff_config(),
        device=CPU,
        diagnostics=diagnostics,
    )
    assert isinstance(peak, float)
    assert diagnostics["skipped_batches"] == 1.0


def test_ff_adapter_surfaces_diagnostics_as_unmeasured_metrics(monkeypatch):
    torch.manual_seed(0)
    config = _ff_config()
    model = FF_MLP(config, CPU)
    train_loader, val_loader = _loaders()

    real_generate = ff_module.generate_hinton_inputs
    calls = {"n": 0}

    def poison_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("poisoned once")
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(ff_module, "generate_hinton_inputs", poison_once)

    adapter = FFAdapter()
    context = TrainingContext(config, model, train_loader, val_loader, CPU)
    result = adapter.fit(context)
    entry = result.metrics["skipped_batches"]
    assert entry.value == 1.0
    assert entry.provenance.source == "trainer"
    assert entry.provenance.measured is False


def test_mf_second_nan_epoch_raises(monkeypatch):
    torch.manual_seed(0)
    config = _mf_config()
    model = MF_MLP(input_dim=4, hidden_dims=[3], num_classes=2)
    train_loader, val_loader = _loaders()

    nan = torch.tensor(float("nan"))
    monkeypatch.setattr(mf_module, "mf_local_loss_fn", lambda *a, **k: nan)

    with pytest.raises(RuntimeError, match="NaN/Inf"):
        train_mf_model(
            model=model,
            train_loader=train_loader,
            config=config,
            device=CPU,
            input_adapter=lambda tensor: tensor.view(tensor.shape[0], -1),
            val_loader=None,
        )


def test_mf_single_nan_break_is_counted(monkeypatch):
    torch.manual_seed(0)
    model = MF_MLP(input_dim=4, hidden_dims=[3], num_classes=2)
    train_loader, _ = _loaders()

    nan = torch.tensor(float("nan"))
    monkeypatch.setattr(mf_module, "mf_local_loss_fn", lambda *a, **k: nan)

    matrix_optimizer = torch.optim.Adam(
        [model.get_projection_matrix(0)], lr=1e-3, weight_decay=0.0
    )
    diagnostics: dict[str, float] = {}
    train_mf_matrix_only(
        model=model,
        matrix_index=0,
        optimizer=matrix_optimizer,
        criterion=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        epochs=1,
        device=CPU,
        input_adapter=lambda tensor: tensor.view(tensor.shape[0], -1),
        early_stopping_config={},
        diagnostics=diagnostics,
    )
    assert diagnostics["nan_loss_breaks"] == 1.0
