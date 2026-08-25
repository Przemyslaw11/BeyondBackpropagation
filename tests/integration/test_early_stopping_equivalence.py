"""D1 early-stopping equivalence tests (characterization -> unification).

These tests feed identical synthetic validation-metric streams through the
REAL trainer loops (only the per-epoch evaluators are scripted) and compare
the observed stop points with

1. the legacy inline decision rule as it existed before D1 (NaN => bad epoch;
   improve iff strictly ``value > best + delta`` / ``value < best - delta``;
   stop when bad epochs ``>=`` patience), and
2. the canonical ``training.early_stopping.EarlyStopping`` used by BP.

Status after the D1 swap (step 2): all four trainers use EarlyStopping with
``patience - 1``, which reproduces the legacy ``>=`` boundary exactly.
Consequences, pinned here:

* Finite/NaN streams: trainer stop points are byte-identical to the legacy
  rule (asserted for every stream/patience combination below).
* +/-INF EDGE CASE RESOLVED: all algorithms now follow canonical semantics
  (any non-finite value is a bad epoch). The pre-swap divergence is recorded
  in ``test_inf_now_follows_canonical_semantics_after_d1_unification``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import beyond_backprop.algorithms.cafo as cafo_module
import beyond_backprop.algorithms.ff as ff_module
import beyond_backprop.algorithms.mf as mf_module
from beyond_backprop.algorithms.cafo import train_cafo_model
from beyond_backprop.algorithms.ff import train_ff_model
from beyond_backprop.algorithms.mf import train_mf_matrix_only
from beyond_backprop.architectures.cafo_cnn import CaFo_CNN
from beyond_backprop.architectures.ff_mlp import FF_MLP
from beyond_backprop.architectures.mf_mlp import MF_MLP
from beyond_backprop.training.early_stopping import EarlyStopping

DEVICE = torch.device("cpu")
MAX_EPOCHS = 40

NAN = float("nan")
INF = float("inf")

# Streams long enough that any stopping rule under test triggers within them.
STREAMS: dict[str, list[float]] = {
    "stagnant": [1.0] * 14,
    "improving": [10.0 - 0.25 * i for i in range(14)],
    "recovery_then_stagnation": [
        5.0,
        4.9,
        4.95,
        4.8,
        4.85,
        4.99,
        4.99,
        4.99,
        4.99,
        4.99,
        4.99,
        4.99,
        4.99,
        4.99,
    ],
    "nan_gaps": [
        5.0,
        NAN,
        4.9,
        NAN,
        NAN,
        4.85,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
    ],
    "exact_tie_with_delta": [
        1.0,
        1.0,
        1.0,
        0.4,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
}


def legacy_stop_epoch(
    stream: list[float], patience: int, mode: str, min_delta: float
) -> int | None:
    """Stop point of the verbatim inline rule; None if it never stops."""
    best = -INF if mode == "max" else INF
    bad = 0
    for epoch, value in enumerate(stream[:MAX_EPOCHS], start=1):
        if math.isnan(value):
            bad += 1
        elif (
            (value > best + min_delta) if mode == "max" else (value < best - min_delta)
        ):
            best = value
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            return epoch
    return None


def canonical_stop_epoch(
    stream: list[float], patience: int, mode: str, min_delta: float
) -> int | None:
    """Stop point of the canonical EarlyStopping class; None if never."""
    stopping = EarlyStopping(patience=patience, mode=mode, min_delta=min_delta)
    for epoch, value in enumerate(stream[:MAX_EPOCHS], start=1):
        if stopping.update(float(value), epoch=epoch):
            return epoch
    return None


def assert_characterized(
    run: Callable[[], int],
    stream: list[float],
    patience: int,
    mode: str,
    delta: float,
    *,
    canonical_exactly_one_later: bool = False,
) -> None:
    """Assert the real trainer matches the legacy rule exactly, then document
    how the canonical class relates for the same stream.

    With ``canonical_exactly_one_later`` (flat-tail streams) the canonical
    class must stop exactly one epoch later; otherwise (streams that can still
    improve after the inline stop) it must merely stop strictly later or never.
    """
    inline_epochs = run()
    expected = legacy_stop_epoch(stream, patience, mode, delta)
    assert inline_epochs == (expected or MAX_EPOCHS)

    canonical = canonical_stop_epoch(stream, patience, mode, delta)
    if expected is None:
        assert canonical is None
    elif canonical_exactly_one_later:
        assert canonical == inline_epochs + 1  # documented D1 off-by-one
    else:
        assert canonical is not None and canonical > inline_epochs


def _loaders() -> tuple[DataLoader, DataLoader]:
    dataset = TensorDataset(torch.rand(4, 1, 2, 2), torch.tensor([0, 1, 0, 1]))
    return DataLoader(dataset, batch_size=2), DataLoader(dataset, batch_size=4)


def _cycler(stream: list[float]) -> tuple[Callable[[], float], list[int]]:
    """Return a value popper and the list that records how often it was read."""
    calls: list[int] = []
    state = {"i": 0}

    def next_value() -> float:
        calls.append(state["i"])
        value = stream[state["i"] % len(stream)]
        state["i"] += 1
        return value

    return next_value, calls


def run_ff(monkeypatch, stream, patience, mode, delta) -> int:
    next_value, calls = _cycler(stream)

    def fake_eval(model: object, loader: object, device: object) -> dict[str, float]:
        return {"eval_accuracy": next_value(), "eval_loss": NAN}

    monkeypatch.setattr(ff_module, "evaluate_ff_model", fake_eval)
    config = {
        "experiment_name": "es-char",
        "model": {"name": "FF_MLP", "params": {"hidden_dims": [3]}},
        "data": {"num_classes": 2, "input_channels": 1, "image_size": 2},
        "data_loader": {},
        "checkpointing": {},
        "training": {
            "epochs": MAX_EPOCHS,
            "log_interval": 9999,
            "early_stopping_enabled": True,
            "early_stopping_metric": "FF_Hinton/Val_Acc_Epoch",
            "early_stopping_patience": patience,
            "early_stopping_mode": mode,
            "early_stopping_min_delta": delta,
        },
        "algorithm_params": {},
    }
    train_loader, val_loader = _loaders()
    train_ff_model(
        model=FF_MLP(config, DEVICE),
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=DEVICE,
    )
    return len(calls)


def run_mf_matrix_only(monkeypatch, stream, patience, mode, delta) -> int:
    next_value, calls = _cycler(stream)

    def fake_eval(**kwargs: object) -> float:
        return next_value()

    monkeypatch.setattr(mf_module, "evaluate_mf_local_loss", fake_eval)
    model = MF_MLP(input_dim=4, hidden_dims=[3], num_classes=2)
    projection = model.get_projection_matrix(0)
    projection.requires_grad_(True)
    optimizer = torch.optim.Adam([projection], lr=1e-3)
    train_loader, val_loader = _loaders()
    train_mf_matrix_only(
        model=model,
        matrix_index=0,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        epochs=MAX_EPOCHS,
        device=DEVICE,
        input_adapter=lambda images: images.view(images.shape[0], -1),
        early_stopping_config={
            "mf_early_stopping_enabled": True,
            "mf_early_stopping_patience": patience,
            "mf_early_stopping_min_delta": delta,
        },
        val_loader=val_loader,
    )
    return len(calls)


def run_cafo_predictor(monkeypatch, stream, patience, mode, delta) -> int:
    next_value, calls = _cycler(stream)

    def fake_eval(*args: object, **kwargs: object) -> tuple[float, float]:
        value = next_value()
        return value, 100.0 - value

    monkeypatch.setattr(cafo_module, "evaluate_cafo_predictor", fake_eval)
    config = {
        "experiment_name": "es-char",
        "data": {"num_classes": 2},
        "data_loader": {},
        "checkpointing": {},
        "algorithm_params": {
            "train_blocks": False,
            "num_epochs_per_block": MAX_EPOCHS,
            "log_interval": 9999,
            "predictor_early_stopping_enabled": True,
            "predictor_early_stopping_metric": "val_loss",
            "predictor_early_stopping_mode": mode,
            "predictor_early_stopping_patience": patience,
            "predictor_early_stopping_min_delta": delta,
        },
    }
    train_loader, val_loader = _loaders()
    train_cafo_model(
        model=CaFo_CNN(
            input_channels=1, block_channels=[2], image_size=2, num_classes=2
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=DEVICE,
    )
    return len(calls)


TRAINERS = {
    "ff": (run_ff, "max"),
    "mf_matrix_only": (run_mf_matrix_only, "min"),
    "cafo_predictor": (run_cafo_predictor, "min"),
}


@pytest.mark.parametrize("trainer_name", sorted(TRAINERS))
@pytest.mark.parametrize(
    ["stream_name", "patience"],
    [
        ("stagnant", 1),
        ("stagnant", 3),
        ("improving", 1),
        ("recovery_then_stagnation", 2),
        ("nan_gaps", 2),
        ("exact_tie_with_delta", 2),
    ],
)
def test_trainer_stop_point_matches_legacy_rule_and_documents_canonical_off_by_one(
    monkeypatch, trainer_name, stream_name, patience
):
    runner, mode = TRAINERS[trainer_name]
    assert_characterized(
        lambda: runner(monkeypatch, STREAMS[stream_name], patience, mode, 0.5),
        STREAMS[stream_name],
        patience,
        mode,
        0.5,
        # Flat tail => the ONLY possible difference is the documented boundary.
        canonical_exactly_one_later=(stream_name == "stagnant"),
    )


def test_inf_now_follows_canonical_semantics_after_d1_unification(monkeypatch):
    """After the D1 swap, +/-inf counts as a bad epoch everywhere.

    Deliberate protocol decision: the pre-unification inline rule treated
    +inf in max mode as an improvement (pinned by the step-1 characterization);
    finite/NaN streams remain byte-identical.
    """
    stream = [5.0, INF, 6.0, 5.0, 5.0, 5.0, 5.0]
    # Unified behaviour: canonical non-finite handling (inf => bad epoch)
    # combined with the preserved legacy "bad >= patience" boundary via
    # patience-1 => stop @6.  (Pre-swap inline stopped @5; canonical with the
    # full configured patience would stop @7.)
    epochs = run_ff(monkeypatch, stream, patience=3, mode="max", delta=0.0)
    assert epochs == 6
    assert epochs == canonical_stop_epoch(stream, 2, "max", 0.0)
    assert epochs != legacy_stop_epoch(stream, 3, "max", 0.0)
