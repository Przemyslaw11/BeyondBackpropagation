"""Tests for the R1 fixed legacy checkpoint saver in utils.training_support."""

from __future__ import annotations

import os

import pytest
import torch

from beyond_backprop.utils.training_support import save_checkpoint


def _state() -> dict:
    return {
        "epoch": 3,
        "best_metric_value": 91.5,
        "state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer": {"step": 7},
    }


def test_save_checkpoint_writes_legacy_payload_shapes(tmp_path) -> None:
    state = _state()
    save_checkpoint(
        state,
        is_best=True,
        filename="ff_checkpoint_epoch_3.pth",
        best_filename="ff_exp_best.pth",
        checkpoint_dir=str(tmp_path),
    )

    assert (tmp_path / "ff_checkpoint_epoch_3.pth").exists()
    assert (tmp_path / "ff_exp_best.pth").exists()
    # Main file keeps the full legacy dict; best file stays a raw state_dict.
    loaded = torch.load(tmp_path / "ff_checkpoint_epoch_3.pth", weights_only=False)
    assert set(loaded) == set(state)
    assert loaded["epoch"] == 3
    assert loaded["best_metric_value"] == 91.5
    assert torch.equal(loaded["state_dict"]["weight"], state["state_dict"]["weight"])
    best = torch.load(tmp_path / "ff_exp_best.pth", weights_only=False)
    assert torch.equal(best["weight"], state["state_dict"]["weight"])


def test_save_checkpoint_raises_instead_of_swallowing(tmp_path, monkeypatch) -> None:
    def broken_save(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint({"state_dict": {}}, is_best=False, checkpoint_dir=str(tmp_path))
    # No temp files left behind.
    assert list(tmp_path.iterdir()) == []


def test_failed_best_save_leaves_existing_checkpoint_intact(tmp_path, monkeypatch) -> None:
    good_state = _state()
    save_checkpoint(
        good_state,
        is_best=True,
        filename="checkpoint.pth",
        best_filename="model_best.pth",
        checkpoint_dir=str(tmp_path),
    )
    before = torch.load(tmp_path / "model_best.pth")

    calls = iter([None, OSError("disk full")])

    def flaky_save(*args: object, **kwargs: object) -> None:
        result = next(calls)
        if isinstance(result, OSError):
            raise result

    monkeypatch.setattr(torch, "save", flaky_save)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(
            {**good_state, "best_metric_value": 99.0},
            is_best=True,
            filename="checkpoint2.pth",
            best_filename="model_best.pth",
            checkpoint_dir=str(tmp_path),
        )

    monkeypatch.undo()
    # The previous best checkpoint was not corrupted or deleted, and no
    # temp files linger.
    after = torch.load(tmp_path / "model_best.pth", weights_only=False)
    assert set(after) == set(before)
    assert all(torch.equal(after[k], before[k]) for k in before)
    assert not [name for name in os.listdir(tmp_path) if name.startswith(".")]
