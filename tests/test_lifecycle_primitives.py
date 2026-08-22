from pathlib import Path

import pytest
import torch

from beyond_backprop.checkpointing import CheckpointError, CheckpointManager
from beyond_backprop.contracts import MetricProvenance, MetricValue, ResourceSnapshot
from beyond_backprop.monitoring import NoOpResourceMonitor
from beyond_backprop.training import EarlyStopping


def test_early_stopping_supports_min_max_and_serialization() -> None:
    stopping = EarlyStopping(patience=1, mode="min", min_delta=0.1)
    assert not stopping.update(1.0, epoch=1)
    assert not stopping.update(1.05, epoch=2)
    assert stopping.update(1.06, epoch=3)

    restored = EarlyStopping.from_state_dict(stopping.state_dict())
    assert restored.state_dict() == stopping.state_dict()

    maximizing = EarlyStopping(patience=0, mode="max")
    assert not maximizing.update(10.0, epoch=1)
    assert maximizing.update(9.0, epoch=2)


def test_checkpoint_manager_saves_atomic_versioned_payload(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "checkpoints")
    state = {"weight": torch.tensor([1.0, 2.0])}
    path = manager.save(
        "model.pth",
        model_state=state,
        optimizer_state={"step": 3},
        epoch=4,
        algorithm="mf",
        best_metric_name="val_loss",
        best_metric_value=0.25,
        config_hash="abc123",
    )

    assert path.exists()
    assert not list(path.parent.glob(".model.pth.*"))
    payload = manager.load("model.pth")
    assert torch.equal(payload["state_dict"]["weight"], state["weight"])
    assert payload["optimizer_state_dict"] == {"step": 3}
    assert payload["metadata"]["format_version"] == 1
    assert payload["metadata"]["config_hash"] == "abc123"


def test_checkpoint_manager_reports_missing_and_corrupt_checkpoints(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path)
    with pytest.raises(CheckpointError, match="not found"):
        manager.load("missing.pth")

    (tmp_path / "broken.pth").write_bytes(b"not a torch checkpoint")
    with pytest.raises(CheckpointError, match="Could not load"):
        manager.load("broken.pth")


def test_metric_and_resource_snapshots_preserve_units_and_provenance() -> None:
    metric = MetricValue(3.17, "Wh", MetricProvenance(source="nvml", measured=True))
    assert metric.to_dict() == {
        "value": 3.17,
        "unit": "Wh",
        "source": "nvml",
        "measured": True,
    }

    snapshot = ResourceSnapshot(
        duration_sec=2.5, energy_wh=3.17, measured=True, source="nvml"
    )
    metrics = snapshot.to_metrics()
    assert metrics["duration_sec"].unit == "s"
    assert metrics["energy_wh"].provenance.measured


def test_noop_monitor_is_explicitly_disabled() -> None:
    snapshot = NoOpResourceMonitor().stop()
    assert snapshot.measured is False
    assert snapshot.source == "disabled"
