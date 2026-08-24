from __future__ import annotations

import json
import sys

from beyond_backprop.contracts import MetricProvenance, MetricValue, RunStatus
from beyond_backprop.monitoring import (
    CodeCarbonResourceMonitor,
    NvmlResourceMonitor,
    WallClockResourceMonitor,
)
from beyond_backprop.tracking import LocalFileTracker


def test_local_tracker_persists_events_and_status(tmp_path):
    tracker = LocalFileTracker(tmp_path)
    tracker.log_config({"seed": 7})
    tracker.log_metrics({"loss": MetricValue(1.5, "", MetricProvenance("test", True))})
    tracker.log_artifact("summary.json")
    tracker.finish(RunStatus.SUCCEEDED)

    assert json.loads((tmp_path / "config.json").read_text()) == {"seed": 7}
    assert json.loads((tmp_path / "artifacts.json").read_text()) == ["summary.json"]
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "succeeded"
    assert '"loss"' in (tmp_path / "metrics.jsonl").read_text()


def test_wall_clock_monitor_is_dependency_free():
    monitor = WallClockResourceMonitor()
    monitor.start()
    snapshot = monitor.stop()

    assert snapshot.measured
    assert snapshot.source == "monotonic-clock"
    assert snapshot.duration_sec is not None
    assert snapshot.duration_sec >= 0


def test_optional_monitors_report_unavailable_dependencies(monkeypatch):
    nvml = NvmlResourceMonitor()
    nvml.start()
    nvml_snapshot = nvml.stop()
    assert nvml_snapshot.source == "nvml-unavailable"
    assert not nvml_snapshot.measured

    monkeypatch.setitem(sys.modules, "codecarbon", None)
    carbon = CodeCarbonResourceMonitor()
    carbon.start()
    carbon_snapshot = carbon.stop()
    assert carbon_snapshot.source == "codecarbon-unavailable"
    assert not carbon_snapshot.measured
