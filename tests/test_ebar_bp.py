"""Tests for the Ebar_BP reducer.

The fixture contains only the three fields the script is allowed to read
(experiment_name, seed, epochs_completed), plus a deliberately absurd accuracy
that must not influence the result.  W1 gating is tested with a path containing
``results`` and with a record carrying ``secondary_eval_seconds``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "jpdc" / "ebar_bp.py"


def _write(
    root: Path,
    dataset: str,
    epochs: dict[int, float],
    *,
    marker: bool = False,
) -> Path:
    experiment = f"bp_{dataset}_mlp_3x2000"
    directory = root / "runs" / experiment
    directory.mkdir(parents=True, exist_ok=True)
    for seed, value in epochs.items():
        record = {
            "experiment_name": experiment,
            "seed": seed,
            "epochs_completed": value,
            "test_accuracy": 999.0,
        }
        if marker:
            record["secondary_eval_seconds"] = 2.9
        (directory / f"{experiment}_seed{seed}.json").write_text(json.dumps(record))
    return root


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def test_ebar_and_budgets_from_epochs_only(tmp_path):
    epochs = {seed: float(50 + index) for index, seed in enumerate(range(42, 62))}
    root = _write(tmp_path / "fixtures", "cifar10", epochs)
    out = tmp_path / "ebar.json"
    result = _run(
        "--results-dir",
        str(root),
        "--datasets",
        "cifar10",
        "--out",
        str(out),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    row = payload["datasets"]["cifar10"]
    assert row["n"] == 20
    assert row["ebar_bp"] == pytest.approx(59.5)
    assert row["budget_s4"] == 15
    assert row["budget_s3"] == 20


def test_missing_seed_is_rejected(tmp_path):
    epochs = {seed: float(50 + index) for index, seed in enumerate(range(42, 62))}
    del epochs[61]
    root = _write(tmp_path / "fixtures", "cifar10", epochs)
    result = _run("--results-dir", str(root), "--datasets", "cifar10")
    assert result.returncode != 0
    assert "missing" in (result.stderr + result.stdout).lower()


def test_results_path_requires_w1_ack(tmp_path):
    epochs = {seed: 55.0 for seed in range(42, 62)}
    root = _write(tmp_path / "results", "cifar10", epochs)
    result = _run("--results-dir", str(root), "--datasets", "cifar10")
    assert result.returncode != 0
    assert "w1-ack" in result.stderr


def test_w1_marker_requires_w1_ack(tmp_path):
    epochs = {seed: 55.0 for seed in range(42, 62)}
    root = _write(tmp_path / "fixtures", "cifar10", epochs, marker=True)
    blocked = _run("--results-dir", str(root), "--datasets", "cifar10")
    assert blocked.returncode != 0
    assert "secondary_eval_seconds" in blocked.stderr
    allowed = _run(
        "--results-dir", str(root), "--datasets", "cifar10", "--w1-ack"
    )
    assert allowed.returncode == 0, allowed.stderr
