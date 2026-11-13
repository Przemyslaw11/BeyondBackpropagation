"""Guards the H-W1cap Step 1 trigger (PREREG-phase2.md §3).

The pilot verdict gates a full confirmatory arm, so the reducer must (a) compare
against the same-seed G1 MF-Seq baseline, (b) read a missing W1_M1 early-stopping
line as "trained to the 300 cap", and (c) apply the frozen OR-trigger exactly.
"""

import json
import sys
from pathlib import Path

import pytest

from scripts.jpdc import summarize_cap_pilot as mod

PILOT = "mf_cap300_cifar100_mlp_3x2000"
BASE = "mf_cifar100_mlp_3x2000"
SEEDS = [42, 43, 44, 45, 46]


def _write_run(results: Path, experiment: str, seed: int, acc: float, epochs: int) -> None:
    path = results / "runs" / experiment / f"{experiment}_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"test_accuracy": acc, "epochs_completed": epochs}))


def _write_out(outs: Path, seed: int, acc: float, w1_epochs: int | None, m0: int = 50) -> None:
    lines = [
        f"Config: configs/mf_cap300/cifar100_mlp_3x2000.yaml   Seed: {seed}",
        "2026-10-22 10:00:00,000 - INFO - Started",
        f"2026-10-22 10:00:01,000 - INFO - Finished training for Layer_M0 after {m0} epochs.",
    ]
    if w1_epochs is not None:
        lines.append(
            f"2026-10-22 10:10:00,000 - INFO - Layer_W1_M1: Early Stopping at Epoch {w1_epochs}"
        )
    lines += [
        "2026-10-22 10:20:00,000 - INFO - Layer_W2_M2: Early Stopping at Epoch 40",
        "2026-10-22 10:30:00,000 - INFO - Layer_W3_M3: Early Stopping at Epoch 21",
        f"2026-10-22 10:31:00,000 - INFO - Test Set Results: Acc: {acc:.2f}%, Loss: 2.0",
    ]
    outs.mkdir(parents=True, exist_ok=True)
    (outs / f"BBP_W1CapPilot-1_{seed}.out").write_text("\n".join(lines) + "\n")


@pytest.fixture()
def tree(tmp_path):
    results = tmp_path / "results"
    outs = tmp_path / "outs"
    for seed in SEEDS:
        _write_run(results, BASE, seed, 37.94, 191)
    return results, outs


def test_trigger_fires_when_stage_hits_cap(tree, tmp_path, monkeypatch):
    results, outs = tree
    for seed in SEEDS:
        _write_run(results, PILOT, seed, 38.60, 411)  # +0.66 pp
        _write_out(outs, seed, 38.60, w1_epochs=None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_cap_pilot", "--results-dir", str(results), "--outs-dir", str(outs), "--out-dir", str(tmp_path / "out")],
    )
    mod.main()
    text = (tmp_path / "out" / "cap_pilot_summary.md").read_text()
    assert "Trigger: **YES" in text
    assert "Mean W1_M1 epochs: **300.0**" in text


def test_no_trigger_below_both_thresholds(tree, tmp_path, monkeypatch):
    results, outs = tree
    for seed in SEEDS:
        _write_run(results, PILOT, seed, 38.00, 231)  # +0.06 pp
        _write_out(outs, seed, 38.00, w1_epochs=120)
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_cap_pilot", "--results-dir", str(results), "--outs-dir", str(outs), "--out-dir", str(tmp_path / "out")],
    )
    mod.main()
    text = (tmp_path / "out" / "cap_pilot_summary.md").read_text()
    assert "Trigger: **NO" in text
    assert "Mean W1_M1 epochs: **120.0**" in text
