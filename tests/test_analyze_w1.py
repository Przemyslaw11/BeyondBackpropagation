"""Synthetic fixtures for the frozen W1 reducer (PREREG-phase2.md §3/§4).

The reducer is a decision instrument, so these tests exercise the failure modes
that would silently change a conclusion: missing seeds, missing traces, and a
deliberately planted 2x2 effect.  The fixture writes real per-run JSON files and
real per-run NVML CSV paths; it never reads the shared append-only log.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.jpdc import analyze_w1 as mod

SEEDS = list(range(42, 62))
MAIN = ["bp", "mf_joint", "mf_simul", "mf_seq", "mf_global", "mf_nom0"]


def _values(dataset: str) -> dict[str, float]:
    if dataset == "cifar100":
        return {
            "bp": 40.0,
            "mf_joint": 43.0,
            "mf_simul": 42.0,
            "mf_seq": 48.0,
            "mf_global": 42.5,
            "mf_nom0": 48.5,
            "bp_fixed": 47.8,
            "bp_bestval": 40.2,
        }
    return {
        "bp": 62.0,
        "mf_joint": 64.0,
        "mf_simul": 63.0,
        "mf_seq": 66.0,
        "mf_global": 63.5,
        "mf_nom0": 66.2,
        "bp_fixed": 65.8,
        "bp_bestval": 62.1,
    }


def _write_trace(results: Path, name: str, energy_wh: float) -> str:
    """Write a two-interval trace whose integrated energy is energy_wh."""
    monitoring = results / "monitoring"
    monitoring.mkdir(parents=True, exist_ok=True)
    power = energy_wh * 1800.0  # 2 seconds at this power -> energy_wh
    path = monitoring / name
    path.write_text(
        "timestamp_sec,power_watts\n"
        f"0.0,{power}\n"
        f"1.0,{power}\n"
        f"2.0,{power}\n"
    )
    return f"monitoring/{name}"


def _write_run(
    results: Path,
    arm: str,
    dataset: str,
    seed: int,
    accuracy: float,
    *,
    passes: float = 100.0,
    energy_wh: float = 10.0,
    duration: float = 100.0,
    secondary: dict | None = None,
    trace: bool = True,
) -> None:
    experiment = mod.EXPERIMENTS[arm].format(dataset=dataset)
    record = {
        "experiment_name": experiment,
        "arm": arm,
        "dataset": dataset,
        "seed": seed,
        "test_accuracy": accuracy,
        "test_loss": 1.0,
        "epochs_completed": passes,
        "training_duration_sec": duration,
        "total_run_duration_sec": duration + 5.0,
        "monitoring_csv_path": _write_trace(
            results, f"{experiment}_seed{seed}.csv", energy_wh
        )
        if trace
        else None,
    }
    if secondary:
        record.update(secondary)
    path = results / "runs" / experiment / f"{experiment}_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def _build_tree(root: Path, *, planted: bool = True) -> Path:
    results = root / "results"
    rng = np.random.default_rng(12345)
    common_noise = rng.normal(0.0, 0.05, len(SEEDS))
    for dataset in ("cifar10", "cifar100"):
        values = _values(dataset)
        for seed_index, seed in enumerate(SEEDS):
            for arm in MAIN + ["bp_fixed", "bp_bestval"]:
                if arm == "mf_seq":
                    passes, energy = 200.0, 20.0
                elif arm == "mf_nom0":
                    passes, energy = 120.0, 12.0
                elif arm == "mf_global":
                    passes, energy = 210.0, 21.0
                elif arm == "bp_fixed":
                    passes, energy = 200.0, 20.0
                else:
                    passes, energy = 100.0, 10.0
                _write_run(
                    results,
                    arm,
                    dataset,
                    seed,
                    values[arm] + common_noise[seed_index],
                    passes=passes,
                    energy_wh=energy,
                    duration=100.0 + seed_index,
                )
    # A deliberately planted 2x2 effect on C100:
    # J=43, S=42, Q=48, G=42.5 -> A=+2.75, P=-2.25, interaction=-6.5.
    # H-W1a: G-Q=-5.5 <= -2 and G-S=+0.5 <= +1.
    # H-W1c: nom0-seq=+0.5, pass/energy saving=40%.
    # H-W1b: G-Q=-5.5 < -1, so threshold-exclusive with H-W1a.
    # H-W1d: bp_fixed-seq=-0.2 at a 1.0 pass ratio.
    assert planted
    return results


def _hypothesis(path: Path, dataset: str = "cifar100", name: str = "H-W1a") -> pd.Series:
    frame = pd.read_csv(path / "w1_hypotheses.csv")
    rows = frame[(frame.dataset == dataset) & (frame.hypothesis == name)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_planted_2x2_effect_is_recovered(tmp_path):
    results = _build_tree(tmp_path)
    out = tmp_path / "out"
    report = mod.analyze(results, out)

    assert _hypothesis(out, name="H-W1a")["status"] == "supported"
    assert _hypothesis(out, name="H-W1b")["status"].startswith("not supported")
    assert _hypothesis(out, name="H-W1c")["status"] == "supported"
    assert "retired" in _hypothesis(out, name="H-W1d")["status"]

    effects = pd.read_csv(out / "w1_2x2.csv")
    c100_acc = effects[
        (effects.dataset == "cifar100") & (effects.metric == "test_accuracy")
    ].set_index("effect")
    assert c100_acc.loc["loss_attachment", "mean"] == pytest.approx(2.75)
    assert c100_acc.loc["prefix_gradients", "mean"] == pytest.approx(-2.25)
    assert c100_acc.loc["interaction", "mean"] == pytest.approx(-6.5)
    assert "2x2 attribution table" in report["summary"]
    assert "H-W1a" in report["summary"]


def test_missing_seed_is_reported_and_blocks_support(tmp_path):
    results = _build_tree(tmp_path)
    experiment = mod.EXPERIMENTS["mf_seq"].format(dataset="cifar100")
    missing_path = (
        results / "runs" / experiment / f"{experiment}_seed42.json"
    )
    missing_path.unlink()

    out = tmp_path / "out"
    mod.analyze(results, out)
    per_seed = pd.read_csv(out / "w1_per_seed.csv")
    seq_c100 = per_seed[
        (per_seed.arm == "mf_seq") & (per_seed.dataset == "cifar100")
    ]
    assert len(seq_c100) == 19
    assert 42 not in set(seq_c100.seed)

    missing = pd.read_csv(out / "w1_missing_seeds.csv")
    row = missing[
        (missing.arm == "mf_seq") & (missing.dataset == "cifar100")
    ].iloc[0]
    assert row["missing_seeds"] == "42"
    assert row["n"] == 19
    assert "missing [42]" in (out / "w1_summary.md").read_text()
    assert _hypothesis(out, name="H-W1a")["status"].startswith("incomplete")


def test_missing_trace_is_nan_and_strictly_json_declared(tmp_path):
    results = tmp_path / "results"
    _write_run(
        results,
        "mf_seq",
        "cifar100",
        42,
        48.0,
        passes=200.0,
        energy_wh=20.0,
        trace=False,
    )
    frame = mod.collect(results)
    row = frame.iloc[0]
    assert pd.isna(row["energy_wh"])
    assert row["energy_source"] == "missing"


def test_secondary_tables_read_probe_and_stage_fields(tmp_path):
    results = tmp_path / "results"
    secondary = {
        "final_train_loss": 0.5,
        "final_train_accuracy": 80.0,
        "final_val_loss": 0.6,
        "final_val_accuracy": 70.0,
        "final_test_loss": 0.7,
        "final_test_accuracy": 60.0,
        "secondary_eval_seconds": 1.25,
        "linear_probe_test_accuracy": {"a_1": 50.0, "a_2": 55.0},
        "effective_rank": {"a_1": 3.0, "a_2": 4.0},
        "local_head_test_accuracy": {"M_0": 40.0, "M_1": 45.0},
        "stage_epochs": [
            {
                "stage": "M0",
                "epochs": 10,
                "stopping_epoch": 10,
                "max_epochs": 100,
                "stopped_early": True,
            }
        ],
    }
    _write_run(
        results,
        "mf_seq",
        "cifar100",
        42,
        48.0,
        passes=200.0,
        energy_wh=20.0,
        secondary=secondary,
    )
    out = tmp_path / "out"
    report = mod.analyze(results, out)
    summary = report["secondary"]
    assert not summary["probe"].empty
    assert set(summary["probe"].layer) == {"a_1", "a_2"}
    assert set(summary["head"]["head"]) == {"M_0", "M_1"}
    assert summary["stage"].iloc[0]["n"] == 1
    text = (out / "w1_summary.md").read_text()
    assert "Linear probes and effective rank" in text
    assert "Stage epochs and stopping" in text
    assert "1.250" in text


def test_holm_handles_missing_p_without_inventing_a_test():
    adjusted = mod.holm([0.01, float("nan"), 0.02])
    assert adjusted[0] == pytest.approx(0.02)
    assert pd.isna(adjusted[1])
    assert adjusted[2] == pytest.approx(0.02)
