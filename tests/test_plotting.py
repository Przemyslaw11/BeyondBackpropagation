"""The tidy table is the contract: every figure and paper table reads from it.

These tests are the guard against a silent join error. A wrong unit or a wrong
key would still render a plausible-looking figure, so the aggregate is checked
against two independently produced artefacts: the pre-registered Phase 3
analysis and the Phase 4 bootstrap summaries.
"""

from __future__ import annotations

import ast
import json
import statistics
from pathlib import Path

import pytest

from scripts.analyze_ablation_ladder import RUNGS, _rung_key
from src.plotting import tidy

REPO_ROOT = Path(__file__).resolve().parents[1]
TIDY_CSV = REPO_ROOT / "artifacts" / "tidy" / "runs.csv"
LADDER_ANALYSIS = REPO_ROOT / "results" / "ladder_analysis.json"
PHASE4_SUMMARIES = (
    REPO_ROOT / "results" / "phase4_ff_summary.json",
    REPO_ROOT / "results" / "phase4_cafo_summary.json",
)

DATASET_LABELS = {
    "mnist": "MNIST",
    "fashionmnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
}


@pytest.fixture(scope="module")
def table() -> tidy.Table:
    if not TIDY_CSV.is_file():
        pytest.skip(f"{TIDY_CSV} absent; run scripts/build_tidy_table.py")
    return tidy.Table.load(TIDY_CSV)


def test_rung_mapping_matches_the_preregistered_analysis():
    """tidy.rung_of must never drift from the analysis script that owns it."""
    assert [key for key, _, _ in RUNGS] == list(tidy.RUNG_ORDER)
    cases = [
        ("BP", "recompute"),
        ("BP_DS", "recompute"),
        ("MF_JOINT", "recompute"),
        ("MF", "recompute"),
        ("MF", "cache_device"),
        ("MF", "cache_host"),
        ("FF", "recompute"),
        ("CaFo", "recompute"),
    ]
    for algorithm, cache in cases:
        record = {"algorithm": algorithm, "activation_cache": cache}
        assert tidy.rung_of(algorithm, cache) == (_rung_key(record) or "")


def test_every_metric_carries_a_unit_and_an_instrument(table: tidy.Table):
    for row in table.rows:
        assert row["unit"], row
        assert row["instrument"], row


def test_memory_metrics_are_not_confusable(table: tidy.Table):
    """The submitted paper put a host-RSS trace under an NVML memory caption."""
    instruments = {
        row["metric"]: row["instrument"]
        for row in table.rows
        if any(token in row["metric"] for token in ("mem", "rss", "alloc"))
    }
    assert "NVML" in instruments["peak_gpu_mem_used_mib"]
    assert "torch" in instruments["peak_torch_alloc_mib"]
    assert "psutil" in instruments["peak_process_rss_mib"]
    assert "psutil" in instruments["trace_peak_process_rss_mib"]


def test_run_ids_are_unique_per_source(table: tidy.Table):
    """equal_epochs and equal_epochs_noval share names; they must not merge."""
    pairs = {(row["run_id"], row["source_file"]) for row in table.rows}
    by_id: dict[str, set[str]] = {}
    for run_id, source in pairs:
        by_id.setdefault(run_id, set()).add(source)
    collisions = {k: v for k, v in by_id.items() if len(v) > 1}
    assert not collisions, collisions


def _config_group(configuration: str) -> str:
    dataset, architecture = configuration.split("_", 1)
    dims = ast.literal_eval(architecture)
    return f"{DATASET_LABELS[dataset]} {len(dims)}x{dims[0]}"


@pytest.mark.skipif(not LADDER_ANALYSIS.is_file(), reason="Phase 3 analysis absent")
def test_ladder_means_reproduce_the_preregistered_analysis(table: tidy.Table):
    """Every mean in results/ladder_analysis.json must fall out of the table."""
    entries = json.loads(LADDER_ANALYSIS.read_text(encoding="utf-8"))
    checked = 0
    for entry in entries:
        group = _config_group(entry["configuration"])
        ladder = table.where(phase="phase3_ladder", config_group=group)
        for contrast in entry["contrasts"]:
            metric = contrast["metric"]
            left = ladder.where(rung=contrast["rung_a"]).values(metric)
            right = ladder.where(rung=contrast["rung_b"]).values(metric)
            shared = sorted(set(left) & set(right))
            assert len(shared) == contrast["n_pairs"], (group, contrast)
            for side, values in (("a", left), ("b", right)):
                got = statistics.fmean(values[seed] for seed in shared)
                assert got == pytest.approx(contrast[f"mean_{side}"], rel=1e-9), (
                    group,
                    metric,
                    contrast[f"rung_{side}"],
                )
                checked += 1
    assert checked >= 200


@pytest.mark.parametrize("summary_path", PHASE4_SUMMARIES, ids=lambda p: p.name)
def test_phase4_group_statistics_reproduce(table: tidy.Table, summary_path: Path):
    """The Phase 4 bootstrap summaries were produced by a different code path."""
    if not summary_path.is_file():
        pytest.skip(f"{summary_path} absent")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    phase4 = table.where(phase="phase4")
    checked = 0
    for experiment, group in summary["groups"].items():
        rows = phase4.where(experiment_name=_phase4_experiment(experiment))
        for metric, stats in group["metrics"].items():
            values = rows.values(metric)
            selected = [values[seed] for seed in group["seeds"] if seed in values]
            assert len(selected) == stats["n"], (experiment, metric)
            assert statistics.fmean(selected) == pytest.approx(
                stats["mean"], rel=1e-9
            ), (experiment, metric)
            checked += 1
    assert checked >= 40


def _phase4_experiment(name: str) -> str:
    """The summaries shorten two FF experiment names."""
    return {"ff_hinton_mnist_mlp_3x1000_SGD": "ff_hinton_mnist_mlp_3x1000_SGD"}.get(
        name, name
    )


def test_trace_energy_agrees_with_the_recorded_energy(table: tidy.Table):
    """Re-integrating the NVML CSV must recover the energy in the summary.

    If it does not, the tidy table is reading a different trace from the one the
    run actually produced.
    """
    by_run: dict[str, dict[str, float]] = {}
    for row in table.rows:
        by_run.setdefault(row["run_id"], {})[row["metric"]] = float(row["value"])
    compared = 0
    for run_id, metrics in by_run.items():
        power = metrics.get("trace_mean_power_w")
        duration = metrics.get("trace_duration_sec")
        recorded = metrics.get("total_gpu_energy_joules")
        if power is None or duration is None or not recorded:
            continue
        assert power * duration == pytest.approx(recorded, rel=0.02), run_id
        compared += 1
    assert compared > 900
