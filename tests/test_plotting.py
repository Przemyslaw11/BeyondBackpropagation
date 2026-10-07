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
TIDY_CSV = REPO_ROOT / "results" / "tidy" / "runs.csv"
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


# --- The figures -------------------------------------------------------------


@pytest.fixture(scope="module")
def live(table: tidy.Table) -> tidy.Table:
    """What scripts/make_figures.py actually plots: nothing superseded."""
    return table.where(superseded="")


def test_figure_constants_match_the_preregistered_analysis():
    """A margin that drifts between the test and the plot is a silent lie."""
    from scripts import analyze_ablation_ladder as ladder
    from src.plotting import figures

    assert figures.EQUIVALENCE_MARGIN_PP == ladder.ACCURACY_EQUIVALENCE_MARGIN_PP
    assert figures.N_BOOTSTRAP == ladder.N_BOOTSTRAP
    assert figures.BOOTSTRAP_SEED == ladder.BOOTSTRAP_SEED


def test_excluding_superseded_runs_drops_exactly_the_two_known_protocols(
    table: tidy.Table, live: tidy.Table
):
    dropped = set(table.distinct("run_id")) - set(live.distinct("run_id"))
    sources = {run_id.split("/")[0] for run_id in dropped}
    assert sources == {"equal_epochs_noval", "reproduction_m0mem_bug"}


@pytest.mark.parametrize("summary_path", PHASE4_SUMMARIES, ids=lambda p: p.stem)
def test_figure_bootstrap_reproduces_the_preregistered_intervals(
    live: tidy.Table, summary_path: Path
):
    """The forest plot must not draw an interval the analysis did not compute."""
    from src.plotting import figures

    if not summary_path.is_file():
        pytest.skip(f"{summary_path} absent")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checked = 0
    for contrast in summary["contrasts"]:
        for row in contrast["rows"]:
            if row["n_pairs"] < 2:
                continue
            baseline = live.where(
                phase="phase4", experiment_name=contrast["baseline"]
            ).values(row["metric"])
            method = live.where(
                phase="phase4", experiment_name=contrast["method"]
            ).values(row["metric"])
            if not baseline or not method:
                continue
            estimate = (
                figures.paired_difference_ci
                if row["kind"] == "difference"
                else figures.paired_ratio_ci
            )
            point, low, high, n = estimate(baseline, method)
            assert n == row["n_pairs"]
            assert point == pytest.approx(row["point"], rel=1e-9, abs=1e-9)
            # The endpoints cannot match exactly: summarize_phase4.py advances one
            # shared Generator across every contrast, so its draw sequence depends
            # on call order. At n=5 the resampling grid is coarse, and this is the
            # resulting Monte-Carlo slack -- a real join or unit error would move
            # the point estimate, or the interval by many multiples of its width.
            span = max(abs(row["high"] - row["low"]), 1e-12)
            assert abs(low - row["low"]) < 0.15 * span
            assert abs(high - row["high"]) < 0.15 * span
            checked += 1
    assert checked >= 20


def test_every_forest_entry_carries_a_verdict_and_a_seed_count(live: tidy.Table):
    from src.plotting import figures

    entries = figures._forest_rows_ladder(live) + figures._forest_rows_phase4(live)
    assert len(entries) >= 25
    for _, _, point, low, high, n in entries:
        assert n >= 2
        assert low <= point <= high
        assert figures.equivalence_verdict(
            low, high, figures.EQUIVALENCE_MARGIN_PP
        ) in {"equivalent", "different", "inconclusive"}


def test_every_plotted_trace_is_nvml_and_not_a_wandb_system_panel(live: tidy.Table):
    """W&B's own system.* stream is what the submitted figures used. None of the
    trace columns here can come from it: they are read from the monitor's CSV."""
    hardware = {
        column: instrument
        for column, (_, instrument) in tidy.TRACE_COLUMNS.items()
        if column != "timestamp_sec"
    }
    assert hardware
    for column, instrument in hardware.items():
        lowered = instrument.lower()
        assert "nvml" in lowered or "psutil" in lowered, column
        assert "wandb" not in lowered and "system" not in lowered, column

    trace_rows = [r for r in live.rows if r["metric"].startswith("trace_")]
    assert trace_rows
    for row in trace_rows:
        assert row["monitoring_csv_path"].startswith("results/monitoring/")
        assert "wandb" not in row["instrument"].lower()


def test_figures_render_byte_identically(live: tidy.Table, tmp_path: Path):
    """The PDF backend stamps a creation date unless it is suppressed."""
    import hashlib

    from src.plotting import figures, style

    style.apply_style()
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    for build in (figures.ladder_waterfall, figures.equivalence_forest):
        a = build(live, first)
        b = build(live, second)
        assert hashlib.sha256(a.read_bytes()).hexdigest() == (
            hashlib.sha256(b.read_bytes()).hexdigest()
        ), f"{a.name} is not reproducible"
