#!/usr/bin/env python3
"""W0b: where the MF-Seq energy goes — per-stage energy share of the PPAM-era runs.

M0 is a dead stage: phase 2 consumes a_i for i >= 0 and evaluation reads a_L/M_L,
so no prediction ever reads M_0 (test_m0_dead_stage.py pins this). But it is not
free: on the canonical runs it is ~23% of all epochs. This script gives the
energy side of that fact by segmenting each run's own NVML trace at the stage
boundaries logged by the training process:

- M0 window:   "Starting ... Layer_M0" -> "Finished training for Layer_M0"
- W1_M1 window: "Starting ... Layer_W1_M1" -> "Starting ... Layer_W2_M2" (or its
  early-stopping line, whichever the log wrote)
- W2_M2 / W3_M3 windows: the same rule; W3_M3 ends at "Test Set Results"
- remainder: monitoring start -> M0 start plus evaluation after test start

The trace timestamp is relative to the monitoring start; the CSV filename carries
the wall-clock start, and log lines carry wall-clock timestamps, so each boundary
maps to a trace-relative second. Windows are half-open [start, end).

Assumes the canonical PPAM-era logs (complete blocks per seed, no reruns).
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.jpdc.analyze_g1 import BOOTSTRAP_N, BOOTSTRAP_SEED

EXPERIMENTS = ["mf_cifar10_mlp_3x2000", "mf_cifar100_mlp_3x2000"]
TS_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
FILENAME_TS_RE = re.compile(r"_(\d{8}_\d{6})\.csv$")
HEADER_RE = re.compile(r"Overriding config seed with EXPERIMENT_SEED=(\d+)")
BOUNDARY_PATTERNS = {
    "m0_start": r"Starting MF training for Layer_M0",
    "m0_end": r"Finished training for Layer_M0 after",
    "w1_start": r"Starting MF training for Layer_W1_M1",
    "w1_end": r"(Layer_W1_M1: Early Stopping at Epoch|Starting MF training for Layer_W2_M2)",
    "w2_start": r"Starting MF training for Layer_W2_M2",
    "w2_end": r"(Layer_W2_M2: Early Stopping at Epoch|Starting MF training for Layer_W3_M3)",
    "w3_start": r"Starting MF training for Layer_W3_M3",
    "w3_end": r"Test Set Results"
}
COMPILED = {k: re.compile(v) for k, v in BOUNDARY_PATTERNS.items()}


def block_boundaries(log_path: Path) -> dict[int, dict[str, float]]:
    """Returns seed -> {boundary: wall-clock epoch seconds} for complete blocks."""
    out: dict[int, dict[str, float]] = {}
    seed = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = HEADER_RE.search(line)
        if m:
            seed = int(m.group(1))
            out[seed] = {}
            continue
        if seed is None or not line[:4].isdigit():
            continue
        try:
            ts = datetime.strptime(line[:23], TS_FORMAT).timestamp()
        except ValueError:
            continue
        for name, pattern in COMPILED.items():
            if name not in out[seed] and pattern.search(line):
                out[seed][name] = ts
    return out


def trace_energy_wh(trace: pd.DataFrame, t0: float, t1: float) -> float:
    ts = pd.to_numeric(trace.timestamp_sec, errors="coerce")
    power = pd.to_numeric(trace.power_watts, errors="coerce")
    mask = (ts >= t0) & (ts < t1)
    if not mask.any():
        return float("nan")
    dt = ts.diff().fillna(0.0)
    return float((power[mask] * dt[mask]).sum()) / 3600.0


def trace_mean_power(trace: pd.DataFrame, t0: float, t1: float) -> float:
    ts = pd.to_numeric(trace.timestamp_sec, errors="coerce")
    mask = (ts >= t0) & (ts < t1)
    if not mask.any():
        return float("nan")
    return float(pd.to_numeric(trace.power_watts, errors="coerce")[mask].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tidy", type=Path, required=True)
    parser.add_argument("--run-logs-root", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    tidy = pd.read_csv(args.tidy, usecols=lambda c: c in {"experiment_name", "seed", "protocol", "metric", "value", "monitoring_csv_path"})
    tidy = tidy[(tidy.protocol == "harmonised") & tidy.experiment_name.isin(EXPERIMENTS)]

    rows = []
    for experiment in EXPERIMENTS:
        log_path = args.run_logs_root / experiment / f"{experiment}_run.log"
        boundaries = block_boundaries(log_path)
        traces = tidy[tidy.experiment_name == experiment]
        trace_paths = (
            traces[traces.metric == "total_gpu_energy_wh"]
            .drop_duplicates("seed")
            .set_index("seed")
            .monitoring_csv_path
        )
        total_energy = (
            traces[traces.metric == "total_gpu_energy_wh"].set_index("seed").value
        )
        for seed, bounds in sorted(boundaries.items()):
            trace_name = Path(trace_paths.get(seed, "")).name
            trace_path = args.traces / trace_name
            if not trace_name or not trace_path.exists():
                continue
            m = FILENAME_TS_RE.search(trace_name)
            if not m:
                continue
            anchor = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp()
            trace = pd.read_csv(trace_path)
            span = float(pd.to_numeric(trace.timestamp_sec, errors="coerce").max())

            def rel(name: str) -> float:
                value = bounds.get(name)
                return float("nan") if value is None else value - anchor

            windows = {
                "M0": (0.0, rel("m0_end")),
                "W1_M1": (rel("w1_start"), rel("w1_end")),
                "W2_M2": (rel("w2_start"), rel("w2_end")),
                "W3_M3": (rel("w3_start"), rel("w3_end")),
            }
            row = dict(experiment=experiment, seed=seed)
            accounted = 0.0
            for stage, (t0, t1) in windows.items():
                energy = trace_energy_wh(trace, t0, t1)
                row[f"{stage}_sec"] = t1 - t0
                row[f"{stage}_wh"] = energy
                row[f"{stage}_power"] = trace_mean_power(trace, t0, t1)
                accounted += 0.0 if np.isnan(energy) else energy
            row["trace_wh"] = trace_energy_wh(trace, 0.0, span)
            row["tidy_total_wh"] = float(total_energy.get(seed, float("nan")))
            row["other_wh"] = row["trace_wh"] - accounted
            for stage in windows:
                row[f"{stage}_share"] = row[f"{stage}_wh"] / row["trace_wh"]
            rows.append(row)

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "w0b_stage_energy.csv", index=False)

    lines = ["# W0b — MF-Seq per-stage energy share (PPAM-era canonical runs, seeds 42-61)", "",
             "M0 is dead compute (see `tests/test_m0_dead_stage.py`): it contributes no "
             "prediction, yet owns a measurable slice of passes, wall-clock and energy.", ""]
    for experiment in EXPERIMENTS:
        sub = df[df.experiment == experiment]
        lines.append(f"## {experiment}")
        lines.append("")
        lines.append("| stage | mean Wh | share of trace | mean power W | mean sec |")
        lines.append("|---|---|---|---|---|")
        for stage in ("M0", "W1_M1", "W2_M2", "W3_M3"):
            sub_stage = sub[sub[f"{stage}_wh"].notna()]
            if sub_stage.empty:
                continue
            lines.append(
                f"| {stage} | {sub_stage[f'{stage}_wh'].mean():.2f} "
                f"[{bootstrap_ci(sub_stage[f'{stage}_wh'])}] "
                f"| {sub_stage[f'{stage}_share'].mean() * 100:.1f}% "
                f"| {sub_stage[f'{stage}_power'].mean():.1f} "
                f"| {sub_stage[f'{stage}_sec'].mean():.0f} |"
            )
        lines.append("")
        lines.append(
            f"Trace total {sub.trace_wh.mean():.2f} Wh vs tidy total "
            f"{sub.tidy_total_wh.mean():.2f} Wh; unsegmented remainder "
            f"{sub.other_wh.mean():+.2f} Wh (monitoring warm-up before M0 start and "
            "post-test teardown)."
        )
        lines.append("")
    (args.out_dir / "w0b_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def bootstrap_ci(values: pd.Series) -> str:
    values = values.to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return "n/a"
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = values[rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))].mean(axis=1)
    return f"{np.percentile(boot, 2.5):.2f}, {np.percentile(boot, 97.5):.2f}"


if __name__ == "__main__":
    main()
