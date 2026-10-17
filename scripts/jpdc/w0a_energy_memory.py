#!/usr/bin/env python3
"""W0a: decisive G1 energy / power / wall-clock / memory table.

Reads per arm x dataset x seed 42-61:
- run JSON:  <results>/runs/<exp>/<exp>_seed<s>.json
- run log:   <results>/<exp>/<exp>_run.log               (passes)
- NVML CSV:  the exact <monitoring_csv_path> stored in the run JSON, and nowhere
  else. A missing strict path is reported, never silently replaced: during the
  2026-10-17 quota window 12 decisive runs finished (JSON written) while their
  NVML CSV write failed, and the previous glob fallback then mixed September
  PPAM-era traces into the decisive energy table. The fallback still surfaces a
  `trace_from=glob` row, flagged for exclusion.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scripts.jpdc.analyze_g1 import ARMS, BOOTSTRAP_N, BOOTSTRAP_SEED, EXPERIMENTS, SEEDS, holm
from scripts.jpdc.parse_run_logs import parse_log


def passes_by_seed(results_dir: Path, experiment: str) -> dict[int, float]:
    """Unit epochs per seed from the latest run block (same rule as analyze_g1)."""
    log_path = results_dir / experiment / f"{experiment}_run.log"
    if not log_path.exists():
        return {}
    latest: dict[tuple[int, str, str], float] = {}
    for row in parse_log(log_path):
        latest[(row["seed"], row["algorithm"], row["unit"])] = float(row["epochs_run"])
    totals: dict[int, float] = {}
    for (seed, _, _), epochs in latest.items():
        totals[seed] = totals.get(seed, 0.0) + epochs
    return totals


def run_json(results_dir: Path, experiment: str, seed: int) -> dict | None:
    path = results_dir / "runs" / experiment / f"{experiment}_seed{seed}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def load_trace(results_dir: Path, record: dict, experiment: str, seed: int):
    """Returns (DataFrame | None, trace_from) using the JSON path strictly."""
    declared = record.get("monitoring_csv_path")
    if declared:
        # The declared path is the run's own trace; the basename is unique
        # (timestamped), so matching it inside results_dir keeps the exact file
        # even when the results tree has been copied to another machine.
        path = results_dir / "monitoring" / Path(declared).name
        if path.exists() and path.stat().st_size > 0:
            try:
                return pd.read_csv(path), "json"
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                pass
    fallback = sorted(
        p
        for p in (results_dir / "monitoring").glob(f"{experiment}_seed{seed}_*.csv")
        if p.stat().st_size > 0
    )
    if fallback:
        try:
            return pd.read_csv(fallback[-1]), "glob"
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass
    return None, "missing"


def trace_stats(trace: pd.DataFrame) -> dict:
    out = {}
    if trace is None or trace.empty or "power_watts" not in trace.columns:
        return {k: float("nan") for k in ("energy_wh", "mean_power_w", "span_sec", "mean_util", "max_temp")}
    power = pd.to_numeric(trace["power_watts"], errors="coerce")
    ts = pd.to_numeric(trace["timestamp_sec"], errors="coerce")
    dt = ts.diff().fillna(0.0)
    out["energy_wh"] = float((power * dt).sum()) / 3600.0
    out["mean_power_w"] = float(power.mean())
    out["span_sec"] = float(ts.max() - ts.min())
    out["mean_util"] = float(pd.to_numeric(trace.get("gpu_util_percent"), errors="coerce").mean())
    out["max_temp"] = float(pd.to_numeric(trace.get("gpu_temp_celsius"), errors="coerce").max())
    return out


def collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    for (arm, dataset), experiment in EXPERIMENTS.items():
        passes = passes_by_seed(results_dir, experiment)
        for seed in SEEDS:
            record = run_json(results_dir, experiment, seed)
            if record is None:
                continue
            trace, trace_from = load_trace(results_dir, record, experiment, seed)
            rows.append(
                dict(
                    arm=arm,
                    dataset=dataset,
                    seed=seed,
                    test_accuracy=record.get("test_accuracy"),
                    passes=passes.get(seed, float("nan")),
                    duration_sec=record.get("total_run_duration_sec"),
                    peak_torch_alloc_mib=record.get("peak_torch_alloc_mib"),
                    peak_gpu_mem_used_mib=record.get("peak_gpu_mem_used_mib"),
                    trace_from=trace_from,
                    **trace_stats(trace),
                )
            )
    return pd.DataFrame(rows)


def ci95(values: pd.Series) -> tuple[float, float]:
    values = values.to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)
    return float(np.mean(values) - half), float(np.mean(values) + half)


def paired(df: pd.DataFrame, arm: str, dataset: str, metric: str) -> dict:
    a = df[(df.arm == arm) & (df.dataset == dataset)].set_index("seed")[metric]
    b = df[(df.arm == "bp") & (df.dataset == dataset)].set_index("seed")[metric]
    common = sorted(a.index.intersection(b.index))
    diffs = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return dict(n=0, mean_diff=float("nan"), ci_low=float("nan"), ci_high=float("nan"), p=1.0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = diffs[rng.integers(0, len(diffs), size=(BOOTSTRAP_N, len(diffs)))].mean(axis=1)
    p = 1.0 if np.allclose(diffs, 0.0) else float(stats.wilcoxon(diffs, alternative="two-sided").pvalue)
    return dict(
        n=len(diffs),
        mean_diff=float(np.mean(diffs)),
        ci_low=float(np.percentile(boot, 2.5)),
        ci_high=float(np.percentile(boot, 97.5)),
        p=p,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    df = collect(args.results_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "w0a_per_seed.csv", index=False)

    lines = ["# W0a — decisive G1 cost table (seeds 42-61)", ""]
    lines.append("Energy = NVML trace integral over the run's own trace file (JSON path), "
                 "never a same-seed trace from another run attempt. "
                 "Power = mean of 5 Hz `power_watts` samples. "
                 "Wall-clock = `total_run_duration_sec`. "
                 "Alloc = `peak_torch_alloc_mib` (per-process, excludes CUDA context).")
    lines.append("")

    metrics = [
        ("test_accuracy", "test acc %"),
        ("passes", "passes"),
        ("energy_wh", "energy Wh"),
        ("mean_power_w", "Power W"),
        ("duration_sec", "wall s"),
        ("peak_torch_alloc_mib", "alloc MiB"),
    ]
    for dataset in ("cifar10", "cifar100"):
        lines.append(f"## Protocol A — {dataset}")
        lines.append("")
        header = "| arm | n | " + " | ".join(label for _, label in metrics) + " |"
        sep = "|---|" + "---|" * (len(metrics) + 1)
        lines += [header, sep]
        for arm in ARMS:
            sub = df[(df.arm == arm) & (df.dataset == dataset)]
            if sub.empty:
                continue
            cells = []
            for metric, _ in metrics:
                lo, hi = ci95(sub[metric].astype(float))
                cells.append(f"{sub[metric].astype(float).mean():.3g} [{lo:.3g}, {hi:.3g}]")
            lines.append(f"| {arm} | {len(sub)} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Trace provenance audit")
    lines.append("")
    audit = df.groupby(["dataset", "arm", "trace_from"]).size().unstack(fill_value=0)
    lines.append("```")
    lines.append(audit.to_string())
    lines.append("```")
    lines.append("")

    bad = df[df.trace_from != "json"]
    lines.append(f"Runs whose energy does NOT come from their own trace file: {len(bad)}")
    for _, row in bad.iterrows():
        lines.append(f"- {row.arm} {row.dataset} seed{row.seed}: trace_from={row.trace_from}")
    lines.append("")

    lines.append("## Paired deltas vs BP (same seeds)")
    lines.append("")
    lines.append("| dataset | metric | arm | n | mean diff | 95% CI | raw p |")
    lines.append("|---|---|---|---|---|---|---|")
    for dataset in ("cifar10", "cifar100"):
        for metric in ("energy_wh", "mean_power_w", "duration_sec", "passes", "peak_torch_alloc_mib"):
            pvals, rows = [], []
            for arm in ("mf_joint", "mf_simul", "mf_seq"):
                stat = paired(df, arm, dataset, metric)
                pvals.append(stat["p"])
                rows.append((arm, stat))
            adj = holm(pvals)
            for (arm, stat), p_holm in zip(rows, adj):
                lines.append(
                    f"| {dataset} | {metric} | {arm} | {stat['n']} | "
                    f"{stat['mean_diff']:+.4g} | [{stat['ci_low']:+.4g}, {stat['ci_high']:+.4g}] | "
                    f"{stat['p']:.4f} (Holm {p_holm:.4f}) |"
                )
    lines.append("")
    (args.out_dir / "w0a_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
