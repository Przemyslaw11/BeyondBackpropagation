#!/usr/bin/env python3
"""Reduce raw G1 results to every number used in PREREG.md §5.

Reads (per arm x dataset x seed 42-61):
- run JSON:        <results>/runs/<exp>/<exp>_seed<s>.json   (accuracy, durations)
- run log:         <results>/<exp>/<exp>_run.log             (unit epochs = passes)
- NVML CSV:        <results>/monitoring/<exp>_seed<s>_*.csv  (energy, Wh)

Writes <out-dir>/g1_per_seed.csv and <out-dir>/g1_summary.md, and applies the
frozen A/B/C decision rule to the CIFAR-100 comparison. Committed before the
decisive runs per PREREG.md §6; its hash is recorded in PREREG.md §10.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scripts.jpdc.parse_run_logs import parse_log

SEEDS = list(range(42, 62))
ARMS = ["bp", "mf_joint", "mf_simul", "mf_seq"]
EXPERIMENTS = {
    ("bp", "cifar10"): "bp_cifar10_mlp_3x2000",
    ("bp", "cifar100"): "bp_cifar100_mlp_3x2000",
    ("mf_joint", "cifar10"): "mf_joint_cifar10_mlp_3x2000",
    ("mf_joint", "cifar100"): "mf_joint_cifar100_mlp_3x2000",
    ("mf_simul", "cifar10"): "mf_simul_cifar10_mlp_3x2000",
    ("mf_simul", "cifar100"): "mf_simul_cifar100_mlp_3x2000",
    ("mf_seq", "cifar10"): "mf_cifar10_mlp_3x2000",
    ("mf_seq", "cifar100"): "mf_cifar100_mlp_3x2000",
}
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 0


def run_json(results_dir: Path, experiment: str, seed: int) -> dict | None:
    path = results_dir / "runs" / experiment / f"{experiment}_seed{seed}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def run_passes(results_dir: Path, experiment: str, seed: int) -> float:
    log_path = results_dir / experiment / f"{experiment}_run.log"
    if not log_path.exists():
        return float("nan")
    rows = [r for r in parse_log(log_path) if r["seed"] == seed]
    if not rows:
        return float("nan")
    return float(sum(r["epochs_run"] for r in rows))


def run_energy_wh(results_dir: Path, experiment: str, seed: int) -> float:
    paths = sorted((results_dir / "monitoring").glob(f"{experiment}_seed{seed}_*.csv"))
    if not paths:
        return float("nan")
    trace = pd.read_csv(paths[-1])
    dt = trace["timestamp_sec"].diff().fillna(0.0)
    joules = float((trace["power_watts"] * dt).sum())
    return joules / 3600.0


def collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    for (arm, dataset), experiment in EXPERIMENTS.items():
        for seed in SEEDS:
            record = run_json(results_dir, experiment, seed)
            if record is None:
                continue
            rows.append(
                dict(
                    arm=arm,
                    dataset=dataset,
                    seed=seed,
                    test_accuracy=record.get("test_accuracy"),
                    test_loss=record.get("test_loss"),
                    passes=run_passes(results_dir, experiment, seed),
                    energy_wh=run_energy_wh(results_dir, experiment, seed),
                    duration_sec=record.get("total_run_duration_sec"),
                    epochs_completed=record.get("epochs_completed"),
                )
            )
    return pd.DataFrame(rows)


def ci95(values: np.ndarray) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)
    return float(np.mean(values) - half), float(np.mean(values) + half)


def paired_stats(df: pd.DataFrame, arm: str, dataset: str, metric: str) -> dict:
    arm_values = (
        df[(df.arm == arm) & (df.dataset == dataset)].set_index("seed")[metric]
    )
    bp_values = df[(df.arm == "bp") & (df.dataset == dataset)].set_index("seed")[metric]
    common = sorted(arm_values.index.intersection(bp_values.index))
    a = arm_values.loc[common].to_numpy(dtype=float)
    b = bp_values.loc[common].to_numpy(dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    diffs = a - b
    if len(diffs) == 0:
        return dict(
            n=0,
            mean_arm=float("nan"),
            mean_bp=float("nan"),
            mean_diff=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            p_value=1.0,
        )
    if np.allclose(diffs, 0.0):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(diffs, alternative="two-sided").pvalue)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = diffs[rng.integers(0, len(diffs), size=(BOOTSTRAP_N, len(diffs)))].mean(axis=1)
    return dict(
        n=len(diffs),
        mean_arm=float(np.mean(a)),
        mean_bp=float(np.mean(b)),
        mean_diff=float(np.mean(diffs)),
        ci_low=float(np.percentile(boot, 2.5)),
        ci_high=float(np.percentile(boot, 97.5)),
        p_value=p_value,
    )


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, p_values[index] * (len(p_values) - rank))
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    df = collect(args.results_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "g1_per_seed.csv", index=False)

    summary_lines = ["# G1 summary (PREREG.md §5)", ""]
    summary_lines.append(
        f"Reduced {len(df)} runs from `{args.results_dir}`; "
        f"seeds {SEEDS[0]}-{SEEDS[-1]}."
    )
    summary_lines.append("")

    summary_lines.append("## Per-arm means (95% CI)")
    summary_lines.append("")
    summary_lines.append("| arm | dataset | n | test acc % | passes | energy Wh | duration s |")
    summary_lines.append("|---|---|---|---|---|---|---|")
    for dataset in ("cifar10", "cifar100"):
        for arm in ARMS:
            sub = df[(df.arm == arm) & (df.dataset == dataset)]
            if sub.empty:
                continue
            acc_ci = ci95(sub["test_accuracy"].to_numpy(dtype=float))
            summary_lines.append(
                f"| {arm} | {dataset} | {len(sub)} | "
                f"{sub['test_accuracy'].mean():.2f} [{acc_ci[0]:.2f}, {acc_ci[1]:.2f}] | "
                f"{sub['passes'].mean():.1f} | {sub['energy_wh'].mean():.2f} | "
                f"{sub['duration_sec'].mean():.0f} |"
            )
    summary_lines.append("")

    accuracy_stats = {}
    for dataset in ("cifar10", "cifar100"):
        accuracy_stats[dataset] = paired_stats(df, "mf_simul", dataset, "test_accuracy")
    p_raw = [accuracy_stats[d]["p_value"] for d in ("cifar10", "cifar100")]
    p_holm = holm(p_raw)
    accuracy_stats["cifar10"]["p_holm"] = p_holm[0]
    accuracy_stats["cifar100"]["p_holm"] = p_holm[1]

    summary_lines.append("## MF-Simul vs BP (paired by seed, Wilcoxon + bootstrap)")
    summary_lines.append("")
    summary_lines.append("| dataset | n | mean diff pp | 95% CI | p | p (Holm) | passes ratio R |")
    summary_lines.append("|---|---|---|---|---|---|---|")
    passes_ratio = {}
    for dataset in ("cifar10", "cifar100"):
        stat = accuracy_stats[dataset]
        passes = (
            df[(df.arm == "mf_simul") & (df.dataset == dataset)]["passes"].mean()
            / df[(df.arm == "bp") & (df.dataset == dataset)]["passes"].mean()
        )
        passes_ratio[dataset] = passes
        summary_lines.append(
            f"| {dataset} | {stat['n']} | {stat['mean_diff']:+.2f} | "
            f"[{stat['ci_low']:+.2f}, {stat['ci_high']:+.2f}] | "
            f"{stat['p_value']:.3f} | {stat['p_holm']:.3f} | {passes:.2f} |"
        )
    summary_lines.append("")

    mf_seq_stat = paired_stats(df, "mf_seq", "cifar100", "test_accuracy")
    summary_lines.append("## MF-Seq vs BP (CIFAR-100, drift-control arm)")
    summary_lines.append("")
    if mf_seq_stat.get("n"):
        summary_lines.append(
            f"mean diff {mf_seq_stat['mean_diff']:+.2f} pp "
            f"[{mf_seq_stat['ci_low']:+.2f}, {mf_seq_stat['ci_high']:+.2f}], "
            f"p = {mf_seq_stat['p_value']:.3f}, n = {mf_seq_stat['n']}"
        )
    else:
        summary_lines.append("no contemporaneous MF-Seq runs found (see PREREG §7 fallback)")
    summary_lines.append("")

    delta100 = accuracy_stats["cifar100"].get("mean_diff", float("nan"))
    r100 = passes_ratio.get("cifar100", float("nan"))
    mf_seq_gain = mf_seq_stat.get("mean_diff", float("nan"))
    if delta100 >= 2.0 and r100 <= 1.5:
        decision = "A (detachment survives)"
    elif delta100 <= 0.5 and mf_seq_gain >= 2.0:
        decision = "B (schedule artifact)"
    else:
        decision = "C (intermediate)"
    summary_lines.append("## Decision")
    summary_lines.append("")
    summary_lines.append(
        f"CIFAR-100: delta(MF-Simul - BP) = {delta100:+.2f} pp, R = {r100:.2f}, "
        f"MF-Seq gain = {mf_seq_gain:+.2f} pp -> **{decision}**"
    )
    summary_lines.append("")

    (args.out_dir / "g1_summary.md").write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
