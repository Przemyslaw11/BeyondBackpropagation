#!/usr/bin/env python3
"""W0e: drift audit — G1 (2026-10-17) vs PPAM-era (2026-09-10/11) canonical runs.

The G1 reruns show higher mean power/energy on the same arms than the PPAM-era
camera-ready runs even though the training code and configs are identical
(`src/` unchanged from the decisive commit `08118ca` back to July; verified with
git diff). Whatever moved is machine state, time of day, or co-tenancy, and the
PPAM money plot must be read with that in mind. This script quantifies, per arm
x dataset x seed (42-61):

- energy (Wh), mean power (W), passes, wall-clock (s), accuracy
- machine state: mean GPU temp, mean utilization, mean SM clock
- co-tenancy: fraction of NVML samples with compute_processes > 1 and the max
  process count (two array tasks landing on the same physical GPU), read from
  the raw traces of both eras

Sources:
- PPAM era: tidy/runs_camera_ready.csv (protocol=harmonised, the six
  bp/mf/mf_joint cifar10/cifar100 experiments) + tidy/epochs_by_unit.csv for
  passes + raw traces in monitoring/monitoring/ for co-tenancy.
- G1: pull/results run JSONs (passes = epochs_completed) and each run's own
  strict `monitoring_csv_path` trace.

mf_simul has no PPAM-era counterpart (new baseline), so it appears in the G1
table and in the power/temperature comparison but not in the paired drift.

Additive log decomposition of the energy drift, per seed:
    ΔlnE = Δln(passes) + Δln(mean power) + Δln(sec per pass)
attributed to "schedule" (passes), "power", and "throughput" components.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.jpdc.analyze_g1 import BOOTSTRAP_N, BOOTSTRAP_SEED, EXPERIMENTS

PPAM_EXPERIMENTS = {
    ("bp", "cifar10"): "bp_cifar10_mlp_3x2000",
    ("bp", "cifar100"): "bp_cifar100_mlp_3x2000",
    ("mf_seq", "cifar10"): "mf_cifar10_mlp_3x2000",
    ("mf_seq", "cifar100"): "mf_cifar100_mlp_3x2000",
    ("mf_joint", "cifar10"): "mf_joint_cifar10_mlp_3x2000",
    ("mf_joint", "cifar100"): "mf_joint_cifar100_mlp_3x2000",
}
PPAM_METRICS = {
    "test_accuracy": "accuracy",
    "total_gpu_energy_wh": "energy_wh",
    "trace_mean_power_w": "mean_power_w",
    "total_run_duration_sec": "duration_sec",
    "trace_mean_gpu_temp_celsius": "mean_temp",
    "trace_mean_gpu_util_percent": "mean_util",
    "trace_mean_sm_clock_mhz": "mean_sm_clock",
}
SEEDS = list(range(42, 62))


def ppam_runs(tidy_csv: Path, epochs_csv: Path, trace_root: Path) -> pd.DataFrame:
    df = pd.read_csv(tidy_csv, usecols=lambda c: c in set(PPAM_METRICS) | {"experiment_name", "seed", "protocol", "metric", "value", "monitoring_csv_path"})
    df = df[(df.protocol == "harmonised") & df.experiment_name.isin(PPAM_EXPERIMENTS.values())]
    pivot = df[df.metric.isin(PPAM_METRICS)].pivot_table(
        index=["experiment_name", "seed"], columns="metric", values="value", aggfunc="first"
    )
    pivot = pivot.rename(columns=PPAM_METRICS).reset_index()

    epochs = pd.read_csv(epochs_csv)
    epochs = epochs[epochs.experiment.isin(PPAM_EXPERIMENTS.values())]
    passes = epochs.groupby(["experiment", "seed"]).epochs_run.sum().rename("passes").reset_index()
    pivot = pivot.merge(passes, left_on=["experiment_name", "seed"], right_on=["experiment", "seed"], how="left")

    cotn = []
    for _, row in df[df.metric == "total_gpu_energy_wh"].iterrows():
        trace_path = trace_root / Path(row.monitoring_csv_path).name
        stats = trace_extras(trace_path)
        cotn.append(dict(experiment_name=row.experiment_name, seed=row.seed, **stats))
    pivot = pivot.merge(pd.DataFrame(cotn), on=["experiment_name", "seed"], how="left")

    reverse = {v: k for k, v in PPAM_EXPERIMENTS.items()}
    pivot["arm"] = pivot.experiment_name.map(lambda e: reverse[e][0])
    pivot["dataset"] = pivot.experiment_name.map(lambda e: reverse[e][1])
    pivot["era"] = "ppam"
    return pivot[["era", "arm", "dataset", "seed", "accuracy", "passes", "energy_wh", "mean_power_w", "duration_sec", "mean_temp", "mean_util", "mean_sm_clock", "frac_cotn", "max_procs"]]


def trace_extras(path: Path) -> dict:
    out = {k: float("nan") for k in ("frac_cotn", "max_procs")}
    if not path.exists() or path.stat().st_size == 0:
        return out
    try:
        trace = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return out
    if "compute_processes" in trace.columns:
        procs = pd.to_numeric(trace.compute_processes, errors="coerce")
        out["frac_cotn"] = float((procs > 1).mean())
        out["max_procs"] = float(procs.max())
    return out


def g1_runs(results_dir: Path) -> pd.DataFrame:
    rows = []
    for (arm, dataset), experiment in EXPERIMENTS.items():
        for seed in SEEDS:
            jpath = results_dir / "runs" / experiment / f"{experiment}_seed{seed}.json"
            if not jpath.exists():
                continue
            record = json.loads(jpath.read_text())
            row = dict(era="g1", arm=arm, dataset=dataset, seed=seed, accuracy=record.get("test_accuracy"),
                       passes=record.get("epochs_completed"), energy_wh=float("nan"),
                       mean_power_w=float("nan"), duration_sec=record.get("total_run_duration_sec"),
                       mean_temp=float("nan"), mean_util=float("nan"), mean_sm_clock=float("nan"),
                       frac_cotn=float("nan"), max_procs=float("nan"))
            declared = record.get("monitoring_csv_path")
            path = results_dir / "monitoring" / Path(declared).name if declared else None
            if path is not None and path.exists() and path.stat().st_size > 0:
                try:
                    trace = pd.read_csv(path)
                    power = pd.to_numeric(trace.power_watts, errors="coerce")
                    ts = pd.to_numeric(trace.timestamp_sec, errors="coerce")
                    row["energy_wh"] = float((power * ts.diff().fillna(0.0)).sum()) / 3600.0
                    row["mean_power_w"] = float(power.mean())
                    row["mean_temp"] = float(pd.to_numeric(trace.gpu_temp_celsius, errors="coerce").mean())
                    row["mean_util"] = float(pd.to_numeric(trace.gpu_util_percent, errors="coerce").mean())
                    row["mean_sm_clock"] = float(pd.to_numeric(trace.sm_clock_mhz, errors="coerce").mean())
                    if "compute_processes" in trace.columns:
                        procs = pd.to_numeric(trace.compute_processes, errors="coerce")
                        row["frac_cotn"] = float((procs > 1).mean())
                        row["max_procs"] = float(procs.max())
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                    pass
            rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = []
    for (arm, dataset), group in df.groupby(["arm", "dataset"]):
        wide = group.pivot_table(index="seed", columns="era", values=metric)
        if not {"g1", "ppam"}.issubset(wide.columns):
            continue
        wide = wide.dropna()
        diffs = (wide["g1"] - wide["ppam"]).to_numpy(dtype=float)
        if len(diffs) == 0:
            continue
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boot = diffs[rng.integers(0, len(diffs), size=(BOOTSTRAP_N, len(diffs)))].mean(axis=1)
        out.append(dict(arm=arm, dataset=dataset, metric=metric, n=len(diffs),
                        mean_diff=float(np.mean(diffs)),
                        ci_low=float(np.percentile(boot, 2.5)),
                        ci_high=float(np.percentile(boot, 97.5))))
    return pd.DataFrame(out)


def decomposition(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    needed = [(m, era) for m in ("energy_wh", "mean_power_w", "passes", "duration_sec") for era in ("g1", "ppam")]
    for (arm, dataset), group in df.groupby(["arm", "dataset"]):
        wide = group.pivot_table(index="seed", columns="era", values=["energy_wh", "mean_power_w", "passes", "duration_sec"])
        if not set(needed).issubset(wide.columns):
            continue
        wide = wide.dropna(subset=needed)
        if wide.empty:
            continue
        d_ln_passes = np.log(wide[("passes", "g1")] / wide[("passes", "ppam")])
        d_ln_power = np.log(wide[("mean_power_w", "g1")] / wide[("mean_power_w", "ppam")])
        sec_per_pass = wide[("duration_sec", "g1")] / wide[("passes", "g1")] / (wide[("duration_sec", "ppam")] / wide[("passes", "ppam")])
        d_ln_spp = np.log(sec_per_pass)
        d_ln_e = np.log(wide[("energy_wh", "g1")] / wide[("energy_wh", "ppam")])
        out.append(dict(arm=arm, dataset=dataset, n=len(wide),
                        d_ln_energy=float(d_ln_e.mean()),
                        passes_component=float(d_ln_passes.mean()),
                        power_component=float(d_ln_power.mean()),
                        throughput_component=float(d_ln_spp.mean()),
                        residual=float((d_ln_e - d_ln_passes - d_ln_power - d_ln_spp).mean())))
    return pd.DataFrame(out)


def fmt_ci(mean: float, lo: float, hi: float, scale: float = 1.0) -> str:
    return f"{mean * scale:+.3g} [{lo * scale:+.3g}, {hi * scale:+.3g}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tidy", type=Path, required=True)
    parser.add_argument("--epochs-by-unit", type=Path, required=True)
    parser.add_argument("--ppam-traces", type=Path, required=True)
    parser.add_argument("--g1-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    ppam = ppam_runs(args.tidy, args.epochs_by_unit, args.ppam_traces)
    g1 = g1_runs(args.g1_results)
    both = pd.concat([ppam, g1], ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    both.to_csv(args.out_dir / "w0e_per_seed.csv", index=False)

    lines = ["# W0e — hardware/hypervisor drift: G1 (2026-10-17) vs PPAM era (2026-09-10/11)", "",
             "Code and configs are identical between eras (`src/` unchanged from `08118ca` back to July); "
             "any delta is machine state, date/time-of-day, or co-tenancy.", ""]

    lines.append("## Regime means")
    lines.append("")
    metrics = [("accuracy", "acc %"), ("passes", "passes"), ("energy_wh", "energy Wh"),
               ("mean_power_w", "power W"), ("duration_sec", "wall s"), ("mean_temp", "temp C"),
               ("mean_util", "util %"), ("frac_cotn", "frac co-tenant"), ("max_procs", "max procs")]
    for arm in ["bp", "mf_joint", "mf_simul", "mf_seq"]:
        for dataset in ["cifar10", "cifar100"]:
            sub = both[(both.arm == arm) & (both.dataset == dataset)]
            if sub.empty or "ppam" not in sub.era.values:
                continue
            lines.append(f"### {arm} / {dataset}")
            lines.append("")
            lines.append("| era | n | " + " | ".join(label for _, label in metrics) + " |")
            lines.append("|---|" + "---|" * (len(metrics) + 1))
            for era, era_label in [("ppam", "PPAM"), ("g1", "G1")]:
                esub = sub[sub.era == era]
                if esub.empty:
                    continue
                cells = [f"{esub[m].mean():.3g}" for m, _ in metrics]
                lines.append(f"| {era_label} | {len(esub)} | " + " | ".join(cells) + " |")
            lines.append("")

    lines.append("## Paired deltas (G1 − PPAM, same seed)")
    lines.append("")
    for metric, label in [("energy_wh", "energy Wh"), ("mean_power_w", "power W"), ("passes", "passes"),
                          ("duration_sec", "wall s"), ("mean_temp", "temp C"), ("mean_util", "util %"),
                          ("frac_cotn", "frac co-tenant")]:
        deltas = paired_deltas(both, metric)
        if deltas.empty:
            continue
        lines.append(f"**{label}**")
        lines.append("")
        lines.append("| arm | dataset | n | mean diff | 95% CI |")
        lines.append("|---|---|---|---|---|")
        for _, row in deltas.iterrows():
            lines.append(f"| {row.arm} | {row.dataset} | {int(row.n)} | {row.mean_diff:+.3g} | "
                         f"[{row.ci_low:+.3g}, {row.ci_high:+.3g}] |")
        lines.append("")

    lines.append("## Log decomposition of the energy drift")
    lines.append("")
    lines.append("Δln E = Δln passes + Δln power + Δln sec-per-pass (+ residual)")
    lines.append("")
    decomp = decomposition(both)
    lines.append("| arm | dataset | n | Δln E | passes | power | sec/pass | residual |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in decomp.iterrows():
        lines.append(f"| {row.arm} | {row.dataset} | {int(row.n)} | {row.d_ln_energy:+.3f} | "
                     f"{row.passes_component:+.3f} | {row.power_component:+.3f} | "
                     f"{row.throughput_component:+.3f} | {row.residual:+.3f} |")
    lines.append("")

    (args.out_dir / "w0e_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
