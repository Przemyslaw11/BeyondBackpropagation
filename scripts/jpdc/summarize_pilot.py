#!/usr/bin/env python3
"""Exploratory one-pager for the G1 pilot (PREREG.md §9): n=5, CIFAR-10.

Pilot numbers are exploratory and never enter decisive tables. Reads the same
raw artifacts as analyze_g1.py (run JSONs, run logs, NVML CSVs) for seeds 42-46
and prints a markdown summary: per-arm mean±sd test accuracy, epochs, wall
duration, unit-epoch passes, and measured GPU energy.

Usage:
    python3 -m scripts.jpdc.summarize_pilot --results-dir <results> [--out FILE]
"""

import argparse
from pathlib import Path

import numpy as np

from scripts.jpdc.analyze_g1 import run_energy_wh, run_json, run_passes

ARMS = {
    "bp": "bp_cifar10_mlp_3x2000",
    "mf_joint": "mf_joint_cifar10_mlp_3x2000",
    "mf_simul": "mf_simul_cifar10_mlp_3x2000",
    "mf_seq": "mf_cifar10_mlp_3x2000",
}
SEEDS = list(range(42, 47))


def collect(results_dir: Path, seeds: list[int]) -> dict[str, dict[str, np.ndarray]]:
    data: dict[str, dict[str, np.ndarray]] = {}
    for arm, experiment in ARMS.items():
        acc, loss, epochs, duration, passes, wh = [], [], [], [], [], []
        for seed in seeds:
            record = run_json(results_dir, experiment, seed)
            if record is None:
                continue
            acc.append(record.get("test_accuracy", np.nan))
            loss.append(record.get("test_loss", np.nan))
            epochs.append(record.get("epochs_completed", np.nan))
            duration.append(record.get("total_run_duration_sec", np.nan))
            passes.append(run_passes(results_dir, experiment, seed))
            wh.append(run_energy_wh(results_dir, experiment, seed))
        data[arm] = dict(
            test_accuracy=np.asarray(acc, dtype=float),
            test_loss=np.asarray(loss, dtype=float),
            epochs=np.asarray(epochs, dtype=float),
            duration_sec=np.asarray(duration, dtype=float),
            passes=np.asarray(passes, dtype=float),
            energy_wh=np.asarray(wh, dtype=float),
        )
    return data


def fmt(values: np.ndarray, digits: int = 2) -> str:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def summary_md(data: dict[str, dict[str, np.ndarray]]) -> str:
    lines = [
        "# G1 pilot summary (exploratory, CIFAR-10, seeds 42-46)",
        "",
        "| arm | n | test acc (%) | test loss | epochs | wall (h) | passes (unit epochs) | energy (Wh) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, values in data.items():
        n = len(values["test_accuracy"])
        lines.append(
            f"| {arm} | {n} | {fmt(values['test_accuracy'])} | {fmt(values['test_loss'], 3)} "
            f"| {fmt(values['epochs'], 1)} | {fmt(values['duration_sec'] / 3600.0)} "
            f"| {fmt(values['passes'], 1)} | {fmt(values['energy_wh'])} |"
        )

    lines.append("")
    bp_passes = np.mean(data["bp"]["passes"]) if len(data["bp"]["passes"]) else np.nan
    for arm in ("mf_joint", "mf_simul", "mf_seq"):
        mean_passes = np.mean(data[arm]["passes"]) if len(data[arm]["passes"]) else np.nan
        if np.isnan(bp_passes) or np.isnan(mean_passes) or bp_passes == 0:
            continue
        lines.append(f"- passes ratio {arm}/bp: {mean_passes / bp_passes:.2f}")

    bp_acc = np.mean(data["bp"]["test_accuracy"]) if len(data["bp"]["test_accuracy"]) else np.nan
    ms_acc = (
        np.mean(data["mf_simul"]["test_accuracy"])
        if len(data["mf_simul"]["test_accuracy"])
        else np.nan
    )
    if not (np.isnan(bp_acc) or np.isnan(ms_acc)):
        lines.append(f"- MF-Simul - BP test accuracy: {ms_acc - bp_acc:+.2f} pp (CIFAR-10, exploratory)")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = collect(args.results_dir, args.seeds)
    report = summary_md(data)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
