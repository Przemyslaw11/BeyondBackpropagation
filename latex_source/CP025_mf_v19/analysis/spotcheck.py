#!/usr/bin/env python3
"""Independent check: recompute ten manuscript numbers from the raw run records.

This deliberately does not touch artifacts/tidy/runs.csv. It walks
results/**/*.json and the NVML per-run CSVs directly, so a defect in the tidy
build would show up here as a mismatch.

Run from a checkout that has results/ (it is not version controlled):
    python3 analysis/spotcheck.py [--results /path/to/results]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def load(results: Path) -> list:
    runs = []
    for p in sorted(results.rglob("*.json")):
        if p.name in {"ladder_analysis.json", "phase4_cafo_summary.json",
                      "phase4_ff_summary.json", "wandb_archive_meta_backup.json"}:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if isinstance(d, dict) and "experiment_name" in d and "test_accuracy" in d:
            d["_path"] = str(p.relative_to(results))
            runs.append(d)
    return runs


def sel(runs, exp, exclude_dirs=()):
    out = [r for r in runs if r["experiment_name"] == exp
           and not any(x in r["_path"] for x in exclude_dirs)]
    return out


def mean(runs, key):
    v = [r[key] for r in runs if r.get(key) is not None]
    return statistics.fmean(v), len(v)


def trace_mean_power(results: Path, rel_csv: str) -> float:
    """Trapezoidal energy over the trace, divided by its duration."""
    p = results.parent / rel_csv
    ts, pw = [], []
    with open(p) as fh:
        for row in csv.DictReader(fh):
            ts.append(float(row["timestamp_sec"]))
            pw.append(float(row["power_watts"]))
    joules = sum((ts[i + 1] - ts[i]) * (pw[i + 1] + pw[i]) / 2 for i in range(len(ts) - 1))
    return joules / (ts[-1] - ts[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--numbers", default=str(Path(__file__).with_name("numbers.json")))
    a = ap.parse_args()
    R = Path(a.results).resolve()
    runs = load(R)
    print(f"[load] {len(runs)} raw run records under {R}")
    n_runs_dir = len(list((R / "runs").rglob("*.json")))
    NUM = json.load(open(a.numbers))

    # The confounded and buggy archives, excluded exactly as the paper excludes them.
    DROP = ("equal_epochs_noval/", "reproduction_m0mem_bug/")
    clean = [r for r in runs if not any(d in r["_path"] for d in DROP)]

    checks = []

    def check(label, got, want, tol):
        ok = abs(got - want) <= tol
        checks.append(ok)
        print(f"  [{'OK ' if ok else 'FAIL'}] {label:58s} raw={got:12.5f}  paper={want:12.5f}")

    print("\n-- inventory --")
    check("results/runs/*.json count", n_runs_dir, 712, 0)
    check("clean raw run records", len(clean), 910, 0)
    check("superseded raw run records", len(runs) - len(clean), 86, 0)

    print("\n-- matched compute, CIFAR-10 3x2000 --")
    bp = sel(clean, "bp_cifar10_mlp_3x2000")
    mf = sel(clean, "mf_cifar10_mlp_3x2000_equal_epochs")
    mc = NUM["matched_compute"]["CIFAR-10 3x2000"]["metrics"]
    check("BP test accuracy (n=%d)" % len(bp), mean(bp, "test_accuracy")[0],
          mc["test_accuracy"]["bp"]["mean"], 1e-6)
    check("MF energy Wh (n=%d)" % len(mf), mean(mf, "total_gpu_energy_wh")[0],
          mc["total_gpu_energy_wh"]["mf"]["mean"], 1e-6)
    check("MF peak_torch_alloc_mib", mean(mf, "peak_torch_alloc_mib")[0],
          mc["peak_torch_alloc_mib"]["mf"]["mean"], 1e-6)
    check("BP peak_torch_alloc_mib", mean(bp, "peak_torch_alloc_mib")[0],
          mc["peak_torch_alloc_mib"]["bp"]["mean"], 1e-6)

    print("\n-- ladder, CIFAR-100 3x2000 --")
    lad = NUM["ladder"]["CIFAR-100 3x2000"]
    for exp, rung in [("bp_ds_cifar100_mlp_3x2000", "bp_ds"),
                      ("mf_cifar100_mlp_3x2000", "mf_recompute")]:
        s = sel(clean, exp)
        check(f"{rung} test accuracy (n={len(s)})", mean(s, "test_accuracy")[0],
              lad[rung]["test_accuracy"]["mean"], 1e-6)

    print("\n-- activation cache --")
    for ds, exp in [("MNIST 2x1000", "mf_mnist_mlp_2x1000_cache_device"),
                    ("Fashion-MNIST 2x1000", "mf_fashion_mnist_mlp_2x1000_cache_device")]:
        s = sel(clean, exp)
        check(f"{ds} cache_device peak_torch_alloc_mib", mean(s, "peak_torch_alloc_mib")[0],
              NUM["cache"][ds]["cache_device"]["peak_torch_alloc_mib"]["mean"], 1e-6)

    print("\n-- NVML trace, recomputed from the raw power CSV --")
    one = sel(clean, "bp_cifar100_mlp_3x2000")[0]
    got = trace_mean_power(R, one["monitoring_csv_path"])
    print(f"  [info] {one['_path']} -> {got:.4f} W "
          f"(arm mean in paper {NUM['matched_compute']['CIFAR-100 3x2000']['metrics']['trace_mean_power_w']['bp']['mean']:.4f} W)")
    lo, hi = 65.0, 80.0
    ok = lo < got < hi
    checks.append(ok)
    print(f"  [{'OK ' if ok else 'FAIL'}] single-run board power within [{lo}, {hi}] W")

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
