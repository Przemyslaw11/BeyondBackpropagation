#!/usr/bin/env python3
"""Parse results/*_run.log files into per-unit epoch statistics (Phase 0.3a).

Decomposes the MF/BP epoch multiplier into stage count x per-stage convergence
x early-stopping tail.

Log facts (verified on canonical runs):
- MF stages announce themselves via "Starting MF training for Layer_X";
  only M0 emits per-epoch "Epoch k/N - Val Local Loss" curves, later stages
  only emit the stage start and "Early Stopping at Epoch N" lines, or run to
  the epoch cap when early stopping never fires.
- BP/BP_DS/MF_Joint emit per-epoch "Epoch k/N | ... | Val Loss: v" lines and a
  generic "Early Stopping Triggered" message.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

SEED_RE = re.compile(r"Overriding config seed with EXPERIMENT_SEED=(\d+)")
MF_START_RE = re.compile(r"Starting MF training for Layer_([A-Za-z0-9_]+)")
MF_EPOCH_RE = re.compile(r"Layer_([A-Za-z0-9_]+) Epoch (\d+)/(\d+) - Val Local Loss: ([-+0-9.eE]+)")
JOINT_EPOCH_RE = re.compile(r"(\S+) Epoch (\d+)/(\d+) \| .*?Val Loss: ([-+0-9.eE]+)")
STOP_MF_RE = re.compile(r"Layer_([A-Za-z0-9_]+): Early Stopping at Epoch (\d+)")
STOP_GENERIC_RE = re.compile(r"Early Stopping Triggered")

ALGO_PREFIX = {"BP": "bp", "BP_DS": "bp_ds", "MF_Joint": "mf_joint", "MF_Simul": "mf_simul"}

CANONICAL_PAIRS = [
    ("cifar10", "bp_cifar10_mlp_3x2000", "mf_cifar10_mlp_3x2000"),
    ("cifar10_joint", "bp_cifar10_mlp_3x2000", "mf_joint_cifar10_mlp_3x2000"),
    ("cifar10_ds", "bp_cifar10_mlp_3x2000", "bp_ds_cifar10_mlp_3x2000"),
    ("cifar100", "bp_cifar100_mlp_3x2000", "mf_cifar100_mlp_3x2000"),
    ("cifar100_joint", "bp_cifar100_mlp_3x2000", "mf_joint_cifar100_mlp_3x2000"),
    ("cifar100_ds", "bp_cifar100_mlp_3x2000", "bp_ds_cifar100_mlp_3x2000"),
    ("mnist", "bp_baseline_mnist_mlp_2x1000", "mf_mnist_mlp_2x1000"),
    ("mnist_joint", "bp_baseline_mnist_mlp_2x1000", "mf_joint_mnist_mlp_2x1000"),
    ("fashion_mnist", "bp_baseline_fashion_mnist_mlp_2x1000", "mf_fashion_mnist_mlp_2x1000"),
    ("fashion_mnist_joint", "bp_baseline_fashion_mnist_mlp_2x1000", "mf_joint_fashion_mnist_mlp_2x1000"),
]


def file_epoch_cap(lines: list[str]) -> int | None:
    """Returns the unique epoch cap N across all 'Epoch k/N' lines, if unique."""
    caps = {int(m.group(3)) for m in (MF_EPOCH_RE.search(l) or JOINT_EPOCH_RE.search(l) for l in lines) if m}
    return caps.pop() if len(caps) == 1 else None


def parse_log(path: Path) -> list[dict]:
    """Returns one row per (seed, unit) with epoch statistics."""
    lines = path.read_text(errors="replace").splitlines()
    cap = file_epoch_cap(lines)
    rows: list[dict] = []
    seed = None
    units: dict[str, dict] = {}
    active: str | None = None

    def flush() -> None:
        if seed is None:
            return
        for key, u in units.items():
            if not u["vals"] and u["es_epoch"] is None and cap is None:
                continue
            if u["vals"]:
                best_idx = min(range(len(u["vals"])), key=lambda i: u["vals"][i])
                to_best, best_val, final_val = best_idx + 1, u["vals"][best_idx], u["vals"][-1]
            else:
                to_best = best_val = final_val = float("nan")
            epochs_run = u["es_epoch"] if u["es_epoch"] is not None else (len(u["vals"]) or cap)
            rows.append(
                dict(
                    log=path.name,
                    experiment=path.parent.name,
                    seed=seed,
                    algorithm=u["algorithm"],
                    unit=u["unit"],
                    epochs_run=epochs_run,
                    epochs_to_best=to_best,
                    best_val=best_val,
                    final_val=final_val,
                    stop_reason="es" if u["es_epoch"] is not None else "cap",
                    has_curve=bool(u["vals"]),
                )
            )

    def touch(algorithm: str, unit: str, key: str) -> dict:
        if key not in units:
            units[key] = dict(algorithm=algorithm, unit=unit, vals=[], es_epoch=None)
        return units[key]

    for raw in lines:
        line = raw.strip().replace("\r", "")
        m = SEED_RE.search(line)
        if m:
            flush()
            seed = int(m.group(1))
            units = {}
            active = None
            continue
        if seed is None:
            continue
        m = MF_START_RE.search(line)
        if m:
            active = f"mf:{m.group(1)}"
            touch("mf", m.group(1), active)
            continue
        m = MF_EPOCH_RE.search(line)
        if m:
            stage, epoch = m.group(1), int(m.group(2))
            try:
                val = float(m.group(4))
            except ValueError:
                # Rarely a log line is glued to its successor (truncated write
                # plus timestamp), producing e.g. "1.12932026-10-17". Skip the
                # corrupted value; epoch counts fall back to ES/cap logic.
                continue
            active = f"mf:{stage}"
            u = touch("mf", stage, active)
            if len(u["vals"]) < epoch:
                u["vals"].append(val)
            continue
        m = JOINT_EPOCH_RE.search(line)
        if m:
            algo = ALGO_PREFIX.get(m.group(1))
            if algo is None:
                continue
            epoch = int(m.group(2))
            try:
                val = float(m.group(4))
            except ValueError:
                continue
            active = algo
            u = touch(algo, algo, active)
            if len(u["vals"]) < epoch:
                u["vals"].append(val)
            continue
        m = STOP_MF_RE.search(line)
        if m:
            key = f"mf:{m.group(1)}"
            if key in units:
                units[key]["es_epoch"] = int(m.group(2))
            continue
        if STOP_GENERIC_RE.search(line) and active is not None and active in units:
            units[active]["es_epoch"] = len(units[active]["vals"])
    flush()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    all_rows: list[dict] = []
    for log in sorted(args.log_dir.glob("*/*_run.log")):
        all_rows.extend(parse_log(log))
    df = pd.DataFrame(all_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "epochs_by_unit.csv", index=False)
    print(f"units: {len(df)} rows, {df.experiment.nunique()} experiments, {df.seed.nunique()} seeds")

    per_run = (
        df.groupby(["experiment", "algorithm", "seed"])
        .agg(
            n_units=("unit", "nunique"),
            total_epochs=("epochs_run", "sum"),
            total_to_best=("epochs_to_best", "sum"),
            n_cap=("stop_reason", lambda s: (s == "cap").sum()),
        )
        .reset_index()
    )

    out_rows = []
    for label, bp_exp, arm_exp in CANONICAL_PAIRS:
        bp = per_run[per_run.experiment == bp_exp].set_index("seed")
        arm = per_run[per_run.experiment == arm_exp].set_index("seed")
        common = sorted(bp.index.intersection(arm.index))
        if not common:
            continue
        bp_c, arm_c = bp.loc[common], arm.loc[common]
        out_rows.append(
            dict(
                label=label,
                n_seeds=len(common),
                S=arm_c.n_units.mean(),
                bp_epochs=bp_c.total_epochs.mean(),
                arm_epochs=arm_c.total_epochs.mean(),
                multiplier=arm_c.total_epochs.mean() / bp_c.total_epochs.mean(),
                arm_capped=arm_c.n_cap.mean(),
            )
        )
    dec = pd.DataFrame(out_rows).round(3)
    dec.to_csv(args.out_dir / "multiplier_decomposition.csv", index=False)
    print(dec.to_string(index=False))

    bp = per_run[per_run.algorithm == "bp"].copy()
    bp["tail"] = bp.total_epochs - bp.total_to_best
    print("\nBP per-run summary (mean over seeds):")
    print(
        bp.groupby("experiment")
        .agg(epochs=("total_epochs", "mean"), to_best=("total_to_best", "mean"), tail=("tail", "mean"))
        .round(2)
        .to_string()
    )

    mf_stage = df[df.algorithm == "mf"]
    stage = (
        mf_stage.groupby(["experiment", "unit"])
        .agg(
            mean_run=("epochs_run", "mean"),
            mean_best=("epochs_to_best", "mean"),
            pct_es=("stop_reason", lambda s: 100 * (s == "es").mean()),
            pct_cap=("stop_reason", lambda s: 100 * (s == "cap").mean()),
            n=("unit", "size"),
        )
        .round(2)
    )
    print("\nMF per-stage summary (mean over seeds):")
    print(stage.to_string())


if __name__ == "__main__":
    main()
