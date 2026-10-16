#!/usr/bin/env python3
"""Fit P = P_static + kappa * u per NVML trace and aggregate by algorithm (Phase 0.3b)."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

NAME_RE = re.compile(r"^(?P<experiment>.+?)_seed(?P<seed>\d+)_\d{8}_\d{6}\.csv$")


def algorithm_of(experiment: str) -> str:
    """Maps an experiment name to its algorithm family."""
    if experiment.startswith(("mf_simul",)):
        return "mf_simul"
    if experiment.startswith(("mf_joint",)):
        return "mf_joint"
    if experiment.startswith(("bp_ds",)):
        return "bp_ds"
    if experiment.startswith(("bp_",)):
        return "bp"
    if experiment.startswith(("mf_",)):
        return "mf"
    if experiment.startswith(("cafodfa",)):
        return "cafo_dfa"
    if experiment.startswith(("cafo",)):
        return "cafo"
    if experiment.startswith(("ff_",)):
        return "ff"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--monitor-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for path in sorted(args.monitor_dir.rglob("*.csv")):
        m = NAME_RE.match(path.name)
        if not m:
            continue
        try:
            df = pd.read_csv(path, usecols=["power_watts", "gpu_util_percent"]).dropna()
        except (ValueError, pd.errors.ParserError):
            continue
        if len(df) < 50:
            continue
        u = df["gpu_util_percent"].to_numpy(float)
        p = df["power_watts"].to_numpy(float)
        kappa, p_static = np.polyfit(u, p, 1)
        resid = p - (p_static + kappa * u)
        ss_res, ss_tot = float((resid**2).sum()), float(((p - p.mean()) ** 2).sum())
        rows.append(
            dict(
                experiment=m.group("experiment"),
                seed=int(m.group("seed")),
                algorithm=algorithm_of(m.group("experiment")),
                n_samples=len(df),
                p_static_w=round(float(p_static), 1),
                kappa_w_per_pct=round(float(kappa), 3),
                r2=round(1 - ss_res / ss_tot, 4) if ss_tot else float("nan"),
                mean_power_w=round(float(p.mean()), 1),
                mean_util=round(float(u.mean()), 1),
            )
        )

    trace = pd.DataFrame(rows)
    trace.to_csv(args.out, index=False)
    print(f"fitted {len(trace)} traces -> {args.out}")
    agg = (
        trace.groupby("algorithm")
        .agg(
            traces=("kappa_w_per_pct", "size"),
            kappa=("kappa_w_per_pct", "mean"),
            kappa_sd=("kappa_w_per_pct", "std"),
            p_static=("p_static_w", "mean"),
            r2=("r2", "mean"),
            mean_util=("mean_util", "mean"),
            mean_power=("mean_power_w", "mean"),
        )
        .round(3)
    )
    print(agg.to_string())


if __name__ == "__main__":
    main()
