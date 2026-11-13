#!/usr/bin/env python3
"""G2 analysis (PREREG-G2-DRAFT.md section 4). Hash this file before any confirmatory G2 run.

Inputs
- ``--sweep-dir``: g2_sweep.py JSONs, GPU-resident pipeline, one per (dataset, repeat).
- ``--bridge-dir``: g2_sweep.py JSONs, host pipeline (batch 128, width 2000).
- ``--results-dir``: run tree with ``runs/g2_bp_mfinit_<ds>_mlp_3x2000`` (E-B) and the
  W1 core BP runs ``runs/bp_<ds>_mlp_3x2000`` it is paired with by seed.

Endpoints
- Per (cell, arm, repeat): net energy per sample = (counter J - idle W x window s) / samples.
  The per-pass advantage is ``a = e_bp / e_arm`` (a > 1: the MF arm is cheaper per pass).
- H-G2a: per repeat, OLS slope of ``log a`` (MF-Simul) on ``log(batch x width)`` over the
  12 cells; one-sided one-sample t-test, slope > 0.
- H-G2b: ``log a`` (MF-Simul) at the saturated corner; one-sided t-test, > 0.
- H-G2c: host-bridge ``log a`` per MF arm; descriptive (sign reported, no test).
- H-G2d: E-B minus W1 core BP test accuracy, paired by seed; TOST (paired t) at +-margin.
- Holm over {H-G2a, H-G2b, H-G2d} on the primary dataset; the other dataset repeats the
  tests with raw p-values as a consistency check and is never labelled confirmatory.
- Nothing is imputed. Missing repeats, cells or seeds make the row ``incomplete``.
- MF-Seq-nom0 stages are reported per stage, descriptively; their W1-weighted per-pass
  energy is a derived quantity computed elsewhere.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from scripts.jpdc.analyze_w1 import ALPHA, SEEDS, _as_float, _bootstrap_ci, holm

# Proposed constants (PREREG-G2-DRAFT.md section 4); the author fixes them at the freeze.
DATASETS = ("cifar10", "cifar100")
PRIMARY_DATASET = "cifar100"
BATCHES = (128, 512, 2048, 8192)
WIDTHS = (2000, 4000, 8000)
CORNER = (8192, 8000)
BRIDGE_CELL = (128, 2000)
N_REPEATS = 10
MF_ARMS = ("mf_simul", "mf_stage1", "mf_stage2", "mf_stage3")
TOST_MARGIN_PP = 0.5
FAMILY = ("H-G2a", "H-G2b", "H-G2d")


# ---------------------------------------------------------------- loading
def load_sweep(directory: Optional[Path], pipeline: str) -> pd.DataFrame:
    """One row per (dataset, repeat, batch, width, arm) unit, with net energy per sample."""
    rows: List[Dict] = []
    seen = set()
    for path in sorted(directory.glob("*.json")) if directory else []:
        rec = json.loads(path.read_text())
        args = rec["args"]
        if args.get("pipeline") != pipeline:
            raise ValueError(f"{path}: pipeline {args.get('pipeline')!r}, expected {pipeline!r}")
        key = (args["dataset"], int(args["repeat"]))
        if key in seen:
            raise ValueError(f"duplicate (dataset, repeat) {key}: {path}")
        seen.add(key)
        idle_w = _as_float(rec.get("idle_start", {}).get("idle_watts"))
        for u in rec["units"]:
            gross = _as_float(u.get("counter_energy_j"))
            net = gross - idle_w * _as_float(u.get("seconds"))
            rows.append({
                "dataset": key[0], "repeat": key[1], "batch": int(u["batch"]), "width": int(u["width"]),
                "arm": u["arm"], "samples": int(u["samples"]), "seconds": _as_float(u["seconds"]),
                "gross_j": gross, "net_j": net, "trace_j": _as_float(u.get("trace_energy_j")),
                "idle_w": idle_w, "gpu_uuid": rec.get("identity", {}).get("gpu_uuid"), "source": path.name,
                # A non-positive net energy is a measurement failure, never a value.
                "net_j_per_sample": net / int(u["samples"]) if net > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def advantage(units: pd.DataFrame) -> pd.DataFrame:
    """log a = log(e_bp / e_arm) per (dataset, repeat, cell, MF arm)."""
    if units.empty:
        return pd.DataFrame(columns=["dataset", "repeat", "batch", "width", "arm", "log_a"])
    keys = ["dataset", "repeat", "batch", "width"]
    bp = units[units.arm == "bp"].set_index(keys)["net_j_per_sample"]
    mf = units[units.arm.isin(MF_ARMS)].set_index(keys)
    out = mf[["arm"]].copy()
    out["log_a"] = np.log(bp.reindex(mf.index).to_numpy() / mf["net_j_per_sample"].to_numpy())
    return out.reset_index()


# ---------------------------------------------------------------- tests
def _one_sided_t(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    return float(stats.ttest_1samp(values, 0.0, alternative="greater").pvalue)


def _summary(name: str, dataset: str, values: np.ndarray, p: float, complete: bool, note: str) -> Dict:
    lo, hi = _bootstrap_ci(values)
    return {"hypothesis": name, "dataset": dataset, "n": int(np.isfinite(values).sum()),
            "estimate": float(np.nanmean(values)) if len(values) else float("nan"),
            "ci_low": lo, "ci_high": hi, "p_raw": p if complete else float("nan"),
            "complete": complete, "note": note}


def h_g2a(adv: pd.DataFrame, dataset: str) -> Dict:
    sub = adv[(adv.dataset == dataset) & (adv.arm == "mf_simul")]
    slopes, complete = [], True
    for r in range(N_REPEATS):
        cells = sub[sub.repeat == r].dropna(subset=["log_a"])
        if len(cells) != len(BATCHES) * len(WIDTHS):
            complete = False
            continue
        x = np.log(cells.batch.to_numpy(float) * cells.width.to_numpy(float))
        slopes.append(np.polyfit(x, cells.log_a.to_numpy(), 1)[0])
    v = np.array(slopes)
    return _summary("H-G2a", dataset, v, _one_sided_t(v), complete,
                    "slope of log a (MF-Simul) on log(batch x width), per repeat")


def h_g2b(adv: pd.DataFrame, dataset: str) -> Dict:
    sub = adv[(adv.dataset == dataset) & (adv.arm == "mf_simul")
              & (adv.batch == CORNER[0]) & (adv.width == CORNER[1])].dropna(subset=["log_a"])
    v = sub.sort_values("repeat").log_a.to_numpy()
    return _summary("H-G2b", dataset, v, _one_sided_t(v), len(v) == N_REPEATS,
                    f"log a (MF-Simul) at batch {CORNER[0]}, width {CORNER[1]}")


def h_g2c(bridge_adv: pd.DataFrame, dataset: str) -> List[Dict]:
    rows = []
    for arm in MF_ARMS:
        v = bridge_adv[(bridge_adv.dataset == dataset) & (bridge_adv.arm == arm)
                       & (bridge_adv.batch == BRIDGE_CELL[0])
                       & (bridge_adv.width == BRIDGE_CELL[1])].log_a.dropna().to_numpy()
        row = _summary("H-G2c", dataset, v, float("nan"), len(v) == N_REPEATS,
                       f"{arm}: host-bridge log a, descriptive")
        row["p_raw"] = float("nan")
        row["sign"] = "MF cheaper" if row["estimate"] > 0 else ("BP cheaper" if row["estimate"] < 0 else "n/a")
        rows.append(row)
    return rows


def load_init_pairs(results_dir: Optional[Path], dataset: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        acc = {}
        for arm, exp in (("bp_mfinit", f"g2_bp_mfinit_{dataset}_mlp_3x2000"), ("bp", f"bp_{dataset}_mlp_3x2000")):
            path = results_dir / "runs" / exp / f"{exp}_seed{seed}.json" if results_dir else None
            acc[arm] = _as_float(json.loads(path.read_text()).get("test_accuracy")) if path and path.exists() else float("nan")
        rows.append({"dataset": dataset, "seed": seed, **acc, "diff_pp": acc["bp_mfinit"] - acc["bp"]})
    return pd.DataFrame(rows)


def h_g2d(pairs: pd.DataFrame, dataset: str) -> Dict:
    d = pairs.diff_pp.to_numpy(float)
    complete = bool(np.isfinite(d).all()) and len(d) == len(SEEDS)
    p = float("nan")
    if np.isfinite(d).sum() >= 2:
        f = d[np.isfinite(d)]
        p_low = stats.ttest_1samp(f, -TOST_MARGIN_PP, alternative="greater").pvalue
        p_high = stats.ttest_1samp(f, TOST_MARGIN_PP, alternative="less").pvalue
        p = float(max(p_low, p_high))
    return _summary("H-G2d", dataset, d, p, complete,
                    f"BP-MFinit minus BP test accuracy (pp); TOST +-{TOST_MARGIN_PP}")


def hypotheses(adv: pd.DataFrame, bridge_adv: pd.DataFrame, results_dir: Optional[Path]):
    rows, pairs = [], []
    for ds in DATASETS:
        pr = load_init_pairs(results_dir, ds)
        pairs.append(pr)
        rows += [h_g2a(adv, ds), h_g2b(adv, ds), h_g2d(pr, ds)] + h_g2c(bridge_adv, ds)
    table = pd.DataFrame(rows)
    table["p_holm"] = float("nan")
    fam = table[(table.dataset == PRIMARY_DATASET) & table.hypothesis.isin(FAMILY)]
    if fam.complete.all():  # Holm only over a complete family; otherwise every verdict is incomplete
        table.loc[fam.index, "p_holm"] = holm(fam.p_raw.tolist())

    def verdict(row) -> str:
        if row.hypothesis == "H-G2c":
            return "descriptive"
        if not row.complete:
            return "incomplete"
        if row.dataset != PRIMARY_DATASET:
            return "consistent" if row.p_raw < ALPHA else "not consistent"
        return "supported" if row.p_holm < ALPHA else "not supported"

    table["verdict"] = table.apply(verdict, axis=1)
    return table, pd.concat(pairs, ignore_index=True)


def _markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    rows = [" | ".join(fmt(v) for v in r) for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + [f"| {r} |" for r in rows])


# ---------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sweep-dir", type=Path)
    p.add_argument("--bridge-dir", type=Path)
    p.add_argument("--results-dir", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    units = load_sweep(a.sweep_dir, "gpu")
    bridge = load_sweep(a.bridge_dir, "host")
    adv, bridge_adv = advantage(units), advantage(bridge)
    table, pairs = hypotheses(adv, bridge_adv, a.results_dir)

    units.to_csv(a.out_dir / "g2_units.csv", index=False)
    bridge.to_csv(a.out_dir / "g2_bridge_units.csv", index=False)
    adv.to_csv(a.out_dir / "g2_log_advantage.csv", index=False)
    pairs.to_csv(a.out_dir / "g2_init_per_seed.csv", index=False)
    table.to_csv(a.out_dir / "g2_hypotheses.csv", index=False)
    cell_means = (adv.groupby(["dataset", "arm", "batch", "width"]).log_a.mean().apply(math.exp)
                  .rename("a_geomean").reset_index()) if not adv.empty else pd.DataFrame()
    cell_means.to_csv(a.out_dir / "g2_advantage_by_cell.csv", index=False)
    bad = int(units.net_j_per_sample.isna().sum()) if not units.empty else 0
    lines = ["# G2 summary", "", f"Units: {len(units)} sweep, {len(bridge)} bridge; "
             f"non-positive or missing net energy: {bad}.", "", _markdown(table)]
    (a.out_dir / "g2_summary.md").write_text("\n".join(lines) + "\n")
    return table


if __name__ == "__main__":
    main()
