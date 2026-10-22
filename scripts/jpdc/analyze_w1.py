#!/usr/bin/env python3
"""W1 analysis: the attribution ladder (PREREG-phase2.md §3/§4).

Reads per arm x dataset x seed 42-61 from the run artifacts only:
- run JSON:  <results>/runs/<exp>/<exp>_seed<s>.json
- NVML CSV:  the exact ``monitoring_csv_path`` stored in that JSON

Arms (6 main): bp, mf_joint, mf_simul, mf_seq, mf_global (Seq-Global),
mf_nom0.
Controls (descriptive, not in the primary Holm family): bp_fixed,
bp_bestval, bp_sched, mf_seq_protocol_b, mf_nom0_protocol_b, and the
conditional mf_seq_cap300.

The analysis is deliberately strict about the pre-registration:

* Missing seeds are listed, never imputed.  A confirmatory verdict requires
  the full paired seed set 42-61; with fewer seeds the row is reported as
  incomplete and is not labelled supported.
* Accuracy is the primary endpoint.  Passes (JSON ``epochs_completed``),
  NVML energy (strict JSON trace path) and training wall time are reported as
  co-primary costs.  ``wall_sec`` is ``training_duration_sec`` so the
  post-training diagnostic probes are outside the cost window; the full run
  duration is retained as a separate descriptive column.
* Paired contrasts use a two-sided Wilcoxon signed-rank test and a 10,000
  draw percentile bootstrap of the paired mean (fixed seed 0).  The family
  {H-W1a, H-W1b, H-W1c} is Holm-corrected on CIFAR-100 only.
* H-W1a and H-W1c are conjunctions.  Their p-value is the intersection-union
  maximum of the component paired tests (conservative; the declared family
  still has three hypotheses, not five).  The cost side of H-W1c is the
  paired mean saving fraction, with a >= 15% threshold.
* H-W1d and H-W1cap are controls: raw CIs, never in Holm.

The 2x2 attribution table is the one specified in §2:
    {MF-Joint, MF-Simul} x {MF-Seq, Seq-Global}
with factors loss attachment (output-only vs every stage) and prefix
gradients (trainable vs frozen/detached).  Effects are paired by seed.

Writes ``w1_per_seed.csv``, ``w1_summary.md`` and the machine-readable
tables named in ``main``.  The shared append-only run log is never read.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from scripts.jpdc.analyze_g1 import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    run_energy_wh,
    run_passes_json,
)

SEEDS = list(range(42, 62))
DATASETS = ("cifar10", "cifar100")
PRIMARY_DATASET = "cifar100"
ALPHA = 0.05
N_REQUIRED = len(SEEDS)

# Internal arm names; Seq-Global's experiment prefix is mf_global.
ARMS: Dict[str, str] = {
    "bp": "bp_{dataset}_mlp_3x2000",
    "mf_joint": "mf_joint_{dataset}_mlp_3x2000",
    "mf_simul": "mf_simul_{dataset}_mlp_3x2000",
    "mf_seq": "mf_{dataset}_mlp_3x2000",
    "mf_global": "mf_global_{dataset}_mlp_3x2000",
    "mf_nom0": "mf_nom0_{dataset}_mlp_3x2000",
}
CONTROL_ARMS: Dict[str, str] = {
    "bp_fixed": "bp_fixed_{dataset}_mlp_3x2000",
    "bp_bestval": "bp_bestval_{dataset}_mlp_3x2000",
    "bp_sched": "bp_sched_{dataset}_mlp_3x2000",
    "mf_seq_protocol_b": "mf_protocol_b_{dataset}_mlp_3x2000",
    "mf_nom0_protocol_b": "mf_nom0_protocol_b_{dataset}_mlp_3x2000",
    "mf_seq_cap300": "mf_cap300_{dataset}_mlp_3x2000",
}
EXPERIMENTS = {**ARMS, **CONTROL_ARMS}
DISPLAY = {
    "bp": "BP",
    "mf_joint": "MF-Joint",
    "mf_simul": "MF-Simul",
    "mf_seq": "MF-Seq",
    "mf_global": "Seq-Global",
    "mf_nom0": "MF-Seq-nom0",
    "bp_fixed": "BP-fixed",
    "bp_bestval": "BP-bestval",
    "bp_sched": "BP-sched",
    "mf_seq_protocol_b": "MF-Seq Protocol B",
    "mf_nom0_protocol_b": "nom0 Protocol B",
    "mf_seq_cap300": "MF-Seq cap-300",
}
MAIN_ARMS = ("bp", "mf_joint", "mf_simul", "mf_seq", "mf_global", "mf_nom0")
COST_METRICS = ("passes", "energy_wh", "wall_sec")

PER_SEED_COLUMNS = [
    "arm",
    "dataset",
    "seed",
    "test_accuracy",
    "test_loss",
    "passes",
    "energy_wh",
    "wall_sec",
    "total_run_duration_sec",
    "epochs_completed",
    "energy_source",
    "wall_source",
]


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _iter_records(
    results_dir: Path,
) -> Iterable[Tuple[str, str, int, Dict[str, Any]]]:
    for arm, pattern in EXPERIMENTS.items():
        for dataset in DATASETS:
            experiment = pattern.format(dataset=dataset)
            for seed in SEEDS:
                path = (
                    results_dir
                    / "runs"
                    / experiment
                    / f"{experiment}_seed{seed}.json"
                )
                if not path.exists():
                    continue
                try:
                    record = json.loads(path.read_text())
                except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                    continue
                if not isinstance(record, dict):
                    continue
                yield arm, dataset, seed, record


def collect(results_dir: Path) -> pd.DataFrame:
    """Collect one row per available run, without imputing missing seeds."""
    rows: List[Dict[str, Any]] = []
    for arm, dataset, seed, record in _iter_records(results_dir):
        accuracy = _as_float(record.get("test_accuracy"))
        passes = _as_float(run_passes_json(record))
        energy = _as_float(run_energy_wh(results_dir, record))
        training_duration = _as_float(record.get("training_duration_sec"))
        total_duration = _as_float(record.get("total_run_duration_sec"))
        if math.isfinite(training_duration):
            wall_sec = training_duration
            wall_source = "training_duration_sec"
        else:
            wall_sec = total_duration
            wall_source = "total_run_duration_sec"
        rows.append(
            {
                "arm": arm,
                "dataset": dataset,
                "seed": seed,
                "test_accuracy": accuracy,
                "test_loss": _as_float(record.get("test_loss")),
                "passes": passes,
                "energy_wh": energy,
                "wall_sec": wall_sec,
                "total_run_duration_sec": total_duration,
                "epochs_completed": _as_float(record.get("epochs_completed")),
                "energy_source": "json" if math.isfinite(energy) else "missing",
                "wall_source": wall_source,
            }
        )
    if not rows:
        return pd.DataFrame(columns=PER_SEED_COLUMNS)
    return pd.DataFrame(rows, columns=PER_SEED_COLUMNS)


def _series(df: pd.DataFrame, arm: str, dataset: str, metric: str) -> pd.Series:
    sub = df[(df.arm == arm) & (df.dataset == dataset)]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.set_index("seed")[metric].astype(float)


def _bootstrap_ci(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = values[
        rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))
    ].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _wilcoxon_p(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 3:
        return float("nan")
    if np.allclose(diffs, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(diffs, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def _diff_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p": float("nan"),
        }
    lo, hi = _bootstrap_ci(values)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci_low": lo,
        "ci_high": hi,
        "p": _wilcoxon_p(values),
    }


def paired(
    df: pd.DataFrame, arm_a: str, arm_b: str, dataset: str, metric: str
) -> Dict[str, float]:
    """Paired-by-seed difference ``arm_a - arm_b`` with Welch-free bootstrap."""
    a = _series(df, arm_a, dataset, metric)
    b = _series(df, arm_b, dataset, metric)
    common = sorted(a.index.intersection(b.index))
    if common:
        av = a.loc[common].to_numpy(dtype=float)
        bv = b.loc[common].to_numpy(dtype=float)
        ok = np.isfinite(av) & np.isfinite(bv)
        av, bv = av[ok], bv[ok]
        diffs = av - bv
    else:
        diffs = np.array([], dtype=float)
    result = _diff_stats(diffs)
    result.update(
        {
            "n": int(len(diffs)),
            "mean_a": float(np.mean(diffs + bv)) if len(diffs) else float("nan"),
            "mean_b": float(np.mean(bv)) if len(diffs) else float("nan"),
        }
    )
    return result


def paired_ratio(
    df: pd.DataFrame, arm_a: str, arm_b: str, dataset: str, metric: str
) -> Dict[str, float]:
    """Paired ratio ``arm_a / arm_b`` per seed (cost endpoint helper)."""
    a = _series(df, arm_a, dataset, metric)
    b = _series(df, arm_b, dataset, metric)
    common = sorted(a.index.intersection(b.index))
    if not common:
        return _diff_stats(np.array([], dtype=float))
    av = a.loc[common].to_numpy(dtype=float)
    bv = b.loc[common].to_numpy(dtype=float)
    ok = np.isfinite(av) & np.isfinite(bv) & (bv != 0)
    ratios = av[ok] / bv[ok]
    result = _diff_stats(ratios)
    result["n"] = int(len(ratios))
    return result


def saving_fraction(
    df: pd.DataFrame, baseline: str, reduced: str, dataset: str, metric: str
) -> Dict[str, float]:
    """Paired saving ``(baseline - reduced) / baseline`` per seed."""
    base = _series(df, baseline, dataset, metric)
    low = _series(df, reduced, dataset, metric)
    common = sorted(base.index.intersection(low.index))
    if not common:
        return _diff_stats(np.array([], dtype=float))
    b = base.loc[common].to_numpy(dtype=float)
    r = low.loc[common].to_numpy(dtype=float)
    ok = np.isfinite(b) & np.isfinite(r) & (b != 0)
    fractions = (b[ok] - r[ok]) / b[ok]
    result = _diff_stats(fractions)
    result["n"] = int(len(fractions))
    return result


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, order preserved; NaN stays NaN."""
    values = [float(v) for v in p_values]
    finite = [(i, v) for i, v in enumerate(values) if math.isfinite(v)]
    adjusted = [float("nan")] * len(values)
    if not finite:
        return adjusted
    finite.sort(key=lambda item: item[1])
    m = len(finite)
    running = 0.0
    for rank, (index, value) in enumerate(finite):
        candidate = min(1.0, (m - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def arm_present(df: pd.DataFrame, arm: str, dataset: str) -> bool:
    return not df[(df.arm == arm) & (df.dataset == dataset)].empty


def missing_seeds(df: pd.DataFrame, arm: str, dataset: str) -> List[int]:
    present = set(
        int(seed)
        for seed in df[(df.arm == arm) & (df.dataset == dataset)]["seed"].tolist()
    )
    return sorted(set(SEEDS) - present)


def two_by_two(
    df: pd.DataFrame, dataset: str, metric: str
) -> Dict[str, Any]:
    """Paired 2x2 cell means and factorial effects for one metric.

    Coding: loss attachment output-only = -1, every-stage = +1; prefix
    gradients frozen/detached = -1, trainable = +1.  Cells are MF-Joint
    (output, trainable), MF-Simul (output, frozen), MF-Seq (every-stage,
    frozen), Seq-Global (every-stage, trainable).
    """
    cells = {
        "mf_joint": _series(df, "mf_joint", dataset, metric),
        "mf_simul": _series(df, "mf_simul", dataset, metric),
        "mf_seq": _series(df, "mf_seq", dataset, metric),
        "mf_global": _series(df, "mf_global", dataset, metric),
    }
    frame = pd.concat(cells, axis=1).dropna()
    cell_means = {
        arm: {
            "n": int(cells[arm].notna().sum()),
            "mean": float(cells[arm].mean()) if cells[arm].notna().any() else float("nan"),
            "median": float(cells[arm].median()) if cells[arm].notna().any() else float("nan"),
        }
        for arm in cells
    }
    if frame.empty:
        empty = _diff_stats(np.array([], dtype=float))
        return {
            "n": 0,
            "cells": cell_means,
            "loss_attachment": empty,
            "prefix_gradients": empty,
            "interaction": empty,
        }
    j = frame["mf_joint"].to_numpy(dtype=float)
    s = frame["mf_simul"].to_numpy(dtype=float)
    q = frame["mf_seq"].to_numpy(dtype=float)
    g = frame["mf_global"].to_numpy(dtype=float)
    effects = {
        "loss_attachment": ((g + q) / 2.0) - ((j + s) / 2.0),
        "prefix_gradients": ((g + j) / 2.0) - ((q + s) / 2.0),
        "interaction": g - q - j + s,
    }
    return {
        "n": int(len(frame)),
        "cells": cell_means,
        **{name: _diff_stats(values) for name, values in effects.items()},
    }


def _complete(stat: Dict[str, float], required_n: int = N_REQUIRED) -> bool:
    return stat.get("n", 0) == required_n and math.isfinite(stat.get("mean", float("nan")))


def _status_a(
    complete: bool,
    d_seq: Dict[str, float],
    d_simul: Dict[str, float],
    p_value: float,
    corrected: bool,
) -> str:
    if not complete:
        n = min(d_seq.get("n", 0), d_simul.get("n", 0))
        return "not run (arm missing)" if n == 0 else f"incomplete (n={n} < {N_REQUIRED})"
    threshold = d_seq["mean"] <= -2.0 and d_simul["mean"] <= 1.0
    if not threshold:
        return "not supported (threshold not met)"
    if math.isfinite(p_value) and p_value < ALPHA:
        return "supported" if corrected else "consistency threshold met (raw p < .05)"
    return (
        "threshold met; not significant after Holm"
        if corrected
        else "consistency threshold met; raw p not significant"
    )


def _status_b(
    complete: bool, d_seq: Dict[str, float], p_value: float, corrected: bool
) -> str:
    if not complete:
        n = d_seq.get("n", 0)
        return "not run (arm missing)" if n == 0 else f"incomplete (n={n} < {N_REQUIRED})"
    if d_seq["mean"] < -1.0:
        return "not supported (Seq-Global loss exceeds -1.0 pp)"
    if math.isfinite(p_value) and p_value < ALPHA:
        return "supported" if corrected else "consistency threshold met (raw p < .05)"
    return (
        "threshold met; not significant after Holm"
        if corrected
        else "consistency threshold met; raw p not significant"
    )


def _status_c(
    complete: bool,
    d_acc: Dict[str, float],
    pass_save: Dict[str, float],
    energy_save: Dict[str, float],
    p_value: float,
    corrected: bool,
) -> str:
    if not complete:
        n = d_acc.get("n", 0)
        return "not run (arm missing)" if n == 0 else f"incomplete (n={n} < {N_REQUIRED})"
    if d_acc["mean"] <= -1.5:
        return "load-bearing (accuracy drop >= 1.5 pp; dead-stage claim withdrawn)"
    threshold = (
        d_acc["mean"] >= -0.5
        and pass_save["mean"] >= 0.15
        and energy_save["mean"] >= 0.15
    )
    if not threshold:
        return "not supported (accuracy or >=15% cost saving not met)"
    if math.isfinite(p_value) and p_value < ALPHA:
        return "supported" if corrected else "consistency threshold met (raw p < .05)"
    return (
        "threshold met; not significant after Holm"
        if corrected
        else "consistency threshold met; raw p not significant"
    )


def _status_d(complete: bool, d_acc: Dict[str, float]) -> str:
    if not complete:
        n = d_acc.get("n", 0)
        return "not run (arm missing)" if n == 0 else f"incomplete (n={n} < {N_REQUIRED})"
    if d_acc["mean"] < -1.0:
        return "MF-Seq advantage survives"
    if math.isfinite(d_acc.get("ci_low", float("nan"))) and d_acc["ci_low"] >= -1.0:
        return "compute-advantage framing retired (CI excludes worse than -1 pp)"
    return "point estimate is within -1 pp; CI inconclusive"


def hypothesis_rows(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute H-W1a..d and H-W1cap for both datasets."""
    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    for dataset in DATASETS:
        d_gs = paired(df, "mf_global", "mf_seq", dataset, "test_accuracy")
        d_gsim = paired(df, "mf_global", "mf_simul", dataset, "test_accuracy")
        d_ns = paired(df, "mf_nom0", "mf_seq", dataset, "test_accuracy")
        d_bs = paired(df, "bp_fixed", "mf_seq", dataset, "test_accuracy")
        pass_save = saving_fraction(df, "mf_seq", "mf_nom0", dataset, "passes")
        energy_save = saving_fraction(df, "mf_seq", "mf_nom0", dataset, "energy_wh")
        pass_ratio = paired_ratio(df, "bp_fixed", "mf_seq", dataset, "passes")

        p_a = (
            max(d_gs["p"], d_gsim["p"])
            if math.isfinite(d_gs["p"]) and math.isfinite(d_gsim["p"])
            else float("nan")
        )
        p_family = [p_a, d_gs["p"], d_ns["p"]]
        is_primary = dataset == PRIMARY_DATASET
        p_holm = (
            holm(p_family)
            if is_primary and all(math.isfinite(p) for p in p_family)
            else [float("nan")] * 3
        )

        complete_a = _complete(d_gs) and _complete(d_gsim)
        complete_b = _complete(d_gs)
        complete_c = _complete(d_ns) and _complete(pass_save) and _complete(energy_save)
        complete_d = _complete(d_bs) and _complete(pass_ratio)
        complete_cap = _complete(
            paired(df, "mf_seq_cap300", "mf_seq", dataset, "test_accuracy")
        )

        status_a = _status_a(
            complete_a, d_gs, d_gsim, p_holm[0] if is_primary else p_a, is_primary
        )
        status_b = _status_b(
            complete_b, d_gs, p_holm[1] if is_primary else d_gs["p"], is_primary
        )
        status_c = _status_c(
            complete_c,
            d_ns,
            pass_save,
            energy_save,
            p_holm[2] if is_primary else d_ns["p"],
            is_primary,
        )
        status_d = _status_d(complete_d, d_bs)

        cap = paired(df, "mf_seq_cap300", "mf_seq", dataset, "test_accuracy")
        cap_status = "not run"
        if cap["n"]:
            cap_status = (
                f"delta {cap['mean']:+.2f} pp "
                f"[{cap['ci_low']:+.2f}, {cap['ci_high']:+.2f}]"
                if math.isfinite(cap["mean"])
                else "incomplete"
            )

        details[dataset] = {
            "d_gs": d_gs,
            "d_gsim": d_gsim,
            "d_ns": d_ns,
            "d_bs": d_bs,
            "pass_save": pass_save,
            "energy_save": energy_save,
            "pass_ratio": pass_ratio,
            "p_a": p_a,
            "p_holm": p_holm,
            "cap": cap,
        }

        rows.extend(
            [
                {
                    "dataset": dataset,
                    "hypothesis": "H-W1a",
                    "status": status_a,
                    "n": min(d_gs["n"], d_gsim["n"]),
                    "complete_n20": complete_a,
                    "contrast_a": "Seq-Global - MF-Seq",
                    "delta_a": d_gs["mean"],
                    "ci_a_low": d_gs["ci_low"],
                    "ci_a_high": d_gs["ci_high"],
                    "p_a": d_gs["p"],
                    "contrast_b": "Seq-Global - MF-Simul",
                    "delta_b": d_gsim["mean"],
                    "ci_b_low": d_gsim["ci_low"],
                    "ci_b_high": d_gsim["ci_high"],
                    "p_b": d_gsim["p"],
                    "p_raw": p_a,
                    "p_holm": p_holm[0],
                    "cost_a": float("nan"),
                    "cost_b": float("nan"),
                    "note": "threshold: <= -2.0 pp and <= +1.0 pp",
                },
                {
                    "dataset": dataset,
                    "hypothesis": "H-W1b",
                    "status": status_b,
                    "n": d_gs["n"],
                    "complete_n20": complete_b,
                    "contrast_a": "Seq-Global - MF-Seq",
                    "delta_a": d_gs["mean"],
                    "ci_a_low": d_gs["ci_low"],
                    "ci_a_high": d_gs["ci_high"],
                    "p_a": d_gs["p"],
                    "contrast_b": "",
                    "delta_b": float("nan"),
                    "ci_b_low": float("nan"),
                    "ci_b_high": float("nan"),
                    "p_b": float("nan"),
                    "p_raw": d_gs["p"],
                    "p_holm": p_holm[1],
                    "cost_a": float("nan"),
                    "cost_b": float("nan"),
                    "note": "threshold: >= -1.0 pp",
                },
                {
                    "dataset": dataset,
                    "hypothesis": "H-W1c",
                    "status": status_c,
                    "n": d_ns["n"],
                    "complete_n20": complete_c,
                    "contrast_a": "MF-Seq-nom0 - MF-Seq",
                    "delta_a": d_ns["mean"],
                    "ci_a_low": d_ns["ci_low"],
                    "ci_a_high": d_ns["ci_high"],
                    "p_a": d_ns["p"],
                    "contrast_b": "",
                    "delta_b": float("nan"),
                    "ci_b_low": float("nan"),
                    "ci_b_high": float("nan"),
                    "p_b": float("nan"),
                    "p_raw": d_ns["p"],
                    "p_holm": p_holm[2],
                    "cost_a": pass_save["mean"],
                    "cost_b": energy_save["mean"],
                    "note": "threshold: acc >= -0.5 pp and pass/energy saving >= 15%",
                },
                {
                    "dataset": dataset,
                    "hypothesis": "H-W1d",
                    "status": status_d,
                    "n": d_bs["n"],
                    "complete_n20": complete_d,
                    "contrast_a": "BP-fixed - MF-Seq",
                    "delta_a": d_bs["mean"],
                    "ci_a_low": d_bs["ci_low"],
                    "ci_a_high": d_bs["ci_high"],
                    "p_a": d_bs["p"],
                    "contrast_b": "",
                    "delta_b": float("nan"),
                    "ci_b_low": float("nan"),
                    "ci_b_high": float("nan"),
                    "p_b": float("nan"),
                    "p_raw": d_bs["p"],
                    "p_holm": float("nan"),
                    "cost_a": pass_ratio["mean"],
                    "cost_b": float("nan"),
                    "note": "control; raw CI; passes ratio is descriptive (~1x design)",
                },
                {
                    "dataset": dataset,
                    "hypothesis": "H-W1cap",
                    "status": cap_status,
                    "n": cap["n"],
                    "complete_n20": complete_cap,
                    "contrast_a": "MF-Seq cap-300 - MF-Seq",
                    "delta_a": cap["mean"],
                    "ci_a_low": cap["ci_low"],
                    "ci_a_high": cap["ci_high"],
                    "p_a": cap["p"],
                    "contrast_b": "",
                    "delta_b": float("nan"),
                    "ci_b_low": float("nan"),
                    "ci_b_high": float("nan"),
                    "p_b": float("nan"),
                    "p_raw": cap["p"],
                    "p_holm": float("nan"),
                    "cost_a": float("nan"),
                    "cost_b": float("nan"),
                    "note": "conditional full arm; Step 1 pilot gates it",
                },
            ]
        )
    return rows, details


def cost_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    contrast_defs = [
        ("H-W1a", "mf_global", "mf_seq"),
        ("H-W1a", "mf_global", "mf_simul"),
        ("H-W1b", "mf_global", "mf_seq"),
        ("H-W1c", "mf_nom0", "mf_seq"),
        ("H-W1d", "bp_fixed", "mf_seq"),
    ]
    for dataset in DATASETS:
        for hypothesis, arm_a, arm_b in contrast_defs:
            for metric in ("test_accuracy", *COST_METRICS):
                stat = paired(df, arm_a, arm_b, dataset, metric)
                ratio = (
                    paired_ratio(df, arm_a, arm_b, dataset, metric)
                    if metric != "test_accuracy"
                    else None
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "hypothesis": hypothesis,
                        "contrast": f"{DISPLAY[arm_a]} - {DISPLAY[arm_b]}",
                        "metric": metric,
                        "n": stat["n"],
                        "mean_diff": stat["mean"],
                        "median_diff": stat["median"],
                        "ci_low": stat["ci_low"],
                        "ci_high": stat["ci_high"],
                        "p": stat["p"],
                        "ratio_mean": ratio["mean"] if ratio else float("nan"),
                        "ratio_low": ratio["ci_low"] if ratio else float("nan"),
                        "ratio_high": ratio["ci_high"] if ratio else float("nan"),
                    }
                )
    return rows


CONTROL_CONTRASTS = (
    ("BP-bestval - BP", "bp_bestval", "bp"),
    ("BP-sched - BP", "bp_sched", "bp"),
    ("BP-fixed - BP", "bp_fixed", "bp"),
    ("MF-Seq Protocol B - BP", "mf_seq_protocol_b", "bp"),
    ("nom0 Protocol B - MF-Seq Protocol B", "mf_nom0_protocol_b", "mf_seq_protocol_b"),
    ("MF-Seq cap-300 - MF-Seq", "mf_seq_cap300", "mf_seq"),
)


def control_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Descriptive controls; never Holm-corrected and never in a verdict."""
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for label, arm_a, arm_b in CONTROL_CONTRASTS:
            for metric in ("test_accuracy", *COST_METRICS):
                stat = paired(df, arm_a, arm_b, dataset, metric)
                ratio = (
                    paired_ratio(df, arm_a, arm_b, dataset, metric)
                    if metric != "test_accuracy"
                    else None
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "control": label,
                        "metric": metric,
                        "n": stat["n"],
                        "mean_diff": stat["mean"],
                        "median_diff": stat["median"],
                        "ci_low": stat["ci_low"],
                        "ci_high": stat["ci_high"],
                        "p": stat["p"],
                        "ratio_mean": ratio["mean"] if ratio else float("nan"),
                        "ratio_low": ratio["ci_low"] if ratio else float("nan"),
                        "ratio_high": ratio["ci_high"] if ratio else float("nan"),
                    }
                )
    return rows


def _control_lines(rows: List[Dict[str, Any]], dataset: str) -> List[str]:
    lines = [f"### {dataset} descriptive controls (raw CIs; not in Holm)", ""]
    header = ["control", "metric", "n", "mean diff", "95% CI", "raw p", "ratio [95% CI]"]
    table = []
    for row in rows:
        if row["dataset"] != dataset:
            continue
        ratio = (
            f"{_fmt(row['ratio_mean'], 3)} "
            f"[{_fmt(row['ratio_low'], 3)}, {_fmt(row['ratio_high'], 3)}]"
            if math.isfinite(row["ratio_mean"])
            else "n/a"
        )
        table.append(
            [
                row["control"],
                row["metric"],
                str(row["n"]),
                _fmt(row["mean_diff"], 3, True),
                f"[{_fmt(row['ci_low'], 3, True)}, {_fmt(row['ci_high'], 3, True)}]",
                _fmt(row["p"], 4),
                ratio,
            ]
        )
    lines.extend(_md_table(header, table))
    lines.append("")
    return lines


def collect_secondary(
    results_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return long secondary tables: run, probe, head, stage."""
    run_rows: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    stage_rows: List[Dict[str, Any]] = []
    for arm, dataset, seed, record in _iter_records(results_dir):
        base = {"arm": arm, "dataset": dataset, "seed": seed}
        keys = (
            "final_train_loss",
            "final_train_accuracy",
            "final_val_loss",
            "final_val_accuracy",
            "final_test_loss",
            "final_test_accuracy",
            "secondary_eval_seconds",
            "secondary_eval_error",
            "secondary_eval_scope",
        )
        if any(key in record for key in keys):
            run_rows.append(
                {
                    **base,
                    **{key: record.get(key, float("nan")) for key in keys},
                }
            )
        probes = record.get("linear_probe_test_accuracy") or {}
        ranks = record.get("effective_rank") or {}
        if isinstance(probes, dict) or isinstance(ranks, dict):
            layers = sorted(
                set(probes) | set(ranks),
                key=lambda value: int(str(value).split("_")[-1])
                if str(value).split("_")[-1].isdigit()
                else str(value),
            )
            for layer in layers:
                probe_rows.append(
                    {
                        **base,
                        "layer": layer,
                        "probe_accuracy": _as_float(probes.get(layer)),
                        "effective_rank": _as_float(ranks.get(layer)),
                    }
                )
        heads = record.get("local_head_test_accuracy") or {}
        if isinstance(heads, dict):
            for head in sorted(heads):
                head_rows.append(
                    {**base, "head": head, "accuracy": _as_float(heads[head])}
                )
        stages = record.get("stage_epochs") or []
        if isinstance(stages, list):
            for stage in stages:
                if not isinstance(stage, dict):
                    continue
                stage_rows.append(
                    {
                        **base,
                        "stage": stage.get("stage", "unknown"),
                        "epochs": _as_float(stage.get("epochs")),
                        "stopping_epoch": _as_float(stage.get("stopping_epoch")),
                        "max_epochs": _as_float(stage.get("max_epochs")),
                        "stopped_early": bool(stage.get("stopped_early", False)),
                    }
                )
    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(probe_rows),
        pd.DataFrame(head_rows),
        pd.DataFrame(stage_rows),
    )


def _mean_table(
    frame: pd.DataFrame, group_cols: Sequence[str], value_cols: Sequence[str]
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_cols, "n", *value_cols])
    grouped = frame.groupby(list(group_cols), dropna=False)
    out = grouped[list(value_cols)].mean().reset_index()
    out.insert(len(group_cols), "n", grouped.size().to_numpy())
    return out


def secondary_tables(
    run_secondary: pd.DataFrame,
    probes: pd.DataFrame,
    heads: pd.DataFrame,
    stages: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    summary = _mean_table(
        run_secondary,
        ("arm", "dataset"),
        (
            "final_train_loss",
            "final_train_accuracy",
            "final_val_loss",
            "final_val_accuracy",
            "final_test_loss",
            "final_test_accuracy",
            "secondary_eval_seconds",
        ),
    )
    probe_summary = _mean_table(
        probes, ("arm", "dataset", "layer"), ("probe_accuracy", "effective_rank")
    )
    head_summary = _mean_table(heads, ("arm", "dataset", "head"), ("accuracy",))
    stage_summary = _mean_table(
        stages,
        ("arm", "dataset", "stage"),
        ("epochs", "stopping_epoch", "max_epochs"),
    )
    return {
        "run": summary,
        "probe": probe_summary,
        "head": head_summary,
        "stage": stage_summary,
    }


def _fmt(value: Any, digits: int = 2, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return fmt.format(number)


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _arm_summary_lines(df: pd.DataFrame) -> List[str]:
    lines = ["## Per-arm endpoints (mean [95% CI])", ""]
    header = ["arm", "dataset", "n", "acc %", "passes", "energy Wh", "training wall s"]
    rows = []
    for arm in EXPERIMENTS:
        for dataset in DATASETS:
            sub = df[(df.arm == arm) & (df.dataset == dataset)]
            if sub.empty:
                continue
            cells = [DISPLAY[arm], dataset, str(len(sub))]
            for metric in ("test_accuracy", "passes", "energy_wh", "wall_sec"):
                values = sub[metric].astype(float).dropna().to_numpy()
                if len(values) < 2:
                    cells.append(_fmt(values.mean() if len(values) else float("nan")))
                    continue
                half = stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)
                cells.append(
                    f"{values.mean():.2f} [{values.mean() - half:.2f}, "
                    f"{values.mean() + half:.2f}]"
                )
            rows.append(cells)
    lines.extend(_md_table(header, rows))
    lines.append("")
    return lines


def _medians_lines(df: pd.DataFrame) -> List[str]:
    lines = ["## Per-arm medians (PREREG requires mean and median)", ""]
    header = ["arm", "dataset", "n", "acc %", "passes", "energy Wh", "training wall s"]
    rows = []
    for arm in EXPERIMENTS:
        for dataset in DATASETS:
            sub = df[(df.arm == arm) & (df.dataset == dataset)]
            if sub.empty:
                continue
            rows.append(
                [
                    DISPLAY[arm],
                    dataset,
                    str(len(sub)),
                    _fmt(sub.test_accuracy.median()),
                    _fmt(sub.passes.median(), 1),
                    _fmt(sub.energy_wh.median()),
                    _fmt(sub.wall_sec.median(), 1),
                ]
            )
    lines.extend(_md_table(header, rows))
    lines.append("")
    return lines


def _two_by_two_lines(df: pd.DataFrame) -> List[str]:
    lines = ["## 2x2 attribution table", ""]
    lines.append(
        "Cells: rows = loss attachment, columns = prefix gradients. "
        "Effects are paired-by-seed; positive loss attachment means every-stage "
        "loss has higher values than output-only loss; positive prefix gradient "
        "means trainable prefix has higher values than frozen/detached."
    )
    lines.append("")
    for dataset in DATASETS:
        table = two_by_two(df, dataset, "test_accuracy")
        lines.append(f"### {dataset} — accuracy (%)")
        lines.append("")
        header = [
            "loss attachment",
            "frozen/detached prefix",
            "trainable prefix",
            "main effect A",
            "main effect P",
            "interaction",
        ]
        cells = table["cells"]
        rows = [
            [
                "output-only",
                f"{DISPLAY['mf_simul']}: {_fmt(cells['mf_simul']['mean'])} "
                f"(n={cells['mf_simul']['n']})",
                f"{DISPLAY['mf_joint']}: {_fmt(cells['mf_joint']['mean'])} "
                f"(n={cells['mf_joint']['n']})",
                _fmt(table["loss_attachment"]["mean"], 2, True),
                _fmt(table["prefix_gradients"]["mean"], 2, True),
                _fmt(table["interaction"]["mean"], 2, True),
            ],
            [
                "every-stage",
                f"{DISPLAY['mf_seq']}: {_fmt(cells['mf_seq']['mean'])} "
                f"(n={cells['mf_seq']['n']})",
                f"{DISPLAY['mf_global']}: {_fmt(cells['mf_global']['mean'])} "
                f"(n={cells['mf_global']['n']})",
                "",
                "",
                "",
            ],
        ]
        lines.extend(_md_table(header, rows))
        lines.append("")
        for metric in ("test_accuracy", *COST_METRICS):
            effect = two_by_two(df, dataset, metric)
            rows = []
            for name in ("loss_attachment", "prefix_gradients", "interaction"):
                stat = effect[name]
                rows.append(
                    [
                        name,
                        str(stat["n"]),
                        _fmt(stat["mean"], 3, True),
                        _fmt(stat["ci_low"], 3, True),
                        _fmt(stat["ci_high"], 3, True),
                        _fmt(stat["p"], 4),
                    ]
                )
            lines.append(f"**{dataset} / {metric} effects**")
            lines.append("")
            lines.extend(
                _md_table(
                    ["effect", "n", "mean", "95% CI low", "95% CI high", "raw p"],
                    rows,
                )
            )
            lines.append("")
    return lines


def _hypothesis_lines(rows: List[Dict[str, Any]], dataset: str) -> List[str]:
    lines = [f"### {dataset} hypotheses", ""]
    header = [
        "hypothesis",
        "status",
        "n",
        "contrast",
        "delta",
        "95% CI",
        "raw p",
        "Holm p",
        "note",
    ]
    table = []
    for row in rows:
        if row["dataset"] != dataset:
            continue
        contrast = row["contrast_a"]
        delta = _fmt(row["delta_a"], 2, True)
        ci = f"[{_fmt(row['ci_a_low'], 2, True)}, {_fmt(row['ci_a_high'], 2, True)}]"
        if row["hypothesis"] == "H-W1a":
            contrast += " AND " + row["contrast_b"]
            delta += " / " + _fmt(row["delta_b"], 2, True)
            ci += " / [" + _fmt(row["ci_b_low"], 2, True) + ", " + _fmt(row["ci_b_high"], 2, True) + "]"
        if row["hypothesis"] == "H-W1c":
            delta += " (pass save " + _fmt(row["cost_a"] * 100, 0) + "%, energy save " + _fmt(row["cost_b"] * 100, 0) + "%)"
        if row["hypothesis"] == "H-W1d":
            delta += " (passes ratio " + _fmt(row["cost_a"], 2) + ")"
        table.append(
            [
                row["hypothesis"],
                row["status"],
                str(row["n"]),
                contrast,
                delta,
                ci,
                _fmt(row["p_raw"], 4),
                _fmt(row["p_holm"], 4),
                row["note"],
            ]
        )
    lines.extend(_md_table(header, table))
    lines.append("")
    return lines


def _cost_lines(rows: List[Dict[str, Any]], dataset: str) -> List[str]:
    lines = [f"### {dataset} cost endpoints", ""]
    lines.append(
        "Accuracy is test accuracy; passes, energy and wall are paired co-primary "
        "endpoints. Ratios are arm_a/arm_b; training wall excludes post-training probes."
    )
    lines.append("")
    header = [
        "hypothesis",
        "contrast",
        "metric",
        "n",
        "mean diff",
        "95% CI",
        "raw p",
        "ratio [95% CI]",
    ]
    table = []
    for row in rows:
        if row["dataset"] != dataset:
            continue
        ratio = (
            f"{_fmt(row['ratio_mean'], 3)} "
            f"[{_fmt(row['ratio_low'], 3)}, {_fmt(row['ratio_high'], 3)}]"
            if math.isfinite(row["ratio_mean"])
            else "n/a"
        )
        table.append(
            [
                row["hypothesis"],
                row["contrast"],
                row["metric"],
                str(row["n"]),
                _fmt(row["mean_diff"], 3, True),
                f"[{_fmt(row['ci_low'], 3, True)}, {_fmt(row['ci_high'], 3, True)}]",
                _fmt(row["p"], 4),
                ratio,
            ]
        )
    lines.extend(_md_table(header, table))
    lines.append("")
    return lines


def _secondary_lines(tables: Dict[str, pd.DataFrame]) -> List[str]:
    lines = ["## Secondary diagnostic tables (not in Holm or any verdict)", ""]
    lines.append(
        "These are post-training observations from each run's own JSON. They "
        "never enter a decision rule. `secondary_eval_seconds` is measured after "
        "the NVML monitor stops; the full run duration is not used for the wall endpoint."
    )
    lines.append("")
    summary = tables["run"]
    if summary.empty:
        lines.append("No secondary instrumentation fields found in the run JSONs.")
        lines.append("")
        return lines
    header = [
        "arm",
        "dataset",
        "n",
        "train acc",
        "train loss",
        "val acc",
        "val loss",
        "test acc",
        "test loss",
        "probe sec",
    ]
    rows = []
    for _, row in summary.sort_values(["arm", "dataset"]).iterrows():
        rows.append(
            [
                DISPLAY.get(row["arm"], row["arm"]),
                row["dataset"],
                str(int(row["n"])),
                _fmt(row["final_train_accuracy"]),
                _fmt(row["final_train_loss"], 4),
                _fmt(row["final_val_accuracy"]),
                _fmt(row["final_val_loss"], 4),
                _fmt(row["final_test_accuracy"]),
                _fmt(row["final_test_loss"], 4),
                _fmt(row["secondary_eval_seconds"], 3),
            ]
        )
    lines.extend(_md_table(header, rows))
    lines.append("")
    probe = tables["probe"]
    if not probe.empty:
        lines.append("### Linear probes and effective rank")
        lines.append("")
        header = ["arm", "dataset", "layer", "n", "probe acc %", "effective rank"]
        rows = [
            [
                DISPLAY.get(row["arm"], row["arm"]),
                row["dataset"],
                row["layer"],
                str(int(row["n"])),
                _fmt(row["probe_accuracy"]),
                _fmt(row["effective_rank"], 3),
            ]
            for _, row in probe.sort_values(["arm", "dataset", "layer"]).iterrows()
        ]
        lines.extend(_md_table(header, rows))
        lines.append("")
    head = tables["head"]
    if not head.empty:
        lines.append("### Local-head accuracy")
        lines.append("")
        header = ["arm", "dataset", "head", "n", "accuracy %"]
        rows = [
            [
                DISPLAY.get(row["arm"], row["arm"]),
                row["dataset"],
                row["head"],
                str(int(row["n"])),
                _fmt(row["accuracy"]),
            ]
            for _, row in head.sort_values(["arm", "dataset", "head"]).iterrows()
        ]
        lines.extend(_md_table(header, rows))
        lines.append("")
    stage = tables["stage"]
    if not stage.empty:
        lines.append("### Stage epochs and stopping")
        lines.append("")
        header = ["arm", "dataset", "stage", "n", "mean epochs", "mean max", "stopped early"]
        rows = []
        for _, row in stage.sort_values(["arm", "dataset", "stage"]).iterrows():
            rows.append(
                [
                    DISPLAY.get(row["arm"], row["arm"]),
                    row["dataset"],
                    row["stage"],
                    str(int(row["n"])),
                    _fmt(row["epochs"], 2),
                    _fmt(row["max_epochs"], 1),
                    _fmt(row.get("stopped_early", float("nan")), 2),
                ]
            )
        lines.extend(_md_table(header, rows))
        lines.append("")
    return lines


def _missing_lines(df: pd.DataFrame) -> List[str]:
    lines = ["## Missing seeds and not-run arms", ""]
    not_run: List[str] = []
    partial: List[str] = []
    for arm in EXPERIMENTS:
        for dataset in DATASETS:
            present = len(df[(df.arm == arm) & (df.dataset == dataset)])
            if present == 0:
                not_run.append(f"{DISPLAY[arm]} / {dataset}")
                continue
            missing = missing_seeds(df, arm, dataset)
            if missing:
                partial.append(
                    f"- {DISPLAY[arm]} / {dataset}: missing {missing} "
                    f"(n={present}/{N_REQUIRED})"
                )
    if not_run:
        lines.append("- Not run: " + ", ".join(not_run))
    if partial:
        lines.extend(partial)
    if not not_run and not partial:
        lines.append("None: all arms/datasets have seeds 42-61.")
    lines.append("")
    return lines


def _tuning_lines() -> List[str]:
    return [
        "## Tuning disclosure and sources",
        "",
        "Seq-Global and BP-sched each receive 50 lr/wd trials per dataset; "
        "BP-bestval and both Protocol-B arms inherit frozen values and add 0 "
        "trials. The six core arms keep their original budgets.",
        "",
        "Passes come from each run JSON's `epochs_completed`; energy comes "
        "strictly from the exact `monitoring_csv_path` in that JSON. "
        "The shared append-only run log is never read. `wall_sec` is "
        "`training_duration_sec`, so instrumentation outside the NVML window "
        "cannot change the wall endpoint; `total_run_duration_sec` is retained "
        "as a descriptive column.",
        "",
    ]


def analyze(results_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """Run the complete W1 reduction and write all tables."""
    df = collect(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "w1_per_seed.csv", index=False)

    missing = []
    for arm in EXPERIMENTS:
        for dataset in DATASETS:
            absent = missing_seeds(df, arm, dataset)
            if absent:
                missing.append(
                    {
                        "arm": arm,
                        "dataset": dataset,
                        "missing_seeds": ",".join(str(seed) for seed in absent),
                        "n": len(SEEDS) - len(absent),
                    }
                )
    pd.DataFrame(missing).to_csv(out_dir / "w1_missing_seeds.csv", index=False)

    hypothesis, details = hypothesis_rows(df)
    pd.DataFrame(hypothesis).to_csv(out_dir / "w1_hypotheses.csv", index=False)
    costs = cost_rows(df)
    pd.DataFrame(costs).to_csv(out_dir / "w1_contrasts.csv", index=False)
    control = control_rows(df)
    pd.DataFrame(control).to_csv(out_dir / "w1_controls.csv", index=False)

    effect_rows = []
    for dataset in DATASETS:
        for metric in ("test_accuracy", *COST_METRICS):
            table = two_by_two(df, dataset, metric)
            for name in ("loss_attachment", "prefix_gradients", "interaction"):
                stat = table[name]
                effect_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "effect": name,
                        "n": stat["n"],
                        "mean": stat["mean"],
                        "median": stat["median"],
                        "ci_low": stat["ci_low"],
                        "ci_high": stat["ci_high"],
                        "p": stat["p"],
                    }
                )
    pd.DataFrame(effect_rows).to_csv(out_dir / "w1_2x2.csv", index=False)

    run_secondary, probes, heads, stages = collect_secondary(results_dir)
    run_secondary.to_csv(out_dir / "w1_secondary_per_seed.csv", index=False)
    probes.to_csv(out_dir / "w1_probe_per_seed.csv", index=False)
    heads.to_csv(out_dir / "w1_local_head_per_seed.csv", index=False)
    stages.to_csv(out_dir / "w1_stage_epochs.csv", index=False)
    tables = secondary_tables(run_secondary, probes, heads, stages)
    tables["run"].to_csv(out_dir / "w1_secondary_summary.csv", index=False)
    tables["probe"].to_csv(out_dir / "w1_probe_summary.csv", index=False)
    tables["head"].to_csv(out_dir / "w1_local_head_summary.csv", index=False)
    tables["stage"].to_csv(out_dir / "w1_stage_summary.csv", index=False)

    lines = ["# W1 — attribution ladder (PREREG-phase2.md §3/§4)", ""]
    lines.append(
        f"Reduced {len(df)} runs from `{results_dir}`; seeds 42–61; "
        f"primary dataset `{PRIMARY_DATASET}`; CIFAR-10 is the consistency check."
    )
    lines.append(
        "Passes from JSON `epochs_completed`; energy from each run's exact "
        "`monitoring_csv_path`; wall from `training_duration_sec`. Paired "
        "Wilcoxon + 10,000-draw bootstrap; Holm over {H-W1a, H-W1b, H-W1c} "
        "on CIFAR-100 only. Missing seeds are never imputed."
    )
    lines.append("")
    lines.extend(_missing_lines(df))
    lines.extend(_arm_summary_lines(df))
    lines.extend(_medians_lines(df))
    lines.extend(_two_by_two_lines(df))

    lines.append("## Primary hypotheses")
    lines.append("")
    lines.extend(_hypothesis_lines(hypothesis, PRIMARY_DATASET))
    lines.append(
        "### CIFAR-10 consistency (same thresholds; no Holm and no confirmatory support)"
    )
    lines.append("")
    lines.extend(_hypothesis_lines(hypothesis, "cifar10"))
    lines.extend(_cost_lines(costs, PRIMARY_DATASET))
    lines.extend(_cost_lines(costs, "cifar10"))
    lines.extend(_control_lines(control, PRIMARY_DATASET))
    lines.extend(_control_lines(control, "cifar10"))

    lines.append("## Decision notes")
    lines.append("")
    lines.append(
        "H-W1a/H-W1c conjunction p-values are the conservative intersection-union "
        "maximum of their component accuracy tests. H-W1a and H-W1b are "
        "threshold-exclusive; a point estimate in (-2.0, -1.0) pp is intermediate. "
        "H-W1d and H-W1cap are controls with raw CIs, never Holm-corrected. "
        "No conclusion is drawn from an arm with fewer than 20 paired seeds."
    )
    lines.append("")
    lines.extend(_secondary_lines(tables))
    lines.extend(_tuning_lines())

    (out_dir / "w1_summary.md").write_text("\n".join(lines))

    return {
        "per_seed": df,
        "hypotheses": hypothesis,
        "details": details,
        "costs": costs,
        "controls": control,
        "effects": pd.DataFrame(effect_rows),
        "secondary": tables,
        "summary": "\n".join(lines),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.results_dir, args.out_dir)
    print(result["summary"])


if __name__ == "__main__":
    main()
