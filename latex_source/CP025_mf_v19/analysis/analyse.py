#!/usr/bin/env python3
"""Every number in CP025.tex, derived from artifacts/tidy/runs.csv.

Single source of truth. Writes numbers.json (machine-checkable) and the LaTeX
table bodies under ../tables/. Superseded rows are dropped before anything else
happens, so a confounded run cannot reach the manuscript.

Run:  python3 analysis/analyse.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # repository root
TIDY = ROOT / "artifacts" / "tidy" / "runs.csv"
# The pre-registered ladder output, vendored beside this script so the build is
# self-contained; results/ is not version controlled.
LADDER_JSON = HERE / "ladder_analysis.json"
if not LADDER_JSON.exists():
    LADDER_JSON = ROOT / "results" / "ladder_analysis.json"
TABLES = HERE.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# Pre-registered constants, copied from scripts/analyze_ablation_ladder.py.
ACC_MARGIN_PP = 0.25
RATIO_MARGIN = 0.05

# The four MF/BP configurations, in ascending parameter count.
CONFIGS = [
    ("MNIST 2x1000", "MNIST", "2x1000", 3),
    ("Fashion-MNIST 2x1000", "Fashion-MNIST", "2x1000", 3),
    ("CIFAR-10 3x2000", "CIFAR-10", "3x2000", 4),
    ("CIFAR-100 3x2000", "CIFAR-100", "3x2000", 4),
]

# Trainable weights, excluding biases, from the architecture and task dimensions.
# MLP d_in -> h ... -> d_out. Stated in the manuscript, checked here.
SHAPES = {
    "MNIST 2x1000": (784, [1000, 1000], 10),
    "Fashion-MNIST 2x1000": (784, [1000, 1000], 10),
    "CIFAR-10 3x2000": (3072, [2000, 2000, 2000], 10),
    "CIFAR-100 3x2000": (3072, [2000, 2000, 2000], 100),
}

# BP's realised mean epoch count and the MF budget derived from it, transcribed
# from configs/diagnostics/*_equal_epochs.yaml.
BUDGET = {
    "MNIST 2x1000": dict(bp_mean_epochs=25.4, stages=3, per_stage=9, mf_total=27),
    "Fashion-MNIST 2x1000": dict(bp_mean_epochs=25.6, stages=3, per_stage=9, mf_total=27),
    "CIFAR-10 3x2000": dict(bp_mean_epochs=52.0, stages=4, per_stage=13, mf_total=52),
    "CIFAR-100 3x2000": dict(bp_mean_epochs=42.9, stages=4, per_stage=11, mf_total=44),
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_wide() -> pd.DataFrame:
    df = pd.read_csv(TIDY, low_memory=False)
    df["superseded"] = df["superseded"].fillna("")
    n_all = df["run_id"].nunique()
    df = df[df["superseded"] == ""]
    n_clean = df["run_id"].nunique()
    idx = [
        "run_id", "phase", "protocol", "experiment_name", "algorithm", "method",
        "rung", "cache_strategy", "dataset", "architecture", "config_group", "seed",
    ]
    # pivot_table silently drops rows whose index carries a NaN, and `rung` is
    # empty for the phase 4 runs, so fill before pivoting.
    for c in ("rung", "cache_strategy", "method"):
        df[c] = df[c].fillna("none")
    wide = df.pivot_table(index=idx, columns="metric", values="value", aggfunc="first").reset_index()
    assert len(wide) == n_clean, f"pivot lost runs: {len(wide)} != {n_clean}"
    print(f"[load] {n_all} runs in tidy table, {n_clean} after dropping superseded")
    return wide


def arm(w, cg, algorithm, protocol, cache="recompute"):
    return w[(w.config_group == cg) & (w.algorithm == algorithm)
             & (w.protocol == protocol) & (w.cache_strategy == cache)]


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def welch(a: np.ndarray, b: np.ndarray):
    """Welch difference b - a with its standard error and degrees of freedom."""
    d = b.mean() - a.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0.0:
        return d, 0.0, float("inf")
    dfw = se ** 4 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return d, se, dfw


def tost(a: np.ndarray, b: np.ndarray, margin: float):
    """Two one-sided tests for |mean(b) - mean(a)| < margin. Returns the verdict
    vocabulary of the pre-registered analysis: DIFFERENT, EQUIVALENT, INCONCLUSIVE."""
    d, se, dfw = welch(a, b)
    if se == 0.0:
        # Both arms deterministic: the gap is exact, no inference needed.
        return dict(diff=d, lo=d, hi=d, p_tost=0.0 if abs(d) < margin else 1.0,
                    p_diff=0.0 if d != 0 else 1.0,
                    verdict="EQUIVALENT" if abs(d) < margin else "DIFFERENT", exact=True)
    p_tost = max(stats.t.sf((d + margin) / se, dfw), stats.t.cdf((d - margin) / se, dfw))
    p_diff = 2 * stats.t.sf(abs(d) / se, dfw)
    lo = d - stats.t.ppf(0.95, dfw) * se
    hi = d + stats.t.ppf(0.95, dfw) * se
    if p_tost < 0.05:
        verdict = "EQUIVALENT"
    elif p_diff < 0.05:
        verdict = "DIFFERENT"
    else:
        verdict = "INCONCLUSIVE"
    return dict(diff=d, lo=lo, hi=hi, p_tost=p_tost, p_diff=p_diff, verdict=verdict, exact=False)


def paired_tost(bp: pd.DataFrame, mf: pd.DataFrame, metric: str, margin: float):
    """Seed-paired version, which is the pre-registered primary test."""
    m = bp[["seed", metric]].merge(mf[["seed", metric]], on="seed", suffixes=("_a", "_b")).dropna()
    d = (m[metric + "_b"] - m[metric + "_a"]).values
    out = dict(n_pairs=len(m), diff=float(d.mean()))
    if d.std(ddof=1) == 0:
        out.update(lo=float(d.mean()), hi=float(d.mean()), exact=True,
                   verdict="EQUIVALENT" if abs(d.mean()) < margin else "DIFFERENT")
        return out
    se = d.std(ddof=1) / math.sqrt(len(d))
    dfw = len(d) - 1
    p_tost = max(stats.t.sf((d.mean() + margin) / se, dfw), stats.t.cdf((d.mean() - margin) / se, dfw))
    p_diff = 2 * stats.t.sf(abs(d.mean()) / se, dfw)
    out.update(lo=float(d.mean() - stats.t.ppf(0.95, dfw) * se),
               hi=float(d.mean() + stats.t.ppf(0.95, dfw) * se),
               p_tost=float(p_tost), p_diff=float(p_diff), exact=False,
               verdict="EQUIVALENT" if p_tost < 0.05 else ("DIFFERENT" if p_diff < 0.05 else "INCONCLUSIVE"))
    return out


def describe(s: pd.Series):
    v = s.dropna()
    return dict(n=int(len(v)), mean=float(v.mean()),
                sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                median=float(v.median()))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
METRICS = ["test_accuracy", "total_gpu_energy_wh", "peak_torch_alloc_mib",
           "trace_mean_power_w", "training_duration_sec", "trace_mean_gpu_util_percent",
           "peak_gpu_mem_used_mib", "peak_process_rss_mib"]


def main() -> None:
    w = load_wide()
    out: dict = {"provenance": {"tidy": str(TIDY.relative_to(ROOT)),
                                "acc_margin_pp": ACC_MARGIN_PP,
                                "ratio_margin": RATIO_MARGIN}}

    # ---- parameter counts -------------------------------------------------- #
    params = {}
    for cg, (din, hid, dout) in SHAPES.items():
        dims = [din] + hid + [dout]
        params[cg] = int(sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1)))
    out["params"] = params
    out["budget"] = BUDGET
    for cg, b in BUDGET.items():
        assert math.ceil(b["bp_mean_epochs"] / b["stages"]) == b["per_stage"], cg
        assert b["per_stage"] * b["stages"] == b["mf_total"], cg
        assert b["mf_total"] >= b["bp_mean_epochs"], cg

    # ---- matched compute --------------------------------------------------- #
    matched = {}
    for cg, _, _, _ in CONFIGS:
        bp = arm(w, cg, "BP", "harmonised")
        mf = arm(w, cg, "MF", "iso_compute")
        rec = {"n_bp": int(len(bp)), "n_mf": int(len(mf)), "metrics": {}}
        for m in METRICS:
            a, b = bp[m].dropna().values, mf[m].dropna().values
            if len(a) == 0 or len(b) == 0:
                continue
            margin = ACC_MARGIN_PP if m == "test_accuracy" else RATIO_MARGIN * a.mean()
            r = tost(a, b, margin)
            r.update(bp=describe(bp[m]), mf=describe(mf[m]),
                     rel_pct=float(100 * (b.mean() - a.mean()) / a.mean()),
                     paired=paired_tost(bp, mf, m, margin))
            rec["metrics"][m] = r
        matched[cg] = rec
    out["matched_compute"] = matched

    # ---- own convergence --------------------------------------------------- #
    conv = {}
    for cg, _, _, _ in CONFIGS:
        bp = arm(w, cg, "BP", "harmonised")
        mf = arm(w, cg, "MF", "harmonised")
        rec = {"n_bp": int(len(bp)), "n_mf": int(len(mf)), "metrics": {}}
        for m in METRICS:
            a, b = bp[m].dropna().values, mf[m].dropna().values
            if len(a) == 0 or len(b) == 0:
                continue
            margin = ACC_MARGIN_PP if m == "test_accuracy" else RATIO_MARGIN * a.mean()
            r = tost(a, b, margin)
            r.update(bp=describe(bp[m]), mf=describe(mf[m]),
                     rel_pct=float(100 * (b.mean() - a.mean()) / a.mean()),
                     ratio=float(b.mean() / a.mean()))
            rec["metrics"][m] = r
        conv[cg] = rec
    out["own_convergence"] = conv

    # ---- ablation ladder --------------------------------------------------- #
    RUNGS = [("bp", "BP"), ("bp_ds", "BP_DS"), ("mf_joint", "MF_JOINT"), ("mf_recompute", "MF")]
    ladder = {}
    for cg, _, _, _ in CONFIGS:
        rec = {}
        for rung, algo in RUNGS:
            s = w[(w.config_group == cg) & (w.rung == rung) & (w.protocol == "harmonised")
                  & (w.cache_strategy == "recompute")]
            rec[rung] = {m: describe(s[m]) for m in
                         ["test_accuracy", "total_gpu_energy_wh", "peak_torch_alloc_mib",
                          "trace_mean_power_w", "training_duration_sec"]}
            rec[rung]["n"] = int(len(s))
        ladder[cg] = rec
    out["ladder"] = ladder

    # ---- caching ----------------------------------------------------------- #
    cache = {}
    for cg in ["MNIST 2x1000", "Fashion-MNIST 2x1000"]:
        base = arm(w, cg, "MF", "harmonised", "recompute")
        rec = {}
        for cs in ["recompute", "cache_device", "cache_host"]:
            s = arm(w, cg, "MF", "harmonised", cs)
            rec[cs] = {m: describe(s[m]) for m in
                       ["peak_torch_alloc_mib", "peak_process_rss_mib", "total_gpu_energy_wh",
                        "training_duration_sec", "test_accuracy"]}
            rec[cs]["n"] = int(len(s))
            for m in ["peak_torch_alloc_mib", "peak_process_rss_mib", "total_gpu_energy_wh",
                      "training_duration_sec"]:
                b0 = base[m].dropna().mean()
                rec[cs][m]["rel_pct_vs_recompute"] = float(100 * (rec[cs][m]["mean"] - b0) / b0)
        cache[cg] = rec
    out["cache"] = cache

    # ---- legacy per-epoch cross-check -------------------------------------- #
    # The phase 4 reproduction runs are the only ones carrying epochs_completed,
    # so they are the only place the per-pass cost can be formed directly rather
    # than through the matched budget. Different protocol, different runs, same
    # conclusion, which is why it is worth quoting.
    leg = w[w.protocol == "legacy"].copy()
    leg["wh_per_epoch"] = leg.total_gpu_energy_wh / leg.epochs_completed
    leg["s_per_epoch"] = leg.training_duration_sec / leg.epochs_completed
    legacy = {}
    base = leg[leg.experiment_name == "repro_bp_cifar10_mlp_3x2000"]
    for name in sorted(leg.experiment_name.unique()):
        s = leg[leg.experiment_name == name]
        rec = {m: describe(s[m]) for m in
               ["wh_per_epoch", "s_per_epoch", "epochs_completed",
                "total_gpu_energy_wh", "training_duration_sec", "test_accuracy"]}
        rec["n"] = int(len(s))
        for m in ["wh_per_epoch", "s_per_epoch", "total_gpu_energy_wh",
                  "training_duration_sec", "epochs_completed"]:
            b0 = base[m].dropna().mean()
            rec[m]["rel_pct_vs_bp"] = float(100 * (rec[m]["mean"] - b0) / b0)
        legacy[name] = rec
    out["legacy_per_epoch"] = legacy

    # ---- utilisation ------------------------------------------------------- #
    mlp = w[w.architecture.isin(["2x1000", "3x2000"])]
    util = {}
    for algo in ["BP", "BP_DS", "MF_JOINT", "MF"]:
        s = mlp[(mlp.algorithm == algo) & (mlp.protocol.isin(["harmonised", "iso_compute"]))]
        v = s["trace_mean_gpu_util_percent"].dropna()
        util[algo] = dict(n=int(len(v)), mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
    util["all_runs_max"] = float(w["trace_mean_gpu_util_percent"].max())
    util["mlp_max"] = float(mlp["trace_mean_gpu_util_percent"].dropna().max())
    out["utilisation"] = util

    # ---- FF / CaFo context ------------------------------------------------- #
    context = {}
    for cg in sorted(w.config_group.unique()):
        s = w[(w.config_group == cg) & (w.protocol == "harmonised")]
        if not set(s.algorithm) & {"FF", "CaFo"}:
            continue
        bp = s[s.algorithm == "BP"]
        if len(bp) == 0:
            continue
        rec = {"BP": {m: describe(bp[m]) for m in METRICS} | {"n": int(len(bp))}}
        for en in sorted(s[s.algorithm.isin(["FF", "CaFo"])].experiment_name.unique()):
            t = s[s.experiment_name == en]
            rec[en] = {m: describe(t[m]) for m in METRICS} | {"n": int(len(t))}
            for m in ["total_gpu_energy_wh", "training_duration_sec", "peak_torch_alloc_mib"]:
                b0 = bp[m].dropna().mean()
                rec[en][m]["ratio_vs_bp"] = float(rec[en][m]["mean"] / b0) if b0 else float("nan")
            rec[en]["test_accuracy"]["delta_vs_bp"] = float(
                rec[en]["test_accuracy"]["mean"] - bp["test_accuracy"].dropna().mean())
        context[cg] = rec
    out["ff_cafo"] = context

    # ---- pre-registered ladder contrasts ----------------------------------- #
    if LADDER_JSON.exists():
        la = json.load(open(LADDER_JSON))
        keep = {}
        for block in la:
            cfg = block["configuration"]
            for c in block["contrasts"]:
                key = f"{cfg}|{c['metric']}|{c['rung_a']}->{c['rung_b']}"
                ci = c.get("difference_ci") or {}
                keep[key] = dict(n=c["n_pairs"], mean_a=c["mean_a"], mean_b=c["mean_b"],
                                 diff=c["mean_b"] - c["mean_a"],
                                 lo=ci.get("low"), hi=ci.get("high"),
                                 p_holm=c.get("paired_p_holm"), verdict=c.get("verdict"))
        out["prereg_ladder"] = keep
    else:
        out["prereg_ladder"] = {}
        print("[warn] results/ladder_analysis.json not found; pre-registered CIs omitted")

    (HERE / "numbers.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"[write] {HERE/'numbers.json'}")
    emit_tables(out)


# --------------------------------------------------------------------------- #
# LaTeX tables
# --------------------------------------------------------------------------- #
SHORT = {"MNIST 2x1000": "MNIST 2$\\times$1000",
         "Fashion-MNIST 2x1000": "Fashion 2$\\times$1000",
         "CIFAR-10 3x2000": "CIFAR-10 3$\\times$2000",
         "CIFAR-100 3x2000": "CIFAR-100 3$\\times$2000"}


def write_rows(name: str, rows: list) -> None:
    """Write tabular rows with no trailing row break.

    A fragment that ends in \\\\ breaks the \\bottomrule that follows the
    \\input, because the optional-argument scan after \\\\ runs into the end of
    the file. The manuscript supplies the final \\\\ itself.
    """
    (TABLES / name).write_text(" \\\\\n".join(rows) + "\n")


def emit_tables(o: dict) -> None:
    # Both headline tables are transposed, configurations across the columns:
    # fifteen numeric columns do not fit the llncs text block.

    # ---- Table: matched compute ------------------------------------------- #
    def row(label, fmt, pick):
        # Math mode so a negative sign prints as a minus, not a hyphen.
        cells = " & ".join(f"${format(pick(cg), fmt)}$" for cg, _, _, _ in CONFIGS)
        return f"    {label} & {cells}"

    mc = {cg: o["matched_compute"][cg]["metrics"] for cg, _, _, _ in CONFIGS}
    L = [
        row("\\MF{} epoch budget", "d", lambda c: o["budget"][c]["mf_total"]),
        "    \\addlinespace[2pt]\n    \\multicolumn{5}{l}{\\emph{Test accuracy} (\\%)}",
        row("\\quad \\BP{}", ".2f", lambda c: mc[c]["test_accuracy"]["bp"]["mean"]),
        row("\\quad \\MF{}", ".2f", lambda c: mc[c]["test_accuracy"]["mf"]["mean"]),
        row("\\quad $\\Delta$ (pp)", "+.2f", lambda c: mc[c]["test_accuracy"]["diff"]),
        "    \\addlinespace[2pt]\n    \\multicolumn{5}{l}{\\emph{GPU energy} (\\si{\\watt\\hour})}",
        row("\\quad \\BP{}", ".3f", lambda c: mc[c]["total_gpu_energy_wh"]["bp"]["mean"]),
        row("\\quad \\MF{}", ".3f", lambda c: mc[c]["total_gpu_energy_wh"]["mf"]["mean"]),
        row("\\quad $\\Delta$ (\\%)", "+.1f", lambda c: mc[c]["total_gpu_energy_wh"]["rel_pct"]),
        "    \\addlinespace[2pt]\n    \\multicolumn{5}{l}{\\emph{Peak allocated memory} (\\si{\\mebibyte}, per process)}",
        row("\\quad \\BP{}", ".1f", lambda c: mc[c]["peak_torch_alloc_mib"]["bp"]["mean"]),
        row("\\quad \\MF{}", ".1f", lambda c: mc[c]["peak_torch_alloc_mib"]["mf"]["mean"]),
        row("\\quad $\\Delta$ (\\%)", "+.1f", lambda c: mc[c]["peak_torch_alloc_mib"]["rel_pct"]),
        "    \\addlinespace[2pt]\n    \\multicolumn{5}{l}{\\emph{Mean board power} (\\si{\\watt})}",
        row("\\quad \\BP{}", ".1f", lambda c: mc[c]["trace_mean_power_w"]["bp"]["mean"]),
        row("\\quad \\MF{}", ".1f", lambda c: mc[c]["trace_mean_power_w"]["mf"]["mean"]),
        row("\\quad $\\Delta$ (\\%)", "+.1f", lambda c: mc[c]["trace_mean_power_w"]["rel_pct"]),
        "    \\addlinespace[2pt]\n"
        + row("\\emph{Wall-clock time} $\\Delta$ (\\%)", "+.1f",
              lambda c: mc[c]["training_duration_sec"]["rel_pct"]),
    ]
    write_rows("matched_compute_body.tex", L)

    # ---- Table: ladder ----------------------------------------------------- #
    lad = o["ladder"]
    RUNGS = [("bp", "\\BP{}"), ("bp_ds", "\\BPDS{}"),
             ("mf_joint", "\\MFJ{}"), ("mf_recompute", "\\MF{}")]
    # The first block heading lives in the manuscript, because a fragment may
    # not open with \multicolumn: it lands on the file boundary, not on a \cr.
    L = []
    for rung, lab in RUNGS:
        L.append(row("\\quad " + lab, "+.2f",
                     lambda c, r=rung: lad[c][r]["test_accuracy"]["mean"]
                     - lad[c]["bp"]["test_accuracy"]["mean"]))
    L.append("    \\addlinespace[2pt]\n    \\multicolumn{5}{l}{\\emph{GPU energy} (\\% of \\BP{})}")
    for rung, lab in RUNGS:
        L.append(row("\\quad " + lab, ".0f",
                     lambda c, r=rung: 100 * lad[c][r]["total_gpu_energy_wh"]["mean"]
                     / lad[c]["bp"]["total_gpu_energy_wh"]["mean"]))
    for cg, _, _, _ in CONFIGS:          # the caption states the seed range
        assert min(lad[cg][r]["n"] for r, _ in RUNGS) >= 20, cg
    write_rows("ladder_body.tex", L)

    # ---- Table: own convergence -------------------------------------------- #
    L = []
    for cg, _, _, _ in CONFIGS:
        r = o["own_convergence"][cg]["metrics"]
        acc, en, tm = r["test_accuracy"], r["total_gpu_energy_wh"], r["training_duration_sec"]
        L.append(f"    {SHORT[cg]} & {acc['bp']['mean']:.2f} & {acc['mf']['mean']:.2f} "
                 f"& {acc['diff']:+.2f} & {tm['ratio']:.2f} & {en['ratio']:.2f}")
    write_rows("convergence_body.tex", L)

    print(f"[write] {TABLES}/*.tex")


if __name__ == "__main__":
    main()
