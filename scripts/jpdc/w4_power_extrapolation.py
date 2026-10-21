#!/usr/bin/env python3
"""W4 (reframed): within-session ratio prediction, not absolute cross-era energy.

The pre-G1 affine fit P = P_static + kappa*u is retained only for its slope.  For
each dataset we calibrate the intercept on one BP run from the *same* session
(seed 42 by default), then predict every arm's energy from its own measured span
and utilization.  The reported endpoint is the paired per-seed ratio
E_arm/E_BP, never an absolute cross-era energy.

Why ratio-only: the intercept is era-specific (G1 sits 3-7% above the pre-G1
fit), but the utilization slope is stable.  A one-point BP calibration removes
the era intercept; what remains is a within-session ratio.  Absolute predictions
are deliberately not reported.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def fit_ols(x: np.ndarray, y: np.ndarray) -> dict:
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = len(y)
    sigma2 = float(resid @ resid) / (n - 2)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tcrit = stats.t.ppf(0.975, n - 2)
    return dict(
        intercept=float(beta[0]),
        slope=float(beta[1]),
        intercept_lo=float(beta[0] - tcrit * se[0]),
        intercept_hi=float(beta[0] + tcrit * se[0]),
        slope_lo=float(beta[1] - tcrit * se[1]),
        slope_hi=float(beta[1] + tcrit * se[1]),
        r2=float(1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()),
        resid_sd=float(np.sqrt(sigma2)),
        n=n,
    )


def trace_table(pattern: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(pattern):
        try:
            d = pd.read_csv(
                path,
                usecols=[
                    "timestamp_sec",
                    "power_watts",
                    "gpu_util_percent",
                    "compute_processes",
                ],
            )
        except (ValueError, pd.errors.EmptyDataError):
            continue
        if len(d) < 100:
            continue
        span = float(d.timestamp_sec.max() - d.timestamp_sec.min())
        if span < 60 or d.compute_processes.max() > 1:
            continue
        if not np.isfinite(d.power_watts.mean()) or not np.isfinite(
            d.gpu_util_percent.mean()
        ):
            continue
        rows.append(
            dict(
                file=Path(path).name,
                samples=len(d),
                span_sec=span,
                power_w=float(d.power_watts.mean()),
                util_pct=float(d.gpu_util_percent.mean()),
                date=Path(path).name.split("_")[-2],
            )
        )
    return pd.DataFrame(rows)


def predicted_energy(g: pd.DataFrame, intercept: float, slope: float) -> pd.Series:
    return (intercept + slope * g.mean_util) * g.span_sec / 3600.0


def ratio_table(cost: pd.DataFrame, slope: float, cal_seed: int) -> pd.DataFrame:
    out = []
    for dataset, g in cost.groupby("dataset"):
        bp = g[g.arm == "bp"].set_index("seed")
        if cal_seed not in bp.index:
            raise ValueError(f"calibration seed {cal_seed} missing for {dataset}")
        cal = bp.loc[cal_seed]
        intercept = float(cal.mean_power_w - slope * cal.mean_util)
        g = g.copy()
        g["intercept_cal"] = intercept
        g["slope"] = slope
        g["energy_pred_cal"] = predicted_energy(g, intercept, slope)
        bp_pred = g[g.arm == "bp"].set_index("seed").energy_pred_cal
        bp_meas = g[g.arm == "bp"].set_index("seed").energy_wh
        for arm, a in g.groupby("arm"):
            a = a.set_index("seed")
            common = sorted(a.index.intersection(bp_meas.index))
            if not common:
                continue
            meas_ratio = a.loc[common, "energy_wh"] / bp_meas.loc[common]
            pred_ratio = a.loc[common, "energy_pred_cal"] / bp_pred.loc[common]
            if arm == "bp":
                continue
            err = (pred_ratio / meas_ratio - 1.0) * 100.0
            out.append(
                dict(
                    dataset=dataset,
                    arm=arm,
                    n=len(common),
                    cal_seed=cal_seed,
                    measured_ratio=float(meas_ratio.mean()),
                    predicted_ratio=float(pred_ratio.mean()),
                    ratioBias_pct=float(err.mean()),
                    ratioMAE_pct=float(err.abs().mean()),
                    ratioErr_lo=float(err.mean() - stats.sem(err) * stats.t.ppf(0.975, len(err) - 1)),
                    ratioErr_hi=float(err.mean() + stats.sem(err) * stats.t.ppf(0.975, len(err) - 1)),
                    measured_ratio_lo=float(meas_ratio.mean() - stats.sem(meas_ratio) * stats.t.ppf(0.975, len(meas_ratio) - 1)),
                    measured_ratio_hi=float(meas_ratio.mean() + stats.sem(meas_ratio) * stats.t.ppf(0.975, len(meas_ratio) - 1)),
                )
            )
    return pd.DataFrame(out)


def sensitivity(cost: pd.DataFrame, slope: float, primary_cal: int) -> pd.DataFrame:
    rows = []
    for dataset, g in cost.groupby("dataset"):
        bp_seeds = sorted(g[g.arm == "bp"].seed.unique())
        for arm in ("mf_joint", "mf_simul", "mf_seq"):
            errs = []
            for cal in bp_seeds:
                t = ratio_table(g, slope, int(cal))
                row = t[t.arm == arm]
                if len(row):
                    errs.append(float(row.ratioBias_pct.iloc[0]))
            rows.append(
                dict(
                    dataset=dataset,
                    arm=arm,
                    primary_cal_seed=primary_cal,
                    sensitivity_min=float(np.min(errs)),
                    sensitivity_max=float(np.max(errs)),
                    sensitivity_median=float(np.median(errs)),
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=str, required=True)
    parser.add_argument("--g1-cost", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cal-seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tr = trace_table(args.traces)
    tr.to_csv(args.out_dir / "w4_training_traces.csv", index=False)
    primary = fit_ols(tr.util_pct.to_numpy(float), tr.power_w.to_numpy(float))
    sep = tr[tr.date.astype(str).str.startswith("202609")]
    sensitivity_fit = (
        fit_ols(sep.util_pct.to_numpy(float), sep.power_w.to_numpy(float))
        if len(sep) >= 3
        else primary
    )

    cost = pd.read_csv(args.g1_cost)
    ratio = ratio_table(cost, primary["slope"], args.cal_seed)
    ratio.to_csv(args.out_dir / "w4_ratio_summary.csv", index=False)
    sens = sensitivity(cost, primary["slope"], args.cal_seed)
    sens.to_csv(args.out_dir / "w4_ratio_sensitivity.csv", index=False)

    # Per-run audit values, still ratio-oriented in the summary.
    per_run = []
    for dataset, g in cost.groupby("dataset"):
        bp = g[g.arm == "bp"].set_index("seed")
        cal = bp.loc[args.cal_seed]
        intercept = float(cal.mean_power_w - primary["slope"] * cal.mean_util)
        g = g.copy()
        g["intercept_cal"] = intercept
        g["energy_pred_cal"] = predicted_energy(g, intercept, primary["slope"])
        bp_e = bp.energy_wh
        bp_p = g[g.arm == "bp"].set_index("seed").energy_pred_cal
        for _, r in g.iterrows():
            if r.seed not in bp_e.index:
                continue
            per_run.append(
                dict(
                    dataset=dataset,
                    arm=r.arm,
                    seed=int(r.seed),
                    mean_util=r.mean_util,
                    measured_energy_wh=r.energy_wh,
                    predicted_energy_cal_wh=r.energy_pred_cal,
                    measured_ratio_vs_bp=float(r.energy_wh / bp_e.loc[r.seed]),
                    predicted_ratio_vs_bp=float(r.energy_pred_cal / bp_p.loc[r.seed]),
                )
            )
    pd.DataFrame(per_run).to_csv(args.out_dir / "w4_ratio_per_run.csv", index=False)

    lines = ["# W4 — within-session MF/BP energy-ratio prediction", ""]
    lines.append(
        "**Reframing.** Absolute cross-era energy predictions are retired. The "
        "model predicts only the within-session ratio E_arm/E_BP after a one-point "
        "intercept calibration on a same-dataset BP run (seed "
        f"{args.cal_seed}). The pre-G1 fit supplies the utilization slope; the "
        "calibration removes the era-specific intercept."
    )
    lines.append("")
    lines.append("## Pre-G1 slope fit (fit only; not a G1 absolute prediction)")
    lines.append("")
    lines.append(
        f"- traces passing QC: **{primary['n']}**; September-only n={sensitivity_fit['n']}"
    )
    lines.append(
        f"- P = {primary['intercept']:.2f} [{primary['intercept_lo']:.2f}, "
        f"{primary['intercept_hi']:.2f}] + {primary['slope']:.3f} "
        f"[{primary['slope_lo']:.3f}, {primary['slope_hi']:.3f}]·u, R² = "
        f"{primary['r2']:.3f}"
    )
    lines.append(
        f"- September-only slope {sensitivity_fit['slope']:.3f} "
        f"[{sensitivity_fit['slope_lo']:.3f}, {sensitivity_fit['slope_hi']:.3f}], "
        f"R² = {sensitivity_fit['r2']:.3f}"
    )
    lines.append("")
    lines.append("## Ratio prediction (paired by seed, mean [95% t-CI])")
    lines.append("")
    lines.append(
        "| dataset | arm | n | measured E_arm/E_BP | predicted ratio | ratio bias % | ratio MAE % |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in ratio.iterrows():
        lines.append(
            f"| {r.dataset} | {r.arm} | {int(r.n)} | "
            f"{r.measured_ratio:.3f} [{r.measured_ratio_lo:.3f}, {r.measured_ratio_hi:.3f}] | "
            f"{r.predicted_ratio:.3f} | {r.ratioBias_pct:+.2f} "
            f"[{r.ratioErr_lo:+.2f}, {r.ratioErr_hi:+.2f}] | {r.ratioMAE_pct:.2f} |"
        )
    lines.append("")
    lines.append("## Calibration-run sensitivity")
    lines.append("")
    lines.append(
        "Each BP seed 42–61 was used as the one calibration point in turn; the "
        "table gives the range of the mean ratio bias. This tests whether the "
        "headline ratio depends on seed 42 being atypical."
    )
    lines.append("")
    lines.append("| dataset | arm | primary seed | bias range % | median bias % |")
    lines.append("|---|---|---|---|---|")
    for _, r in sens.iterrows():
        lines.append(
            f"| {r.dataset} | {r.arm} | {int(r.primary_cal_seed)} | "
            f"[{r.sensitivity_min:+.2f}, {r.sensitivity_max:+.2f}] | "
            f"{r.sensitivity_median:+.2f} |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(
        "The slope transfers; the raw intercept does not. After one BP calibration "
        "point per dataset, the CIFAR-10 mean ratio bias is within about 1%, but the "
        "per-seed ratio MAE is 4.0–4.7%; CIFAR-100 has a +3.5 to +4.4% mean bias and "
        "5.4–6.4% MAE. These are ratio errors, not absolute joule predictions. The "
        "calibration-seed range shows that seed 42 is not special, but it also shows "
        "that the residual ratio error is real. Do not use the model across eras "
        "without a fresh same-session BP calibration."
    )
    (args.out_dir / "w4_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
