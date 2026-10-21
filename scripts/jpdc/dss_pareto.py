#!/usr/bin/env python3
"""DSS figure data: Pareto tables for the four W1 arms, Protocol A and B.

Protocol A (own convergence) is the decisive G1 set: BP, MF-Joint, MF-Simul
and MF-Seq, seeds 42-61, energy from each run's own trace, memory from the
per-process allocator peak.

Protocol B (matched data passes) exists only for the PPAM-era BP reference and
the staged-MF iso-compute diagnostic. MF-Joint and MF-Simul were never run
under Protocol B; they are emitted as explicit "not measured" rows, never
imputed. BP is not rerun for Protocol B (it keeps its own stopping); MF gets
ceil(BP mean epochs / 4) epochs per stage, i.e. 52 passes on CIFAR-10 and 44 on
CIFAR-100 (13/11 per stage x 4 stages), read from the diagnostic configs.

Absolute energy is not comparable across the two protocol tables: Protocol A is
the 2026-10 G1 era, Protocol B the 2026-09 PPAM era (W0e puts the era drift at
-2% to +8% depending on arm). Compare arms within a protocol, not across.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

SEEDS = list(range(42, 62))
ARMS_A = ["bp", "mf_joint", "mf_simul", "mf_seq"]
ARMS_B = ["bp", "mf_joint", "mf_simul", "mf_seq"]
METRICS = [
    ("test_accuracy", "acc %"),
    ("passes", "passes"),
    ("energy_wh", "energy Wh"),
    ("mean_power_w", "power W"),
    ("peak_torch_alloc_mib", "alloc MiB"),
]
TIDY_METRICS = [
    "test_accuracy",
    "total_gpu_energy_wh",
    "trace_mean_power_w",
    "peak_torch_alloc_mib",
]


def ci(values: np.ndarray) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)
    return float(values.mean() - half), float(values.mean() + half)


def aggregate(
    df: pd.DataFrame, protocol: str, dataset: str, arm: str, era: str
) -> dict:
    row = {
        "protocol": protocol,
        "dataset": dataset,
        "arm": arm,
        "era": era,
        "n": int(len(df)),
        "status": "measured",
    }
    for metric, _ in METRICS:
        vals = df[metric].to_numpy(float) if metric in df else np.array([])
        lo, hi = ci(vals)
        row[f"{metric}_mean"] = float(np.nanmean(vals)) if len(vals) else float("nan")
        row[f"{metric}_lo"] = lo
        row[f"{metric}_hi"] = hi
    return row


def pareto_flags(rows: list[dict]) -> None:
    """Mark arms not dominated on (accuracy high, energy low) within a table."""
    measured = [r for r in rows if r["status"] == "measured"]
    for r in measured:
        dominated = False
        for other in measured:
            if other is r:
                continue
            if (
                other["test_accuracy_mean"] >= r["test_accuracy_mean"]
                and other["energy_wh_mean"] <= r["energy_wh_mean"]
                and (
                    other["test_accuracy_mean"] > r["test_accuracy_mean"]
                    or other["energy_wh_mean"] < r["energy_wh_mean"]
                )
            ):
                dominated = True
                break
        r["pareto"] = not dominated
    for r in rows:
        r.setdefault("pareto", False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-per-seed", type=Path, required=True)
    parser.add_argument("--w0a-per-seed", type=Path, required=True)
    parser.add_argument("--w0e-per-seed", type=Path, required=True)
    parser.add_argument("--tidy", type=Path, required=True)
    parser.add_argument("--eq-configs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    g1 = pd.read_csv(args.g1_per_seed)
    w0a = pd.read_csv(args.w0a_per_seed)
    w0e = pd.read_csv(args.w0e_per_seed)
    tidy = pd.read_csv(args.tidy)

    # ---------- Protocol A ----------
    a_df = g1.merge(
        w0a[["arm", "dataset", "seed", "mean_power_w", "peak_torch_alloc_mib"]],
        on=["arm", "dataset", "seed"],
        how="inner",
        validate="one_to_one",
    )
    a_df = a_df[a_df.seed.isin(SEEDS)]

    rows_a = []
    for dataset in ("cifar10", "cifar100"):
        for arm in ARMS_A:
            sub = a_df[(a_df.arm == arm) & (a_df.dataset == dataset)]
            rows_a.append(aggregate(sub, "A", dataset, arm, "G1 2026-10"))
        pareto_flags(rows_a[-4:])

    # ---------- Protocol B ----------
    # BP reference: PPAM-era canonical harmonised runs.
    # MF-Seq: iso-compute diagnostic (equal epochs per stage).
    b_rows = []
    for dataset, label in (("cifar10", "CIFAR10"), ("cifar100", "CIFAR100")):
        bp_exp = f"bp_{dataset}_mlp_3x2000"
        bp = tidy[
            (tidy.experiment_name == bp_exp)
            & (tidy.protocol == "harmonised")
            & (tidy.phase == "phase3_ladder")
            & (tidy.metric.isin(TIDY_METRICS))
        ]
        bp_wide = bp.pivot_table(
            index="seed", columns="metric", values="value", aggfunc="mean"
        ).reset_index()
        bp_pass = w0e[
            (w0e.era == "ppam") & (w0e.arm == "bp") & (w0e.dataset == dataset)
        ][["seed", "passes"]]
        bp_wide = bp_wide.merge(bp_pass, on="seed", how="inner")
        bp_wide = bp_wide.rename(
            columns={
                "total_gpu_energy_wh": "energy_wh",
                "trace_mean_power_w": "mean_power_w",
                "peak_torch_alloc_mib": "peak_torch_alloc_mib",
                "test_accuracy": "test_accuracy",
            }
        )
        b_rows.append(aggregate(bp_wide, "B", dataset, "bp", "PPAM 2026-09"))

        eq_exp = f"mf_{dataset}_mlp_3x2000_equal_epochs"
        mf = tidy[
            (tidy.experiment_name == eq_exp)
            & (tidy.protocol == "iso_compute")
            & (tidy.metric.isin(TIDY_METRICS))
        ]
        mf_wide = mf.pivot_table(
            index="seed", columns="metric", values="value", aggfunc="mean"
        ).reset_index()
        cfg = yaml.safe_load(
            (args.eq_configs / f"{dataset}_mlp_3x2000_equal_epochs.yaml").read_text()
        )
        per_stage = int(cfg["early_stopping"]["max_epochs"])
        mf_wide["passes"] = float(per_stage * 4)
        mf_wide = mf_wide.rename(
            columns={
                "total_gpu_energy_wh": "energy_wh",
                "trace_mean_power_w": "mean_power_w",
                "peak_torch_alloc_mib": "peak_torch_alloc_mib",
                "test_accuracy": "test_accuracy",
            }
        )
        b_rows.append(aggregate(mf_wide, "B", dataset, "mf_seq", "PPAM 2026-09"))

    # explicit not-measured rows for the two arms that never ran Protocol B
    measured_b = {(r["dataset"], r["arm"]) for r in b_rows}
    for dataset in ("cifar10", "cifar100"):
        for arm in ARMS_B:
            if (dataset, arm) not in measured_b:
                row = {
                    "protocol": "B",
                    "dataset": dataset,
                    "arm": arm,
                    "era": "not run",
                    "n": 0,
                    "status": "not measured",
                    "pareto": False,
                }
                for metric, _ in METRICS:
                    row[f"{metric}_mean"] = float("nan")
                    row[f"{metric}_lo"] = float("nan")
                    row[f"{metric}_hi"] = float("nan")
                b_rows.append(row)

    for dataset in ("cifar10", "cifar100"):
        pareto_flags(
            [r for r in b_rows if r["dataset"] == dataset and r["status"] == "measured"]
        )

    cols = [
        "protocol",
        "dataset",
        "arm",
        "era",
        "n",
        "status",
        "pareto",
    ] + [f"{m}_{s}" for m, _ in METRICS for s in ("mean", "lo", "hi")]
    df_a = pd.DataFrame(rows_a)[cols]
    df_b = pd.DataFrame(b_rows)[cols]
    df_a.to_csv(args.out_dir / "pareto_protocol_a.csv", index=False)
    df_b.to_csv(args.out_dir / "pareto_protocol_b.csv", index=False)

    # ---------- markdown ----------
    lines = ["# DSS Pareto data — Protocol A and Protocol B", ""]
    lines.append(
        "Generated by `scripts/jpdc/dss_pareto.py`. Protocol A: decisive G1 "
        "(2026-10, seeds 42-61), energy from each run's own NVML trace, memory = "
        "`torch.cuda.max_memory_allocated` (per-process, excludes CUDA context). "
        "Protocol B: PPAM-era (2026-09) BP reference and staged-MF matched-pass "
        "diagnostic. MF-Joint/MF-Simul were never run under Protocol B."
    )
    lines.append("")
    for protocol, df, title in (
        ("A", df_a, "Protocol A — own convergence (G1)"),
        ("B", df_b, "Protocol B — matched data passes (PPAM era)"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        for dataset in ("cifar10", "cifar100"):
            sub = df[df.dataset == dataset]
            lines.append(f"### {dataset}")
            lines.append("")
            lines.append(
                "| arm | n | accuracy % | passes | energy Wh | power W | alloc MiB | Pareto |"
            )
            lines.append("|---|---|---|---|---|---|---|---|")
            for _, r in sub.iterrows():
                if r.status != "measured":
                    lines.append(
                        f"| {r.arm} | — | not measured under Protocol B | — | — | — | — | — |"
                    )
                    continue
                lines.append(
                    f"| {r.arm} | {r.n} | "
                    f"{r.test_accuracy_mean:.2f} [{r.test_accuracy_lo:.2f}, {r.test_accuracy_hi:.2f}] | "
                    f"{r.passes_mean:.1f} [{r.passes_lo:.1f}, {r.passes_hi:.1f}] | "
                    f"{r.energy_wh_mean:.2f} [{r.energy_wh_lo:.2f}, {r.energy_wh_hi:.2f}] | "
                    f"{r.mean_power_w_mean:.1f} [{r.mean_power_w_lo:.1f}, {r.mean_power_w_hi:.1f}] | "
                    f"{r.peak_torch_alloc_mib_mean:.0f} "
                    f"[{r.peak_torch_alloc_mib_lo:.0f}, {r.peak_torch_alloc_mib_hi:.0f}] | "
                    f"{'yes' if r.pareto else 'no'} |"
                )
            lines.append("")
    lines.append("## Notes for the figure")
    lines.append("")
    lines.append(
        "- Protocol A numbers are contemporaneous across all four arms, so the "
        "Pareto frontier is internally valid."
    )
    lines.append(
        "- Protocol B has only BP and MF-Seq. The missing MF-Joint/MF-Simul cells "
        "are a data gap, not a null result; filling them needs 2 x 2 x 20 new "
        "runs under the 52/44-pass budget."
    )
    lines.append(
        "- Do not plot Protocol A and B on one absolute-energy axis: W0e measures "
        "era drift of −2.2% (BP CIFAR-10) to +8.3% (BP CIFAR-100), with MF-Seq "
        "at +5.0%/+6.9%. Normalise within protocol or annotate the era."
    )
    lines.append(
        "- Protocol B BP is the same PPAM-era run as the paper's reference; it is "
        "not rerun with a matched budget because B gives MF at least BP's passes."
    )
    (args.out_dir / "dss_pareto_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
