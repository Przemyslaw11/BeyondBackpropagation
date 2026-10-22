#!/usr/bin/env python3
"""W2 link-rate model: time and energy to each arm's measured final accuracy.

This replaces the old four-GPU layer-parallel W2 design.  It is a zero-GPU
accounting model built from first-party G1 artifacts:

  * measured passes and final test accuracy: ``jpdc_data/g1/g1_per_seed.csv``;
  * measured training duration: each run's own JSON, ``training_duration_sec``;
  * measured GPU energy: each run's own NVML trace via the G1 CSV;
  * cut-byte bill: ``w3_cut_per_seed.csv`` (useful cut bytes, not memory
    traffic, not the dead M0 forward);
  * per-step compute time: ``training_duration_sec / (passes * 352)``.

The model is deliberately simple and stated as such: compute and cut
communication are serialized, so

    time_to_final_accuracy = compute_time + cut_bytes / link_rate
    energy_to_final_accuracy = measured_gpu_energy
                               + idle_gpu_power * communication_time
                               + NIC energy

The idle power is the W4 intercept (51.97 W); NIC energy defaults to 1 pJ/bit
and an optional port power.  The target is each arm's **own measured final test
accuracy**, because G1 did not log per-epoch test accuracy.  This is therefore a
time/energy-to-final-accuracy frontier, not a common-threshold crossing.

Seq-Global has not run in G1.  Until W1 produces it, the script emits an
explicit planning proxy: MF-Seq passes and accuracy, BP per-pass compute time
and energy, and the analytic Seq-Global cut bill (forward+backward across the
trainable prefix: W2 0.72 GB/pass, W3 1.44 GB/pass, W1 0).  Proxy rows are
marked ``source`` and must not be read as measurements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

SEEDS = list(range(42, 62))
STEPS_PER_PASS = 352  # ceil(45,000 / 128), matching W3
GB = 1e9
MEASURED = {
    "bp": "bp_{dataset}_mlp_3x2000",
    "mf_simul": "mf_simul_{dataset}_mlp_3x2000",
    "mf_seq": "mf_{dataset}_mlp_3x2000",
}
SEQ_GLOBAL_PROXY = "mf_global_{dataset}_mlp_3x2000"
PREFIX_RATE_GB = {1: 0.0, 2: 0.72, 3: 1.44}


def ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)
    return float(values.mean() - half), float(values.mean() + half)


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI for a paired mean ratio."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def run_json_path(runs_dir: Path, experiment: str, seed: int) -> Path:
    return runs_dir / "runs" / experiment / f"{experiment}_seed{seed}.json"


def training_seconds(runs_dir: Path, experiment: str, seed: int, fallback: float) -> float:
    path = run_json_path(runs_dir, experiment, seed)
    if not path.exists():
        return float(fallback)
    try:
        record = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return float(fallback)
    value = record.get("training_duration_sec")
    return float(value) if value is not None else float(fallback)


def cut_bytes_for_measured(
    cut_per_seed: pd.DataFrame, dataset: str, arm: str, seed: int
) -> float:
    sub = cut_per_seed[
        (cut_per_seed.dataset == dataset)
        & (cut_per_seed.arm == arm)
        & (cut_per_seed.seed == seed)
    ]
    if sub.empty:
        return float("nan")
    return float(sub.gb.sum())


def seq_global_proxy_cut_bytes(
    anatomy: pd.DataFrame, dataset: str, seed: int
) -> float:
    row = anatomy[
        (anatomy.experiment == f"mf_{dataset}_mlp_3x2000")
        & (anatomy.seed == seed)
    ]
    if row.empty:
        return float("nan")
    w2 = float(row.w2_epochs.iloc[0])
    w3 = float(row.w3_epochs.iloc[0])
    return PREFIX_RATE_GB[2] * w2 + PREFIX_RATE_GB[3] * w3


def communication_energy_wh(
    cut_gb: float, comm_sec: float, idle_power_w: float, nic_power_w: float,
    nic_pj_per_bit: float,
) -> float:
    bits = max(cut_gb, 0.0) * 8.0 * GB
    joules = idle_power_w * comm_sec + nic_power_w * comm_sec
    joules += nic_pj_per_bit * 1e-12 * bits
    return joules / 3600.0


def build_rows(
    g1: pd.DataFrame,
    runs_dir: Path,
    cut_per_seed: pd.DataFrame,
    anatomy: pd.DataFrame,
    links_mbps: List[float],
    idle_power_w: float,
    nic_power_w: float,
    nic_pj_per_bit: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in ("cifar10", "cifar100"):
        sub = g1[g1.dataset == dataset]
        for seed in SEEDS:
            measured: Dict[str, Dict[str, float]] = {}
            for arm, pattern in MEASURED.items():
                row = sub[(sub.arm == arm) & (sub.seed == seed)]
                if row.empty:
                    continue
                experiment = pattern.format(dataset=dataset)
                measured[arm] = {
                    "accuracy": float(row.test_accuracy.iloc[0]),
                    "passes": float(row.passes.iloc[0]),
                    "energy_wh": float(row.energy_wh.iloc[0]),
                    "compute_sec": training_seconds(
                        runs_dir, experiment, seed, float(row.duration_sec.iloc[0])
                    ),
                    "cut_gb": cut_bytes_for_measured(cut_per_seed, dataset, arm, seed),
                    "source": "measured G1 JSON/trace + W3 cut bill",
                }
            if "bp" not in measured or "mf_seq" not in measured:
                continue

            # Seq-Global planning proxy: MF-Seq passes/accuracy, BP per-pass
            # compute/energy, analytic trainable-prefix cut bill.
            bp = measured["bp"]
            seq = measured["mf_seq"]
            bp_per_pass_sec = bp["compute_sec"] / bp["passes"]
            bp_per_pass_wh = bp["energy_wh"] / bp["passes"]
            proxy_cut = seq_global_proxy_cut_bytes(anatomy, dataset, seed)
            measured["seq_global"] = {
                "accuracy": seq["accuracy"],
                "passes": seq["passes"],
                "energy_wh": bp_per_pass_wh * seq["passes"],
                "compute_sec": bp_per_pass_sec * seq["passes"],
                "cut_gb": proxy_cut,
                "source": (
                    "PLANNING PROXY: MF-Seq passes/accuracy, BP per-pass "
                    "compute/energy, Seq-Global analytic cuts"
                ),
            }

            for arm, values in measured.items():
                cut_gb = values["cut_gb"]
                if np.isnan(cut_gb):
                    continue
                compute_sec = values["compute_sec"]
                compute_wh = values["energy_wh"]
                passes = values["passes"]
                for link_mbps in links_mbps:
                    comm_sec = cut_gb * 8.0 * GB / (link_mbps * 1e6)
                    comm_wh = communication_energy_wh(
                        cut_gb, comm_sec, idle_power_w, nic_power_w, nic_pj_per_bit
                    )
                    total_sec = compute_sec + comm_sec
                    total_wh = compute_wh + comm_wh
                    rows.append(
                        {
                            "dataset": dataset,
                            "arm": arm,
                            "seed": seed,
                            "link_mbps": float(link_mbps),
                            "target_test_accuracy": values["accuracy"],
                            "passes": passes,
                            "cut_gb_to_target": cut_gb,
                            "compute_sec_per_pass": compute_sec / passes,
                            "compute_sec_per_step": compute_sec
                            / (passes * STEPS_PER_PASS),
                            "compute_sec": compute_sec,
                            "communication_sec": comm_sec,
                            "total_sec": total_sec,
                            "compute_energy_wh": compute_wh,
                            "communication_energy_wh": comm_wh,
                            "total_energy_wh": total_wh,
                            "communication_time_share": comm_sec / total_sec,
                            "communication_energy_share": comm_wh / total_wh,
                            "source": values["source"],
                        }
                    )
    return pd.DataFrame(rows)


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, arm, link), sub in per_seed.groupby(
        ["dataset", "arm", "link_mbps"], sort=True
    ):
        baseline = per_seed[
            (per_seed.arm == "bp")
            & (per_seed.dataset == dataset)
            & (per_seed.link_mbps == link)
        ].set_index("seed")[["total_sec", "total_energy_wh"]]
        paired = sub.set_index("seed")[["total_sec", "total_energy_wh"]].join(
            baseline,
            rsuffix="_bp",
            how="inner",
        )
        record: Dict[str, object] = {
            "dataset": dataset,
            "arm": arm,
            "link_mbps": link,
            "n": len(sub),
            "target_test_accuracy_mean": float(sub.target_test_accuracy.mean()),
            "target_test_accuracy_median": float(sub.target_test_accuracy.median()),
            "passes_mean": float(sub.passes.mean()),
            "passes_median": float(sub.passes.median()),
            "cut_gb_mean": float(sub.cut_gb_to_target.mean()),
            "compute_sec_mean": float(sub.compute_sec.mean()),
            "communication_sec_mean": float(sub.communication_sec.mean()),
            "total_sec_mean": float(sub.total_sec.mean()),
            "compute_energy_wh_mean": float(sub.compute_energy_wh.mean()),
            "communication_energy_wh_mean": float(sub.communication_energy_wh.mean()),
            "total_energy_wh_mean": float(sub.total_energy_wh.mean()),
            "source": str(sub.source.iloc[0]),
        }
        for metric in (
            "total_sec",
            "total_energy_wh",
            "communication_sec",
            "communication_energy_wh",
            "cut_gb_to_target",
        ):
            lo, hi = ci95(sub[metric].to_numpy(float))
            record[f"{metric}_lo"] = lo
            record[f"{metric}_hi"] = hi
        for metric in ("total_sec", "total_energy_wh"):
            ratio_name = (
                "total_sec" if metric == "total_sec" else "total_energy"
            )
            ratios = (paired[metric] / paired[f"{metric}_bp"]).to_numpy(float)
            lo, hi = bootstrap_ci(ratios)
            record[f"{ratio_name}_ratio_vs_bp_mean"] = float(np.mean(ratios))
            record[f"{ratio_name}_ratio_vs_bp_median"] = float(np.median(ratios))
            record[f"{ratio_name}_ratio_vs_bp_lo"] = lo
            record[f"{ratio_name}_ratio_vs_bp_hi"] = hi
        rows.append(record)
    summary = pd.DataFrame(rows)
    baseline = (
        summary[summary.arm == "bp"]
        .set_index(["dataset", "link_mbps"])[["total_sec_mean", "total_energy_wh_mean"]]
        .rename(columns={"total_sec_mean": "bp_sec", "total_energy_wh_mean": "bp_wh"})
    )
    summary = summary.join(baseline, on=["dataset", "link_mbps"])
    # Backwards-compatible columns are the paired mean ratios, not ratios of
    # separately averaged totals.
    summary["total_sec_ratio_vs_bp"] = summary["total_sec_ratio_vs_bp_mean"]
    summary["total_energy_ratio_vs_bp"] = summary["total_energy_ratio_vs_bp_mean"]
    return summary


def write_markdown(
    summary: pd.DataFrame, per_seed: pd.DataFrame, args: argparse.Namespace
) -> str:
    lines = [
        "# W2 link-rate model — time and energy to measured final accuracy",
        "",
        "**Zero-GPU model, not a four-GPU training experiment.** The old W2 "
        "layer-parallel design is superseded (see `PREREG-W2-W3.md` §5, "
        "2026-10-22).",
        "",
        "## Model",
        "",
        f"- Measured compute time and energy from each G1 run's own JSON/NVML "
        f"trace; `{STEPS_PER_PASS}` optimizer steps per pass (45,000 / 128).",
        "- Cut bytes from `w3_cut_per_seed.csv` (useful cut traffic only; the "
        "dead M0 forward is excluded).",
        "- Compute and communication are serialized: `total = compute + bytes/link`.",
        f"- Communication energy uses `P_idle={args.idle_power_w:.2f} W` (W4 "
        f"intercept), `P_nic={args.nic_power_w:.2f} W`, and "
        f"`{args.nic_pj_per_bit:.2f} pJ/bit`; the pJ/bit term is negligible at "
        "these byte counts, the idle term is not.",
        "- Target is each arm's own measured final test accuracy. G1 has no "
        "per-epoch test accuracy, so this is not a common-threshold crossing.",
        "- **Seq-Global is a planning proxy**: MF-Seq passes/accuracy, BP "
        "per-pass compute/energy, and the analytic Seq-Global prefix cut bill. "
        "Replace with W1 measurements when they exist.",
        "",
        "## Link speeds",
        "",
        "| dataset | link | arm | n | target acc % | passes | cut GB | compute s | comm s | total s | compute Wh | comm Wh | total Wh | time/BP [95% CI] | energy/BP [95% CI] |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        link = f"{row.link_mbps:g} Mbit/s"
        lines.append(
            f"| {row.dataset} | {link} | {row.arm} | {int(row.n)} | "
            f"{row.target_test_accuracy_mean:.2f} | {row.passes_mean:.1f} | "
            f"{row.cut_gb_mean:.1f} | {row.compute_sec_mean:.0f} | "
            f"{row.communication_sec_mean:.0f} | {row.total_sec_mean:.0f} | "
            f"{row.compute_energy_wh_mean:.2f} | "
            f"{row.communication_energy_wh_mean:.2f} | "
            f"{row.total_energy_wh_mean:.2f} | "
            f"{row.total_sec_ratio_vs_bp_mean:.2f}× "
            f"[{row.total_sec_ratio_vs_bp_lo:.2f}, {row.total_sec_ratio_vs_bp_hi:.2f}] | "
            f"{row.total_energy_ratio_vs_bp_mean:.2f}× "
            f"[{row.total_energy_ratio_vs_bp_lo:.2f}, {row.total_energy_ratio_vs_bp_hi:.2f}] |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "At 100 Mbit/s the cut-byte bill dominates: BP moves ~2× MF-Simul's "
        "bytes and ~4× MF-Seq's useful bytes, so the slow-link total-time "
        "ordering can reverse relative to the compute-only ordering. At "
        "10 Gbit/s compute dominates and the measured G1 pass/energy ordering "
        "returns. The exact crossover is in the table; it is a model output, "
        "not a measured multi-GPU result.",
        "",
        "The Seq-Global proxy is included only so the model has all four arms. "
        "It is not a W2 measurement and must not be quoted as one. The NCCL "
        "1 MB point-to-point microbenchmark (`scripts/jpdc/nccl_p2p_benchmark.py`, "
        "`scripts/slurm_scripts/w2_nccl_p2p.slurm`) supplies the latency floor "
        "for the 10 Gbit/s end once the author authorizes that short job.",
        "",
        f"Rows: {len(per_seed)} per-seed/link observations; "
        f"summary from {per_seed.seed.nunique()} seeds.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g1-per-seed",
        type=Path,
        default=Path("../jpdc_data/g1/g1_per_seed.csv"),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("../jpdc_data/canonical_runs_v2"),
    )
    parser.add_argument(
        "--w3-cut-dir",
        type=Path,
        default=Path("../jpdc_data/g1/w3_cut"),
    )
    parser.add_argument(
        "--stage-anatomy",
        type=Path,
        default=Path("../jpdc_data/g1/w0c/w0c_stage_anatomy.csv"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("../jpdc_data/g1/w2_link")
    )
    parser.add_argument(
        "--links-mbps", type=float, nargs="+", default=[100.0, 1000.0, 10000.0]
    )
    parser.add_argument("--idle-power-w", type=float, default=51.97)
    parser.add_argument("--nic-power-w", type=float, default=0.0)
    parser.add_argument("--nic-pj-per-bit", type=float, default=1.0)
    args = parser.parse_args()

    g1 = pd.read_csv(args.g1_per_seed)
    cut = pd.read_csv(args.w3_cut_dir / "w3_cut_per_seed.csv")
    anatomy = pd.read_csv(args.stage_anatomy)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_seed = build_rows(
        g1,
        args.runs_dir,
        cut,
        anatomy,
        args.links_mbps,
        args.idle_power_w,
        args.nic_power_w,
        args.nic_pj_per_bit,
    )
    if per_seed.empty:
        raise SystemExit("No W2 rows were built; check input paths.")
    summary = summarise(per_seed)
    per_seed.to_csv(args.out_dir / "w2_link_per_seed.csv", index=False)
    summary.to_csv(args.out_dir / "w2_link_summary.csv", index=False)
    (args.out_dir / "w2_link_summary.json").write_text(
        json.dumps(
            {
                "parameters": {
                    "steps_per_pass": STEPS_PER_PASS,
                    "idle_power_w": args.idle_power_w,
                    "nic_power_w": args.nic_power_w,
                    "nic_pj_per_bit": args.nic_pj_per_bit,
                    "links_mbps": args.links_mbps,
                    "target": "each arm's measured final test accuracy",
                },
                "rows": summary.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n"
    )
    markdown = write_markdown(summary, per_seed, args)
    (args.out_dir / "w2_link_summary.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
