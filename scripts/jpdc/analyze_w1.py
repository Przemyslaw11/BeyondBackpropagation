#!/usr/bin/env python3
"""W1 analysis: the attribution ladder (PREREG-phase2.md §3).

Reads per arm x dataset x seed 42-61, exactly like analyze_g1:
- run JSON:  <results>/runs/<exp>/<exp>_seed<s>.json   (accuracy, passes, duration)
- NVML CSV:  the exact `monitoring_csv_path` stored in that JSON (energy, Wh)

Arms (6 main): bp, mf_joint, mf_simul, mf_seq, mf_global, mf_nom0.
Controls (descriptive, not in Holm): bp_long (max_epochs 300), bp_bestval,
bp_sched, mf_seq_protocol_b, mf_nom0_protocol_b, and the conditional
mf_seq_cap300.

Decision blocks (frozen thresholds, CIFAR-100 primary; CIFAR-10 consistency):
- H-W1a (locality):      global - seq <= -2.0 pp AND global - simul <= +1.0 pp
- H-W1b (schedule):      global - seq >= -1.0 pp
- H-W1c (M0 overhead):   nom0 - seq >= -0.5 pp AND savings >= 15% passes AND >= 15% energy
- H-W1d (strong BP):     bp_long - seq >= -1.0 pp  (and passes <= 2x seq passes)
Holm: three primary comparisons (global-seq, global-simul, nom0-seq) on C100.

Missing arms are reported as "not run" and their hypotheses left undecided; the
script never imputes. All numbers come from the run artifacts named above; the
shared append-only log is never read (PREREG.md §10 amendment). Means and
medians are both reported; inference uses the pre-registered Wilcoxon test on
paired differences.
"""

import argparse
import json
from pathlib import Path

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
ARMS = {
    "bp": "bp_{dataset}_mlp_3x2000",
    "mf_joint": "mf_joint_{dataset}_mlp_3x2000",
    "mf_simul": "mf_simul_{dataset}_mlp_3x2000",
    "mf_seq": "mf_{dataset}_mlp_3x2000",
    "mf_global": "mf_global_{dataset}_mlp_3x2000",
    "mf_nom0": "mf_nom0_{dataset}_mlp_3x2000",
}
CONTROL_ARMS = {
    "bp_long": "bp_long_{dataset}_mlp_3x2000",
    "bp_bestval": "bp_bestval_{dataset}_mlp_3x2000",
    "bp_sched": "bp_sched_{dataset}_mlp_3x2000",
    "mf_seq_protocol_b": "mf_protocol_b_{dataset}_mlp_3x2000",
    "mf_nom0_protocol_b": "mf_nom0_protocol_b_{dataset}_mlp_3x2000",
    "mf_seq_cap300": "mf_cap300_{dataset}_mlp_3x2000",
}


def collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    for arm, pattern in {**ARMS, **CONTROL_ARMS}.items():
        for dataset in ("cifar10", "cifar100"):
            experiment = pattern.format(dataset=dataset)
            for seed in SEEDS:
                jpath = results_dir / "runs" / experiment / f"{experiment}_seed{seed}.json"
                if not jpath.exists():
                    continue
                try:
                    record = json.loads(jpath.read_text())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                rows.append(
                    dict(
                        arm=arm,
                        dataset=dataset,
                        seed=seed,
                        test_accuracy=record.get("test_accuracy"),
                        passes=run_passes_json(record),
                        energy_wh=run_energy_wh(results_dir, record),
                        duration_sec=record.get("total_run_duration_sec"),
                    )
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "arm",
                "dataset",
                "seed",
                "test_accuracy",
                "passes",
                "energy_wh",
                "duration_sec",
            ]
        )
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, arm_a: str, arm_b: str, dataset: str, metric: str) -> dict:
    a = df[(df.arm == arm_a) & (df.dataset == dataset)].set_index("seed")[metric]
    b = df[(df.arm == arm_b) & (df.dataset == dataset)].set_index("seed")[metric]
    common = sorted(a.index.intersection(b.index))
    diffs = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 3:
        return dict(
            n=len(diffs),
            mean=float("nan"),
            median=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            p=1.0,
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = diffs[rng.integers(0, len(diffs), size=(BOOTSTRAP_N, len(diffs)))].mean(axis=1)
    p = 1.0 if np.allclose(diffs, 0.0) else float(stats.wilcoxon(diffs, alternative="two-sided").pvalue)
    return dict(
        n=len(diffs),
        mean=float(np.mean(diffs)),
        median=float(np.median(diffs)),
        ci_low=float(np.percentile(boot, 2.5)),
        ci_high=float(np.percentile(boot, 97.5)),
        p=p,
    )


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvals[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def arm_present(df: pd.DataFrame, arm: str, dataset: str) -> bool:
    sub = df[(df.arm == arm) & (df.dataset == dataset)]
    return len(sub) >= 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    df = collect(args.results_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "w1_per_seed.csv", index=False)

    lines = ["# W1 — attribution ladder (PREREG-phase2.md)", ""]
    lines.append("Accuracy primary; passes/energy co-primary. Pairing by seed (42–61), "
                 "Wilcoxon + 10k bootstrap; Holm across the three C100 primary comparisons.")
    lines.append("")

    lines.append("## Per-arm means (95% CI of the mean)")
    lines.append("")
    lines.append("| arm | dataset | n | acc % | passes | energy Wh | wall s |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in list(ARMS) + list(CONTROL_ARMS):
        for dataset in ("cifar10", "cifar100"):
            sub = df[(df.arm == arm) & (df.dataset == dataset)]
            if sub.empty:
                continue
            cells = []
            for metric in ("test_accuracy", "passes", "energy_wh", "duration_sec"):
                values = sub[metric].astype(float).dropna()
                if len(values) < 2:
                    cells.append("n/a")
                    continue
                half = stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)
                cells.append(f"{values.mean():.3g} [{values.mean() - half:.3g}, {values.mean() + half:.3g}]")
            lines.append(f"| {arm} | {dataset} | {len(sub)} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-arm medians")
    lines.append("")
    lines.append("| arm | dataset | n | median acc % | median passes | median energy Wh | median wall s |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in list(ARMS) + list(CONTROL_ARMS):
        for dataset in ("cifar10", "cifar100"):
            sub = df[(df.arm == arm) & (df.dataset == dataset)]
            if sub.empty:
                continue
            medians = []
            for metric in ("test_accuracy", "passes", "energy_wh", "duration_sec"):
                values = sub[metric].astype(float).dropna()
                medians.append(f"{values.median():.3g}" if len(values) else "n/a")
            lines.append(
                f"| {arm} | {dataset} | {len(sub)} | " + " | ".join(medians) + " |"
            )
    lines.append("")

    lines.append("## Primary comparisons (CIFAR-100)")
    lines.append("")
    comparisons = []
    for arm_a, arm_b in [("mf_global", "mf_seq"), ("mf_global", "mf_simul"), ("mf_nom0", "mf_seq")]:
        stat = paired(df, arm_a, arm_b, "cifar100", "test_accuracy") if (
            arm_present(df, arm_a, "cifar100") and arm_present(df, arm_b, "cifar100")
        ) else dict(n=0, mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"), p=1.0)
        comparisons.append((arm_a, arm_b, stat))
    pvals = [stat["p"] for _, _, stat in comparisons]
    adjusted = holm(pvals) if all(np.isfinite(pvals)) else pvals
    lines.append("| delta | n | mean pp | median pp | 95% CI | raw p | Holm p |")
    lines.append("|---|---|---|---|---|---|---|")
    for (arm_a, arm_b, stat), p_holm in zip(comparisons, adjusted):
        if stat["n"] == 0:
            lines.append(f"| {arm_a} − {arm_b} | — | not run | — | — | — | — |")
        else:
            lines.append(
                f"| {arm_a} − {arm_b} | {stat['n']} | {stat['mean']:+.2f} | "
                f"{stat['median']:+.2f} | "
                f"[{stat['ci_low']:+.2f}, {stat['ci_high']:+.2f}] | {stat['p']:.4f} | {p_holm:.4f} |"
            )
    lines.append("")

    lines.append("## Decision blocks (frozen thresholds)")
    lines.append("")
    d_global_seq = next(s for a, b, s in comparisons if (a, b) == ("mf_global", "mf_seq"))
    d_global_simul = next(s for a, b, s in comparisons if (a, b) == ("mf_global", "mf_simul"))
    d_nom0_seq = next(s for a, b, s in comparisons if (a, b) == ("mf_nom0", "mf_seq"))

    verdicts = []
    if np.isfinite(d_global_seq["mean"]):
        if d_global_seq["mean"] <= -2.0 and np.isfinite(d_global_simul["mean"]) and d_global_simul["mean"] <= 1.0:
            verdicts.append("H-W1a **supported**: unfreezing the prefix collapses Seq-Global toward the joint arms.")
        elif d_global_seq["mean"] >= -1.0:
            verdicts.append("H-W1b **supported**: the staged schedule keeps its edge with a trainable prefix.")
        else:
            verdicts.append("H-W1a/b **intermediate**: Seq-Global sits between MF-Seq and the joint arms.")
    else:
        verdicts.append("H-W1a/b **not run** (mf_global data missing).")

    if np.isfinite(d_nom0_seq["mean"]):
        passes_saving = paired(df, "mf_nom0", "mf_seq", "cifar100", "passes")
        energy_saving = paired(df, "mf_nom0", "mf_seq", "cifar100", "energy_wh")
        pass_frac = -passes_saving["mean"] / df[(df.arm == "mf_seq") & (df.dataset == "cifar100")].passes.mean()
        energy_frac = -energy_saving["mean"] / df[(df.arm == "mf_seq") & (df.dataset == "cifar100")].energy_wh.mean()
        ok = d_nom0_seq["mean"] >= -0.5 and pass_frac >= 0.15 and energy_frac >= 0.15
        verdicts.append(
            f"H-W1c {'**supported**' if ok else '**not supported**'}: Δacc {d_nom0_seq['mean']:+.2f} pp, "
            f"pass saving {pass_frac * 100:.0f}%, energy saving {energy_frac * 100:.0f}%."
        )
    else:
        verdicts.append("H-W1c **not run** (mf_nom0 data missing).")

    if arm_present(df, "bp_long", "cifar100") and arm_present(df, "mf_seq", "cifar100"):
        d_bp_long = paired(df, "bp_long", "mf_seq", "cifar100", "test_accuracy")
        seq_passes = df[(df.arm == "mf_seq") & (df.dataset == "cifar100")].passes.mean()
        bp_long_passes = df[(df.arm == "bp_long") & (df.dataset == "cifar100")].passes.mean()
        verdicts.append(
            f"H-W1d: BP-long Δ {d_bp_long['mean']:+.2f} pp "
            f"[{d_bp_long['ci_low']:+.2f}, {d_bp_long['ci_high']:+.2f}] at "
            f"{bp_long_passes / seq_passes:.2f}× MF-Seq passes."
        )
    else:
        verdicts.append("H-W1d **not run** (bp_long data missing).")

    for line in verdicts:
        lines.append(f"- {line}")
    lines.append("")

    lines.append("## Control and robustness checks (descriptive; not in Holm)")
    lines.append("")
    lines.append(
        "These are controls added before any W1 confirmatory result. They do not "
        "change the six-arm primary family, thresholds or Holm correction. BP under "
        "Protocol B is the core BP arm; no duplicate BP Protocol-B runs exist."
    )
    lines.append("")
    lines.append("| dataset | contrast | metric | n | mean diff | median diff | 95% CI | raw p |")
    lines.append("|---|---|---|---|---|---|---|---|")
    contrasts = [
        ("bp_bestval", "bp", "test_accuracy", "BP-bestval − BP (pp)"),
        ("bp_sched", "bp", "test_accuracy", "BP-sched − BP (pp)"),
        ("mf_seq_protocol_b", "bp", "test_accuracy", "MF-Seq ProtoB − BP (pp)"),
        ("mf_seq_protocol_b", "bp", "passes", "MF-Seq ProtoB − BP passes"),
        ("mf_seq_protocol_b", "bp", "energy_wh", "MF-Seq ProtoB − BP energy (Wh)"),
        ("mf_nom0_protocol_b", "mf_seq_protocol_b", "test_accuracy", "nom0 ProtoB − MF-Seq ProtoB (pp)"),
        ("mf_nom0_protocol_b", "mf_seq_protocol_b", "passes", "nom0 ProtoB − MF-Seq ProtoB passes"),
        ("mf_nom0_protocol_b", "mf_seq_protocol_b", "energy_wh", "nom0 ProtoB − MF-Seq ProtoB energy (Wh)"),
        ("bp_sched", "bp", "energy_wh", "BP-sched − BP energy (Wh)"),
    ]
    for dataset in ("cifar10", "cifar100"):
        for arm_a, arm_b, metric, label in contrasts:
            if not (arm_present(df, arm_a, dataset) and arm_present(df, arm_b, dataset)):
                continue
            stat = paired(df, arm_a, arm_b, dataset, metric)
            lines.append(
                f"| {dataset} | {label} | {metric} | {stat['n']} | "
                f"{stat['mean']:+.3f} | {stat['median']:+.3f} | "
                f"[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}] | "
                f"{stat['p']:.4f} |"
            )
    lines.append("")
    lines.append(
        "Tuning disclosure: Seq-Global and BP-sched each receive 50 lr/wd trials per "
        "dataset; BP-bestval and both Protocol-B arms inherit frozen values and add "
        "0 trials. The six core arms keep their original budgets."
    )
    lines.append("")

    if arm_present(df, "mf_seq_cap300", "cifar100") and arm_present(df, "mf_seq", "cifar100"):
        d_cap = paired(df, "mf_seq_cap300", "mf_seq", "cifar100", "test_accuracy")
        lines.append(f"Cap sensitivity (H-W1cap step 2): cap-300 Δ {d_cap['mean']:+.2f} pp "
                     f"[{d_cap['ci_low']:+.2f}, {d_cap['ci_high']:+.2f}].")
        lines.append("")

    lines.append("CIFAR-10 consistency table: run the same commands with dataset=cifar10; "
                 "no threshold is re-tuned there.")
    lines.append("")

    (args.out_dir / "w1_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
