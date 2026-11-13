#!/usr/bin/env python3
"""Independent re-derivation of the headline G1 numbers.

Deliberately does NOT import analyze_g1.py or w0a_energy_memory.py. Reads the
frozen per-seed table, recomputes paired deltas vs BP, bootstrap CIs and
Wilcoxon p-values, applies Holm to the two pre-declared families, and compares
the result with the stored summaries.

Families (declared before looking at these outputs):
  * accuracy, decisive: MF-Simul vs BP, CIFAR-10 and CIFAR-100 (2 tests,
    matching PREREG.md's G1 decision family and analyze_g1.py);
  * cost, descriptive: the three MF arms vs BP within each dataset for passes
    and energy separately (3 tests per dataset), matching w0a's convention.

The bootstrap seed differs from production (20261021 vs 0), so CIs can differ
by Monte-Carlo noise; p-values are exact Wilcoxon and should match exactly.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEEDS = list(range(42, 62))
ARMS = ["bp", "mf_joint", "mf_simul", "mf_seq"]
DATASETS = ["cifar10", "cifar100"]
BOOT_N = 20_000
BOOT_SEED = 20261021


def t_ci(values: np.ndarray) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    half = stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)
    return float(values.mean() - half), float(values.mean() + half)


def paired(df: pd.DataFrame, arm: str, dataset: str, metric: str) -> dict:
    a = df[(df.arm == arm) & (df.dataset == dataset)].set_index("seed")[metric]
    b = df[(df.arm == "bp") & (df.dataset == dataset)].set_index("seed")[metric]
    common = sorted(set(a.index) & set(b.index))
    av = a.loc[common].to_numpy(float)
    bv = b.loc[common].to_numpy(float)
    ok = ~(np.isnan(av) | np.isnan(bv))
    diffs = av[ok] - bv[ok]
    if len(diffs) == 0:
        return dict(n=0, mean_diff=np.nan, lo=np.nan, hi=np.nan, p=1.0)
    if np.allclose(diffs, 0.0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(diffs, alternative="two-sided").pvalue)
    rng = np.random.default_rng(BOOT_SEED)
    boot = diffs[rng.integers(0, len(diffs), size=(BOOT_N, len(diffs)))].mean(axis=1)
    return dict(
        n=len(diffs),
        mean_diff=float(diffs.mean()),
        lo=float(np.percentile(boot, 2.5)),
        hi=float(np.percentile(boot, 97.5)),
        p=p,
    )


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    out = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, pvals[idx] * (len(pvals) - rank)))
        out[idx] = running
    return out.tolist()


def parse_per_arm(path: Path) -> dict:
    out = {}
    row = re.compile(
        r"^\| (bp|mf_joint|mf_simul|mf_seq) \| (cifar10|cifar100) \| (\d+) \|"
    )
    for line in path.read_text().splitlines():
        m = row.match(line)
        if m:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            out[(m.group(1), m.group(2))] = {
                "accuracy": float(cells[3].split()[0]),
                "passes": float(cells[4]),
                "energy": float(cells[5]),
            }
    return out


def parse_w0a(path: Path) -> dict:
    out = {}
    row = re.compile(
        r"^\| (cifar10|cifar100) \| (energy_wh|passes|mean_power_w) \| "
        r"(mf_joint|mf_simul|mf_seq) \| (\d+) \| ([+-][\d.e+-]+) \| "
        r"\[([+-][\d.e+-]+), ([+-][\d.e+-]+)\] \| "
        r"([\d.]+) \(Holm ([\d.]+)\) \|"
    )
    for line in path.read_text().splitlines():
        m = row.match(line)
        if m:
            out[(m.group(1), m.group(2), m.group(3))] = dict(
                mean=float(m.group(5)),
                lo=float(m.group(6)),
                hi=float(m.group(7)),
                p=float(m.group(8)),
                p_holm=float(m.group(9)),
            )
    return out


def parse_g1_simul(path: Path) -> dict:
    text = path.read_text()
    section = text.split("## MF-Simul vs BP")[1].split("## MF-Seq")[0]
    out = {}
    row = re.compile(
        r"^\| (cifar10|cifar100) \| 20 \| ([+-][\d.]+) \| "
        r"\[([+-][\d.]+), ([+-][\d.]+)\] \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|"
    )
    for line in section.splitlines():
        m = row.match(line)
        if m:
            out[m.group(1)] = dict(
                mean=float(m.group(2)),
                lo=float(m.group(3)),
                hi=float(m.group(4)),
                p=float(m.group(5)),
                p_holm=float(m.group(6)),
                R=float(m.group(7)),
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-seed", type=Path, required=True)
    parser.add_argument("--cost-per-seed", type=Path, required=True)
    parser.add_argument("--stored-summary", type=Path, required=True)
    parser.add_argument("--w0a-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.per_seed)
    cost = pd.read_csv(args.cost_per_seed)
    df = df[df.seed.isin(SEEDS)]
    cost = cost[cost.seed.isin(SEEDS)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-arm means ----
    rows = []
    for dataset in DATASETS:
        for arm in ARMS:
            sub = df[(df.arm == arm) & (df.dataset == dataset)]
            csub = cost[(cost.arm == arm) & (cost.dataset == dataset)]
            row = {"arm": arm, "dataset": dataset, "n": len(sub)}
            for metric in ("test_accuracy", "passes", "energy_wh"):
                lo, hi = t_ci(sub[metric].to_numpy(float))
                row[f"{metric}_mean"] = float(sub[metric].mean())
                row[f"{metric}_lo"] = lo
                row[f"{metric}_hi"] = hi
            lo, hi = t_ci(csub["mean_power_w"].to_numpy(float))
            row["power_mean"], row["power_lo"], row["power_hi"] = (
                float(csub.mean_power_w.mean()),
                lo,
                hi,
            )
            rows.append(row)
    per_arm = pd.DataFrame(rows)
    per_arm.to_csv(args.out_dir / "independent_per_arm.csv", index=False)

    # ---- paired deltas + families ----
    paired_rows = []
    for metric in ("test_accuracy", "passes", "energy_wh"):
        for dataset in DATASETS:
            tests = [(arm, paired(df, arm, dataset, metric)) for arm in ARMS[1:]]
            adj = holm([t[1]["p"] for t in tests])
            for (arm, st), ph in zip(tests, adj):
                paired_rows.append(
                    dict(
                        family=f"{metric}:{dataset}",
                        metric=metric,
                        dataset=dataset,
                        arm=arm,
                        n=st["n"],
                        mean_diff=st["mean_diff"],
                        ci_low=st["lo"],
                        ci_high=st["hi"],
                        p_raw=st["p"],
                        p_holm=ph,
                    )
                )

    # decisive accuracy family: MF-Simul across the two datasets
    dec = [paired(df, "mf_simul", ds, "test_accuracy") for ds in DATASETS]
    dec_adj = holm([d["p"] for d in dec])

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(args.out_dir / "independent_paired.csv", index=False)

    # ---- comparisons with stored summaries ----
    stored = parse_per_arm(args.stored_summary)
    w0a = parse_w0a(args.w0a_summary)
    g1s = parse_g1_simul(args.stored_summary)
    checks = []

    def add(scope, dataset, arm, metric, independent, stored_value, tol):
        delta = abs(independent - stored_value)
        checks.append(
            dict(
                scope=scope,
                dataset=dataset,
                arm=arm,
                metric=metric,
                independent=independent,
                stored=stored_value,
                delta=delta,
                tolerance=tol,
                ok=bool(delta <= tol),
            )
        )

    for _, row in per_arm.iterrows():
        key = (row.arm, row.dataset)
        if key in stored:
            add(
                "per-arm",
                row.dataset,
                row.arm,
                "accuracy",
                row.test_accuracy_mean,
                stored[key]["accuracy"],
                0.01,
            )
            add(
                "per-arm",
                row.dataset,
                row.arm,
                "passes",
                row.passes_mean,
                stored[key]["passes"],
                0.05,
            )
            add(
                "per-arm",
                row.dataset,
                row.arm,
                "energy",
                row.energy_wh_mean,
                stored[key]["energy"],
                0.02,
            )

    for _, row in paired_df.iterrows():
        key = (row.dataset, row.metric, row.arm)
        if key in w0a:
            ref = w0a[key]
            add(
                "paired-mean",
                row.dataset,
                row.arm,
                row.metric,
                row.mean_diff,
                ref["mean"],
                0.02,
            )
            add(
                "paired-CI-lo",
                row.dataset,
                row.arm,
                row.metric,
                row.ci_low,
                ref["lo"],
                0.06,
            )
            add(
                "paired-CI-hi",
                row.dataset,
                row.arm,
                row.metric,
                row.ci_high,
                ref["hi"],
                0.06,
            )
            add(
                "paired-p", row.dataset, row.arm, row.metric, row.p_raw, ref["p"], 0.001
            )
            add(
                "paired-p-holm",
                row.dataset,
                row.arm,
                row.metric,
                row.p_holm,
                ref["p_holm"],
                0.001,
            )

    bp_passes = {
        ds: per_arm[(per_arm.arm == "bp") & (per_arm.dataset == ds)].passes_mean.iloc[0]
        for ds in DATASETS
    }
    for ds, st, ph in zip(DATASETS, dec, dec_adj):
        ref = g1s[ds]
        add("G1-accuracy", ds, "mf_simul", "mean", st["mean_diff"], ref["mean"], 0.01)
        add("G1-accuracy", ds, "mf_simul", "CI-lo", st["lo"], ref["lo"], 0.06)
        add("G1-accuracy", ds, "mf_simul", "CI-hi", st["hi"], ref["hi"], 0.06)
        add("G1-accuracy", ds, "mf_simul", "p", st["p"], ref["p"], 0.001)
        add("G1-accuracy", ds, "mf_simul", "p-holm", ph, ref["p_holm"], 0.001)
        tmp = per_arm[
            (per_arm.arm == "mf_simul") & (per_arm.dataset == ds)
        ].passes_mean.iloc[0]
        add(
            "G1-accuracy",
            ds,
            "mf_simul",
            "passes-ratio",
            tmp / bp_passes[ds],
            ref["R"],
            0.01,
        )

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(args.out_dir / "independent_vs_stored.csv", index=False)
    bad = checks_df[~checks_df.ok]

    # ---- markdown ----
    lines = ["# Independent verification of the G1 headline numbers", ""]
    lines.append(
        f"Separate reducer (no import of `analyze_g1.py`/`w0a_energy_memory.py`), "
        f"bootstrap seed {BOOT_SEED}, {BOOT_N:,} resamples. Sources: "
        f"`{args.per_seed.name}`, `{args.cost_per_seed.name}`."
    )
    lines.append("")
    lines.append("## Per-arm means (95% t-CI)")
    lines.append("")
    lines.append("| arm | dataset | n | accuracy % | passes | energy Wh | power W |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in per_arm.iterrows():
        lines.append(
            f"| {r.arm} | {r.dataset} | {r.n} | "
            f"{r.test_accuracy_mean:.2f} [{r.test_accuracy_lo:.2f}, {r.test_accuracy_hi:.2f}] | "
            f"{r.passes_mean:.1f} | {r.energy_wh_mean:.2f} | "
            f"{r.power_mean:.1f} [{r.power_lo:.1f}, {r.power_hi:.1f}] |"
        )
    lines.append("")
    lines.append("## Paired deltas vs BP (arm − BP)")
    lines.append("")
    lines.append("| metric | dataset | arm | n | mean diff | 95% CI | raw p | Holm |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in paired_df.iterrows():
        lines.append(
            f"| {r.metric} | {r.dataset} | {r.arm} | {r.n} | {r.mean_diff:+.3f} | "
            f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.p_raw:.4f} | {r.p_holm:.4f} |"
        )
    lines.append("")
    lines.append("**Decisive accuracy family (MF-Simul × 2 datasets, Holm 2):**")
    lines.append("")
    for ds, st, ph in zip(DATASETS, dec, dec_adj):
        lines.append(
            f"- {ds}: {st['mean_diff']:+.2f} pp [{st['lo']:+.2f}, {st['hi']:+.2f}], "
            f"raw p={st['p']:.4f}, Holm p={ph:.4f}"
        )
    lines.append("")
    lines.append("## Agreement with the stored summaries")
    lines.append("")
    lines.append(
        f"{len(checks_df) - len(bad)}/{len(checks_df)} checks agree within tolerance. "
        "Tolerances: 0.01 pp accuracy, 0.02 Wh energy, 0.02 W power, 0.05 passes, "
        "0.06 CI bound (different bootstrap seed), 0.001 p-value, 0.01 passes ratio."
    )
    if len(bad):
        lines.append("")
        lines.append("### Disagreements")
        lines.append("")
        lines.append(
            "| scope | dataset | arm | metric | independent | stored | delta | tol |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, d in bad.iterrows():
            lines.append(
                f"| {d.scope} | {d.dataset} | {d.arm} | {d.metric} | "
                f"{d.independent:.5f} | {d.stored:.5f} | {d.delta:.5f} | {d.tolerance} |"
            )
    else:
        lines.append("")
        lines.append("No disagreement outside tolerance.")
    lines.append("")
    lines.append(
        "The production summaries use bootstrap seed 0; this run uses "
        f"{BOOT_SEED}. CIs therefore differ by Monte-Carlo noise only; p-values "
        "are exact Wilcoxon and do not depend on the bootstrap."
    )
    (args.out_dir / "verification_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
