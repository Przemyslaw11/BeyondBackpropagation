#!/usr/bin/env python3
"""W0c: cap-binding anatomy of the decisive G1 runs (per-task `.out`, not logs).

For every (experiment, seed) it keeps the *final* attempt — the per-task stdout
whose test accuracy matches the surviving run JSON and, among those, the latest
start time — and reports:

- MF-Seq stage epochs (M0, W1_M1, W2_M2, W3_M3), whether a stage hit the 100-epoch
  cap (`epochs >= cap`) and the M0 share of total epochs;
- BP / MF-Joint / MF-Simul epochs and cap binding from their JSON `epochs_completed`;
- a cross-check that MF-Seq's per-stage sum equals the JSON `epochs_completed`
  (the artifact the pass counts come from);
- the PPAM-era baseline from `tidy/epochs_by_unit.csv` for comparison (the shared
  log anatomy of the canonical September runs, which is intact).

The shared run log is deliberately not read (2026-10-17 amendment).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.jpdc.parse_task_outs import parse_out

REPO_ROOT = Path(__file__).resolve().parents[2]


def experiment_name(config_path: str) -> str:
    """The .out names the config file; results trees use the YAML experiment_name."""
    path = REPO_ROOT / config_path
    if path.exists():
        try:
            return str(yaml.safe_load(path.read_text()).get("experiment_name", Path(config_path).stem))
        except yaml.YAMLError:
            pass
    return Path(config_path).stem

SEEDS = list(range(42, 62))
MF_EXPERIMENTS = ["mf_cifar10_mlp_3x2000", "mf_cifar100_mlp_3x2000"]
OTHER_EXPERIMENTS = [
    "bp_cifar10_mlp_3x2000",
    "bp_cifar100_mlp_3x2000",
    "mf_joint_cifar10_mlp_3x2000",
    "mf_joint_cifar100_mlp_3x2000",
    "mf_simul_cifar10_mlp_3x2000",
    "mf_simul_cifar100_mlp_3x2000",
]
CAP = 100


def final_outs(outs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(outs_dir.glob("BBP_G1*.out")):
        if path.name.startswith("BBP_G1Pilot"):
            continue
        row = parse_out(path)
        if row is None:
            continue
        row["experiment"] = experiment_name(row["config"])
        row["out_mtime"] = path.stat().st_mtime
        row["start"] = row["start"] or ""
        rows.append(row)
    return pd.DataFrame(rows)


def pick_final(outs: pd.DataFrame, results_dir: Path, experiment: str, seed: int) -> dict | None:
    jpath = results_dir / "runs" / experiment / f"{experiment}_seed{seed}.json"
    if not jpath.exists():
        return None
    record = json.loads(jpath.read_text())
    acc = record.get("test_accuracy")
    candidates = outs[
        (outs.experiment == experiment)
        & (outs.seed == seed)
        & (outs.test_accuracy.notna())
        & ((outs.test_accuracy - acc).abs() <= 0.011)
    ].sort_values(["start", "out_mtime"])
    if candidates.empty:
        return None
    row = candidates.iloc[-1].to_dict()
    row["json_epochs"] = record.get("epochs_completed")
    row["json_accuracy"] = acc
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outs-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--epochs-by-unit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    outs = final_outs(args.outs_dir)
    rows = []
    for experiment in MF_EXPERIMENTS + OTHER_EXPERIMENTS:
        for seed in SEEDS:
            row = pick_final(outs, args.results_dir, experiment, seed)
            if row is None:
                continue
            rows.append(row)
    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "w0c_stage_anatomy.csv", index=False)

    lines = ["# W0c — cap binding and stage anatomy of the decisive G1 runs", ""]
    lines.append("Source: per-task SLURM stdout (final attempt per arm x seed, matched to the "
                 "surviving run JSON by test accuracy and start time). Cap = 100 epochs per unit. "
                 "The shared run log is not read (2026-10-17 amendment).")
    lines.append("")
    lines.append("Coverage: all 40 MF-Seq runs (20 per dataset) have stdout, so the cap anatomy is "
                 "complete. Of the 160 decisive tasks, 56 lost their `.out` when the quota watchdog "
                 "killed job 3372435 before buffers flushed; 119/120 arm x seed cells are matched "
                 "across the six arms (only mf_joint C10 seed 49 is missing).")
    lines.append("")

    for experiment in MF_EXPERIMENTS:
        sub = df[df.experiment == experiment].copy()
        if sub.empty:
            continue
        # A completed run whose stage has no "Early Stopping" line trained the
        # full cap (the log line is only written when ES fires). Test Set Results
        # exists only for completed runs, so this inference is safe.
        for col in ("w1_epochs", "w2_epochs", "w3_epochs"):
            sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(CAP)
        sub["m0_epochs"] = pd.to_numeric(sub["m0_epochs"], errors="coerce")
        lines.append(f"## {experiment} (n={len(sub)})")
        lines.append("")
        lines.append("| stage | mean epochs | median | max | cap-bound share |")
        lines.append("|---|---|---|---|---|")
        for stage, col in [("M0", "m0_epochs"), ("W1_M1", "w1_epochs"), ("W2_M2", "w2_epochs"), ("W3_M3", "w3_epochs")]:
            values = sub[col].dropna()
            if values.empty:
                lines.append(f"| {stage} | n/a | | | |")
                continue
            cap_share = float((values >= CAP).mean()) * 100
            lines.append(f"| {stage} | {values.mean():.1f} | {values.median():.0f} "
                         f"| {values.max():.0f} | {cap_share:.0f}% |")
        m0_values = pd.to_numeric(sub.m0_epochs, errors="coerce")
        totals = pd.to_numeric(sub.json_epochs, errors="coerce")
        stage_sum = sub[["m0_epochs", "w1_epochs", "w2_epochs", "w3_epochs"]].sum(axis=1, min_count=1)
        mismatch = (stage_sum - totals).abs() > 0.5
        lines.append("")
        lines.append(f"Per-stage sum vs JSON `epochs_completed`: "
                     f"{(~mismatch).sum()}/{len(sub)} match; M0 share of passes "
                     f"{100 * m0_values.sum() / totals.sum():.1f}%; "
                     f"runs with any stage at cap {int((sub[['m0_epochs','w1_epochs','w2_epochs','w3_epochs']].fillna(0) >= CAP).any(axis=1).sum())}/{len(sub)}.")
        if mismatch.any():
            bad = sub[mismatch][["seed", "json_epochs", "m0_epochs", "w1_epochs", "w2_epochs", "w3_epochs"]]
            lines.append("")
            lines.append("Mismatched runs:")
            lines.append("")
            lines.append("```")
            lines.append(bad.to_string(index=False))
            lines.append("```")
        lines.append("")

    lines.append("## Other arms (JSON `epochs_completed`)")
    lines.append("")
    lines.append("| experiment | n | mean epochs | max | at cap |")
    lines.append("|---|---|---|---|---|")
    for experiment in OTHER_EXPERIMENTS:
        sub = df[df.experiment == experiment]
        if sub.empty:
            continue
        values = pd.to_numeric(sub.json_epochs, errors="coerce")
        lines.append(f"| {experiment} | {len(sub)} | {values.mean():.1f} | {values.max():.0f} "
                     f"| {int((values >= CAP).sum())}/{len(sub)} |")
    lines.append("")

    ppam = pd.read_csv(args.epochs_by_unit, usecols=["experiment", "seed", "unit", "epochs_run", "stop_reason"])
    ppam = ppam[ppam.experiment.isin(MF_EXPERIMENTS)]
    if not ppam.empty:
        lines.append("## PPAM-era baseline (canonical `epochs_by_unit.csv`, same cap)")
        lines.append("")
        lines.append("| stage | mean epochs | median | max | cap-bound share |")
        lines.append("|---|---|---|---|---|")
        for stage in ("M0", "W1_M1", "W2_M2", "W3_M3"):
            values = ppam[ppam.unit == stage].epochs_run
            if values.empty:
                continue
            lines.append(f"| {stage} | {values.mean():.1f} | {values.median():.0f} "
                         f"| {values.max():.0f} | {100 * (ppam[ppam.unit == stage].stop_reason == 'cap').mean():.0f}% |")
        lines.append("")

    (args.out_dir / "w0c_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
