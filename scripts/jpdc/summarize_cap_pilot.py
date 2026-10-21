#!/usr/bin/env python3
"""H-W1cap Step 1 reducer: exploratory cap-300 pilot (PREREG-phase2.md §3).

Reads the 5 pilot run JSONs (seeds 42-46, CIFAR-100) plus their per-task stdout,
and applies the frozen trigger:

    run the full n=20 cap-300 arm iff
      mean accuracy gain vs the same-seed G1 MF-Seq baseline >= +0.5 pp
      OR mean W1_M1 epochs > 130.

Because a stage that never triggers early stopping writes no "Early Stopping"
line, a missing W1_M1 value is read as "trained to the cap" (300 here), exactly
as in `w0c_cap_anatomy.py`. The stage-sum cross-check against the JSON's
`epochs_completed` guards that inference.

This is exploratory: pilot numbers are never pooled with decisive tables.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.jpdc.parse_task_outs import parse_out

PILOT_EXPERIMENT = "mf_cap300_cifar100_mlp_3x2000"
BASELINE_EXPERIMENT = "mf_cifar100_mlp_3x2000"
SEEDS = [42, 43, 44, 45, 46]
CAP = 300
ACC_TRIGGER_PP = 0.5
EPOCH_TRIGGER = 130


def baseline_frame(results_dir: Path) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        path = results_dir / "runs" / BASELINE_EXPERIMENT / f"{BASELINE_EXPERIMENT}_seed{seed}.json"
        if path.exists():
            rows.append({"seed": seed, "test_accuracy": json.loads(path.read_text()).get("test_accuracy")})
    return pd.DataFrame(rows)


def pilot_outs(outs_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(outs_dir.glob("BBP_W1CapPilot*.out")):
        row = parse_out(path)
        if row is None:
            continue
        parts = Path(str(row["config"])).parts
        if len(parts) >= 2 and parts[0] == "configs" and parts[1] == "mf_cap300":
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--outs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    base = baseline_frame(args.results_dir)
    base_acc = float(base.test_accuracy.mean()) if not base.empty else float("nan")

    rows = []
    for seed in SEEDS:
        path = args.results_dir / "runs" / PILOT_EXPERIMENT / f"{PILOT_EXPERIMENT}_seed{seed}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        rows.append(
            {
                "seed": seed,
                "test_accuracy": record.get("test_accuracy"),
                "epochs_completed": record.get("epochs_completed"),
            }
        )
    pilot = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    outs = pilot_outs(args.outs_dir)
    stage_rows = []
    for seed in pilot.seed if not pilot.empty else []:
        candidates = outs[(outs.seed == seed) & outs.test_accuracy.notna()] if not outs.empty else pd.DataFrame()
        if not candidates.empty:
            acc = float(pilot.loc[pilot.seed == seed, "test_accuracy"].iloc[0])
            candidates = candidates[(candidates.test_accuracy - acc).abs() <= 0.011]
        if candidates.empty:
            stage_rows.append({"seed": seed, "m0_epochs": None, "w1_epochs": None, "w2_epochs": None, "w3_epochs": None})
            continue
        row = candidates.sort_values("start").iloc[-1]
        stage_rows.append(
            {
                "seed": seed,
                "m0_epochs": row.get("m0_epochs"),
                "w1_epochs": row.get("w1_epochs"),
                "w2_epochs": row.get("w2_epochs"),
                "w3_epochs": row.get("w3_epochs"),
            }
        )
    stages = pd.DataFrame(stage_rows)
    if not stages.empty:
        for col in ("w1_epochs", "w2_epochs", "w3_epochs"):
            stages[col] = pd.to_numeric(stages[col], errors="coerce").fillna(CAP)
        stages["m0_epochs"] = pd.to_numeric(stages["m0_epochs"], errors="coerce")

    lines = ["# H-W1cap Step 1 — cap-300 pilot (exploratory, n=5)", ""]
    lines.append(
        "Seeds 42-46, CIFAR-100, MF-Seq with `early_stopping.max_epochs: 300`; all other "
        "hyperparameters identical. Baseline = the same seeds in the decisive G1 MF-Seq arm "
        f"({BASELINE_EXPERIMENT}). This pilot never enters a decisive table."
    )
    lines.append("")

    if pilot.empty:
        lines.append("Pilot run JSONs not found; nothing to reduce yet.")
        (args.out_dir / "cap_pilot_summary.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return

    delta = float(pilot.test_accuracy.mean()) - base_acc if pd.notna(base_acc) else float("nan")
    w1_mean = float(stages.w1_epochs.mean()) if not stages.empty else float("nan")
    stage_sum = (
        stages[["m0_epochs", "w1_epochs", "w2_epochs", "w3_epochs"]].sum(axis=1, min_count=1)
        if not stages.empty
        else pd.Series(dtype=float)
    )
    complete = len(stages) == len(pilot) and bool((stage_sum.notna()).all())
    trigger = (pd.notna(delta) and delta >= ACC_TRIGGER_PP) or (pd.notna(w1_mean) and w1_mean > EPOCH_TRIGGER)

    lines.append("| seed | test acc % | epochs | M0 | W1_M1 | W2_M2 | W3_M3 |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, row in pilot.iterrows():
        srow = stages[stages.seed == row.seed]
        cells = [f"{row.test_accuracy:.2f}", f"{int(row.epochs_completed)}"]
        if srow.empty:
            cells += ["n/a"] * 4
        else:
            s = srow.iloc[0]
            cells += [
                "n/a" if pd.isna(s.m0_epochs) else f"{int(s.m0_epochs)}",
                f"{int(s.w1_epochs)}",
                f"{int(s.w2_epochs)}",
                f"{int(s.w3_epochs)}",
            ]
        lines.append(f"| {int(row.seed)} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append(f"- Mean accuracy: **{pilot.test_accuracy.mean():.2f}%**; G1 MF-Seq same-seed baseline: "
                 f"**{base_acc:.2f}%**; Δ = **{delta:+.2f} pp** (trigger ≥ +{ACC_TRIGGER_PP:.1f}).")
    lines.append(f"- Mean W1_M1 epochs: **{w1_mean:.1f}** (trigger > {EPOCH_TRIGGER}); missing ES lines "
                 f"read as cap = {CAP}.")
    lines.append(f"- Stage-sum cross-check: {'complete' if complete else 'INCOMPLETE — inspect before trusting W1 numbers'}.")
    lines.append("")
    lines.append(f"## Trigger: **{'YES — run the full n=20 cap-300 arm' if trigger else 'NO — keep the 100-epoch cap and report W0c'}**")
    lines.append("")
    if not trigger:
        lines.append(
            "Report the W0c cap-sensitivity table as the defense; do not run the cap-300 arm at n=20."
        )
    else:
        lines.append(
            "Submit `configs/mf_cap300/*` at n=20 with the same hygiene as the W1 family, then "
            "reduce with `analyze_w1.py` (H-W1cap Step 2)."
        )
    lines.append("")

    (args.out_dir / "cap_pilot_per_seed.csv").write_text(pilot.to_csv(index=False))
    if not stages.empty:
        (args.out_dir / "cap_pilot_stages.csv").write_text(stages.to_csv(index=False))
    (args.out_dir / "cap_pilot_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
