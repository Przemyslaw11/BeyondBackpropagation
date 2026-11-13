#!/usr/bin/env python3
"""W0d: baseline-strength audit of the BP reference (zero GPU).

The strong-BP control (H-W1d) only matters if the 100-epoch shared cap is not
what stops BP. This audit reads each decisive BP run's own task stdout and
checks, per seed:

- how many epochs BP actually ran (JSON `epochs_completed` cross-check);
- the epoch of the best validation loss and the early-stopping tail
  (last - best, expected ≈ patience = 20 when the stop is genuine);
- whether validation loss was rising over that tail (i.e. the stop was not a
  plateau cut short).

It also compares BP's accuracy with the PPAM-era canonical mean, so a weakened
baseline would show up as an accuracy drop against its own published record.

No numbers are taken from the shared append-only log.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from scripts.jpdc.w0c_cap_anatomy import final_outs, pick_final

EXPERIMENTS = ["bp_cifar10_mlp_3x2000", "bp_cifar100_mlp_3x2000"]
SEEDS = list(range(42, 62))
EPOCH_RE = re.compile(
    r"BP Epoch (\d+)/\d+ \| LR: [^|]+ \| Train Loss: [\d.]+, Acc: [\d.]+% "
    r"\| Val Loss: ([\d.]+), Acc: ([\d.]+)%"
)
ES_RE = re.compile(r"Early Stopping Triggered")


def epoch_curve(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            rows.append(
                {
                    "epoch": int(m.group(1)),
                    "val_loss": float(m.group(2)),
                    "val_acc": float(m.group(3)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outs-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--tidy", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    outs = final_outs(args.outs_dir)
    rows = []
    curves = {}
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            picked = pick_final(outs, args.results_dir, experiment, seed)
            if picked is None:
                continue
            path = args.outs_dir / picked["out_file"]
            text = path.read_text(errors="replace")
            curve = epoch_curve(text)
            if curve.empty:
                continue
            best_idx = int(curve.val_loss.idxmin())
            best = curve.loc[best_idx]
            last = curve.iloc[-1]
            rows.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "epochs": int(last.epoch),
                    "json_epochs": picked["json_epochs"],
                    "best_epoch": int(best.epoch),
                    "tail": int(last.epoch - best.epoch),
                    "best_val_loss": float(best.val_loss),
                    "last_val_loss": float(last.val_loss),
                    "val_loss_increase": float(last.val_loss - best.val_loss),
                    "best_val_acc": float(best.val_acc),
                    "last_val_acc": float(last.val_acc),
                    "test_accuracy": float(picked["json_accuracy"]),
                    "early_stopping_logged": bool(ES_RE.search(text)),
                }
            )
            curves[(experiment, seed)] = curve

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "w0d_bp_convergence.csv", index=False)

    lines = ["# W0d — BP baseline-strength audit (decisive G1 runs)", ""]
    lines.append(
        "Per-task stdout only (final attempt matched to the surviving JSON); shared log not read. "
        "Tail = last epoch − best-validation-loss epoch; the shared early-stopping rule has "
        "patience 20, so a genuine stop leaves a tail of ≈20."
    )
    lines.append("")

    for experiment in EXPERIMENTS:
        sub = df[df.experiment == experiment]
        if sub.empty:
            continue
        dataset = "cifar100" if "_cifar100_" in experiment else "cifar10"
        lines.append(f"## {experiment} (n={len(sub)})")
        lines.append("")
        lines.append(
            f"- epochs: **{sub.epochs.mean():.1f}** (max {sub.epochs.max()}), cap = 100; "
            f"JSON `epochs_completed` = last printed epoch + 1 in "
            f"{int((sub.epochs + 1 == sub.json_epochs).sum())}/{len(sub)} runs"
        )
        lines.append(
            f"- best-val-loss epoch: **{sub.best_epoch.mean():.1f}** "
            f"(median {sub.best_epoch.median():.0f}, max {sub.best_epoch.max()})"
        )
        lines.append(
            f"- early-stopping tail: **{sub['tail'].mean():.1f}** epochs "
            f"(median {sub['tail'].median():.0f}; "
            f"{int((sub['tail'] >= 15).sum())}/{len(sub)} with tail ≥ 15)"
        )
        lines.append(
            f"- validation loss rose over the tail in {int((sub.val_loss_increase > 0).sum())}/{len(sub)} runs; "
            f"mean increase **{sub.val_loss_increase.mean():+.4f}** "
            f"({sub.best_val_loss.mean():.4f} → {sub.last_val_loss.mean():.4f})"
        )
        lines.append(
            f"- test accuracy: **{sub.test_accuracy.mean():.2f}%** "
            f"(SD {sub.test_accuracy.std(ddof=1):.2f}); "
            f"ES line present in {int(sub.early_stopping_logged.sum())}/{len(sub)}"
        )

        tidy = pd.read_csv(args.tidy)
        ppam = tidy[
            (tidy.experiment_name == f"bp_{dataset}_mlp_3x2000")
            & (tidy.protocol == "harmonised")
            & (tidy.metric == "test_accuracy")
        ].value
        if not ppam.empty:
            lines.append(
                f"- PPAM-era canonical mean: **{ppam.mean():.2f}%**; "
                f"G1 − PPAM = **{sub.test_accuracy.mean() - ppam.mean():+.2f} pp**"
            )
        lines.append("")

    df.to_csv(args.out_dir / "w0d_bp_convergence.csv", index=False)
    (args.out_dir / "w0d_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
