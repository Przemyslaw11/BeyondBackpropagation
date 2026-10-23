#!/usr/bin/env python3
"""Compute Ebar_BP from the core BP run JSONs.

Definition (PREREG-phase2.md; W1_MANIFEST_ADDITIONS.md, Protocol B):
    Ebar_BP = mean(passes) over the 20 core BP runs, per dataset
    per-stage Protocol B budget = ceil(Ebar_BP / S)
      S = 4  for contemporaneous MF-Seq Protocol B
      S = 3  for MF-Seq-nom0 Protocol B

For BP one epoch is one pass, so ``passes`` is the run's ``epochs_completed``.

The script reads ONLY ``experiment_name``, ``seed`` and ``epochs_completed``
(plus the presence of ``secondary_eval_seconds`` as a W1 marker).  It never
reads accuracy, loss, energy, duration or any secondary endpoint.

Safety: a results path containing a ``results`` component, or a record that
carries ``secondary_eval_seconds``, is treated as a possible W1 tree.  Reading
it requires ``--w1-ack``, which is the explicit acknowledgement that the
author has authorised the Ebar_BP computation on completed core BP runs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


def _parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        elif part:
            seeds.append(int(part))
    return sorted(set(seeds))


def _find_runs_root(results_dir: Path) -> Path:
    nested = results_dir / "runs"
    return nested if nested.is_dir() else results_dir


def _collect(
    root: Path, dataset: str, seeds: set[int], w1_ack: bool
) -> tuple[list[float], dict[int, float]]:
    experiment = f"bp_{dataset}_mlp_3x2000"
    directory = root / experiment
    if not directory.is_dir():
        raise SystemExit(f"missing run directory: {directory}")

    by_seed: dict[int, float] = {}
    for path in sorted(directory.glob("*.json")):
        record: Any = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise SystemExit(f"{path}: not a JSON object")
        if "secondary_eval_seconds" in record and not w1_ack:
            raise SystemExit(
                f"{path}: W1-era run detected (secondary_eval_seconds); "
                "pass --w1-ack only after the author authorises Ebar_BP"
            )
        if record.get("experiment_name") != experiment:
            raise SystemExit(
                f"{path}: experiment_name={record.get('experiment_name')!r}, "
                f"expected {experiment!r}"
            )
        seed = record.get("seed")
        if not isinstance(seed, int) or seed not in seeds:
            continue
        epochs = record.get("epochs_completed")
        if not isinstance(epochs, (int, float)) or not math.isfinite(epochs):
            raise SystemExit(f"{path}: missing/invalid epochs_completed")
        if epochs <= 0:
            raise SystemExit(f"{path}: epochs_completed must be positive")
        if seed in by_seed:
            raise SystemExit(f"{path}: duplicate seed {seed} for {experiment}")
        by_seed[seed] = float(epochs)

    values = [by_seed[seed] for seed in sorted(by_seed)]
    return values, by_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="repo results dir (with runs/) or a runs/ dir directly",
    )
    parser.add_argument("--seeds", default="42-61")
    parser.add_argument("--require-n", type=int, default=20)
    parser.add_argument("--datasets", default="cifar10,cifar100")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--w1-ack",
        action="store_true",
        help="acknowledge reading completed W1 core BP runs",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if "results" in results_dir.parts and not args.w1_ack:
        raise SystemExit(
            f"{results_dir}: path contains 'results'; pass --w1-ack only "
            "after the author authorises Ebar_BP on completed core BP runs"
        )

    seeds = _parse_seeds(args.seeds)
    seed_set = set(seeds)
    root = _find_runs_root(results_dir)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]

    output: dict[str, Any] = {
        "definition": "Ebar_BP = mean(epochs_completed) over core BP seeds",
        "seeds": seeds,
        "results_dir": str(results_dir),
        "datasets": {},
    }
    rows: list[tuple[str, int, float, float, float, float, int, int]] = []
    for dataset in datasets:
        values, by_seed = _collect(root, dataset, seed_set, args.w1_ack)
        missing = sorted(seed_set - set(by_seed))
        if len(values) < args.require_n:
            raise SystemExit(
                f"{dataset}: {len(values)} seeds < required {args.require_n}; "
                f"missing {missing}"
            )
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        budget_s4 = math.ceil(mean / 4)
        budget_s3 = math.ceil(mean / 3)
        output["datasets"][dataset] = {
            "n": len(values),
            "ebar_bp": mean,
            "sd": sd,
            "min": min(values),
            "max": max(values),
            "budget_s4": budget_s4,
            "budget_s3": budget_s3,
        }
        rows.append(
            (dataset, len(values), mean, sd, min(values), max(values), budget_s4, budget_s3)
        )

    print(f"{'dataset':10s} {'n':>3s} {'Ebar_BP':>9s} {'sd':>7s} "
          f"{'min':>7s} {'max':>7s} {'ceil/4':>7s} {'ceil/3':>7s}")
    for dataset, n, mean, sd, low, high, b4, b3 in rows:
        print(
            f"{dataset:10s} {n:3d} {mean:9.4f} {sd:7.4f} "
            f"{low:7.2f} {high:7.2f} {b4:7d} {b3:7d}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
