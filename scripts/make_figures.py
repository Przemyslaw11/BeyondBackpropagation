"""Regenerates every figure from ``results/tidy/runs.csv``.

Nothing here reads Weights & Biases, and nothing reads ``results/`` except the
NVML trace CSVs the tidy table points at. Run ``scripts/build_tidy_table.py``
first if the table is missing or stale.
"""

import argparse
import hashlib
from pathlib import Path
from typing import Callable, Dict

from src.plotting import figures, style
from src.plotting.tidy import Table

PAPER_FIGURES: Dict[str, Callable] = {
    "ladder_waterfall": figures.ladder_waterfall,
    "ladder_power": figures.ladder_power,
    "frontier": figures.time_memory_frontier,
    "forest": figures.equivalence_forest,
    "ff_resource": figures.ff_resource,
    "ff_cost": figures.ff_cost,
    "cafo": figures.cafo_profile,
    "mf_hardware": figures.mf_hardware,
    "mf_cost": figures.mf_bp_cost_curves,
}

#: Not for the paper. These answer questions the reviewers raised about the
#: protocol rather than claims the paper makes.
DIAGNOSTIC_FIGURES: Dict[str, Callable] = {
    "diag_early_stopping": figures.diag_early_stopping,
    "diag_cache_strategy": figures.diag_cache_strategy,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="results/tidy/runs.csv")
    parser.add_argument(
        "--out", default="plots/generated"
    )
    parser.add_argument("--diagnostics-out", default="results/figures/diagnostics")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Build only these figures. One of: "
        f"{', '.join(list(PAPER_FIGURES) + list(DIAGNOSTIC_FIGURES))}",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    diagnostics_dir = Path(args.diagnostics_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    style.apply_style()
    table = Table.load(Path(args.table))
    # Superseded protocols stay in the table so the provenance is auditable, but
    # nothing superseded reaches a figure.
    live = table.where(superseded="")
    dropped = len(table.distinct("run_id")) - len(live.distinct("run_id"))
    print(
        f"{args.table}: {len(table.rows)} rows, {len(table.distinct('run_id'))} runs; "
        f"{dropped} superseded runs excluded"
    )
    table = live

    targets = [(n, f, out_dir) for n, f in PAPER_FIGURES.items()]
    targets += [(n, f, diagnostics_dir) for n, f in DIAGNOSTIC_FIGURES.items()]
    if args.only:
        targets = [t for t in targets if t[0] in args.only]

    for name, build, directory in targets:
        path = build(table, directory)
        size_kb = path.stat().st_size / 1024
        print(f"  {name:20s} -> {path}  ({size_kb:.0f} KiB)  {_sha256(path)[:16]}")
    print(f"done: {len(targets)} figures")


if __name__ == "__main__":
    main()
