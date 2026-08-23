"""Builds the tidy table every camera-ready figure and table reads from.

    python scripts/build_tidy_table.py
    python scripts/build_tidy_table.py --out artifacts/tidy --no-traces
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

from src.plotting import tidy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/tidy"), help="output directory"
    )
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help="skip the NVML time-series reduction (much faster, fewer metrics)",
    )
    args = parser.parse_args()

    rows = tidy.build(with_traces=not args.no_traces)
    table_path = tidy.write(rows, args.out / "runs.csv")
    provenance_path = tidy.write_provenance(args.out / "provenance.csv")

    runs = {row.run_id for row in rows}
    by_phase = collections.Counter(row.phase for row in rows)
    missing = collections.Counter()
    for spec in tidy.SUMMARY_METRICS + tidy.TRACE_METRICS:
        present = {row.run_id for row in rows if row.metric == spec.key}
        if len(present) < len(runs):
            missing[spec.key] = len(runs) - len(present)

    print(f"wrote {table_path} ({len(rows)} rows, {len(runs)} runs)")
    print(f"wrote {provenance_path}")
    for phase, count in sorted(by_phase.items()):
        run_count = len({row.run_id for row in rows if row.phase == phase})
        print(f"  {phase:24s} {run_count:4d} runs  {count:6d} rows")
    if missing:
        print("metrics absent for some runs:")
        for key, count in missing.most_common():
            print(f"  {key:32s} missing on {count} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
