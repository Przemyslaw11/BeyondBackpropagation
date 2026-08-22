#!/usr/bin/env python3
"""Legacy single-experiment wrapper around the canonical CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beyond_backprop.cli.main import main as canonical_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Beyond Backpropagation experiment."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=("slurm", "local"), default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    command = ["experiment", "run", "--config", args.config]
    if args.backend is not None:
        command.extend(("--set", f"general.backend={args.backend}"))
    for override in args.overrides:
        command.extend(("--set", override))
    if args.artifact_dir is not None:
        command.extend(("--artifact-dir", args.artifact_dir))
    if args.dry_run:
        command.append("--dry-run")
    return canonical_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
