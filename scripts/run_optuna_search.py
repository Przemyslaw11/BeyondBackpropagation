#!/usr/bin/env python3
"""Legacy argument-translation shim for the canonical tuning command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beyond_backprop.cli.main import main as canonical_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical hyperparameter tuning.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--backend", choices=("local", "slurm"), default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = ["tune", "--config", args.config]
    for flag, value in (
        ("--output-dir", args.output_dir),
        ("--study-name", args.study_name),
        ("--n-trials", args.n_trials),
        ("--backend", args.backend),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    return canonical_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
