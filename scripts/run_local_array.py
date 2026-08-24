#!/usr/bin/env python3
"""Legacy local batch wrapper around the canonical batch command."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beyond_backprop.cli.main import main as canonical_main


def collect_configs(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    if args.config_dir:
        paths.extend(
            str(path)
            for path in sorted(Path(args.config_dir).rglob("*.yaml"))
            if path.name != "base.yaml"
        )
    if args.configs:
        paths.extend(args.configs)
    if args.glob:
        paths.extend(sorted(glob.glob(args.glob, recursive=True)))
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        resolved = str(Path(path).resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiment configs sequentially on the local backend."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config-dir")
    group.add_argument("--configs", nargs="+")
    group.add_argument("--glob")
    parser.add_argument("--backend", choices=("local", "slurm"), default="local")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = collect_configs(args)
    if args.max_configs is not None:
        paths = paths[: args.max_configs]
    if not paths:
        print("No config files found.", file=sys.stderr)
        return 2

    failures = 0
    for path in paths:
        command = [
            "experiment",
            "run",
            "--config",
            path,
            "--set",
            f"general.backend={args.backend}",
        ]
        for override in args.overrides:
            command.extend(("--set", override))
        if args.dry_run:
            command.append("--dry-run")
        failures += int(canonical_main(command) != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
