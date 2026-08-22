"""Small, side-effect-free configuration commands for the migration boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import ConfigValidationError, load_experiment_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beyond-backprop")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate-config", "Validate a YAML configuration without loading data"),
        ("inspect-config", "Show the resolved configuration summary"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument(
            "--base-config", default=Path("configs/base.yaml"), type=Path
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_experiment_config(args.config, args.base_config)
    except (ConfigValidationError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.command == "validate-config":
        print(f"Valid configuration: {args.config}")
        print(f"config_hash: {config.config_hash}")
        return 0

    print(f"experiment_name: {config.experiment_name}")
    print(f"algorithm: {config.algorithm.value}")
    print(f"architecture: {config.architecture.value}")
    print(f"dataset: {config.dataset.value}")
    print(f"backend: {config.backend.value}")
    print(f"device: {config.device}")
    print(f"seed: {config.seed}")
    print(f"batch_size: {config.batch_size}")
    print(f"monitoring: {dict(config.monitoring)}")
    print(f"tracking: {dict(config.tracking)}")
    print(f"config_hash: {config.config_hash}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the console command
    raise SystemExit(main())
