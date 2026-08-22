#!/usr/bin/env python3
"""Legacy single-experiment wrapper around the canonical CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beyond_backprop.cli.main import main as canonical_main

# Kept as patchable compatibility names for callers that imported the legacy
# module and replaced its setup collaborators in tests or automation.
load_config = None
get_execution_backend = None
create_directory_if_not_exists = None
setup_logging = None
run_training = None


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
    if not hasattr(args, "dry_run"):
        return _legacy_main(args)
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


def _legacy_main(args: argparse.Namespace) -> int:
    """Preserve the old callable-module surface for embedded automation."""

    global load_config, get_execution_backend, create_directory_if_not_exists
    global setup_logging, run_training
    if load_config is None:
        from src.training.engine import run_training as legacy_run_training
        from src.utils.backend_policy import get_execution_backend as legacy_backend
        from src.utils.config_parser import load_config as legacy_load_config
        from src.utils.helpers import create_directory_if_not_exists as legacy_mkdir
        from src.utils.logging_utils import setup_logging as legacy_setup_logging

        load_config = legacy_load_config
        get_execution_backend = legacy_backend
        create_directory_if_not_exists = legacy_mkdir
        setup_logging = legacy_setup_logging
        run_training = legacy_run_training
    config = load_config(args.config)
    if getattr(args, "backend", None):
        config.setdefault("general", {})["backend"] = args.backend
    backend = get_execution_backend(config)
    name = config.get("experiment_name", Path(args.config).stem)
    log_file = backend.resolve_log_file(config, name)
    create_directory_if_not_exists(str(Path(log_file).parent))
    setup_logging(
        log_level=config.get("logging", {}).get("level", "INFO"), log_file=log_file
    )
    results = run_training(config, wandb_run=None)
    results.pop("codecarbon_emissions_gCO2e", None)
    return 1 if "error" in results else 0


if __name__ == "__main__":
    raise SystemExit(main())
