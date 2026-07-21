#!/usr/bin/env python3
"""Local batch runner: run multiple experiment configs sequentially.

Replicates the Slurm array workflow (scripts/slurm_scripts/run_array.slurm)
for local macOS execution using the local execution backend (MPS/CPU).

Usage:
  python scripts/run_local_array.py --config-dir configs/bp_baselines/
  python scripts/run_local_array.py --configs configs/ff/mnist.yaml configs/mf/mnist.yaml
  python scripts/run_local_array.py --glob "configs/**/*.yaml"
"""

import argparse
import glob as glob_module
import logging
import os
import sys
import time

from dotenv import load_dotenv

from src.utils.backend_policy import get_execution_backend
from src.utils.config_parser import load_config
from src.utils.helpers import create_directory_if_not_exists
from src.utils.logging_utils import setup_logging


def collect_configs(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    if args.config_dir:
        for root, _dirs, files in os.walk(args.config_dir):
            for f in files:
                if f.endswith((".yaml", ".yml")):
                    paths.append(os.path.join(root, f))
        paths.sort()
    if args.configs:
        paths.extend(args.configs)
    if args.glob:
        paths.extend(sorted(glob_module.glob(args.glob, recursive=True)))
    seen = set()
    unique = []
    for p in paths:
        absp = os.path.abspath(p)
        if absp not in seen:
            seen.add(absp)
            unique.append(p)
    return unique


def run_experiment(config_path: str, backend_name: str) -> dict:
    import pprint

    from src.training.engine import run_training

    config = load_config(config_path)
    config.setdefault("general", {})["backend"] = backend_name
    backend = get_execution_backend(config)

    exp_name = config.get("experiment_name", os.path.splitext(os.path.basename(config_path))[0])
    log_file = backend.resolve_log_file(config, exp_name)
    log_dir = os.path.dirname(log_file)
    create_directory_if_not_exists(log_dir)

    log_level = config.get("logging", {}).get("level", "INFO")
    setup_logging(log_level=log_level, log_file=log_file)

    log = logging.getLogger(__name__)
    log.info("=" * 60)
    log.info("Config: %s", config_path)
    log.info("Backend: %s, Device: %s", backend.name, backend.resolve_device("auto"))
    log.info("=" * 60)

    results = run_training(config, wandb_run=None)
    results.pop("codecarbon_emissions_kgCO2e", None)
    log.info("Results for %s:", config_path)
    for line in pprint.pformat(results).split("\n"):
        log.info(line)
    return results


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run experiment configs sequentially on local backend.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory containing YAML config files (non-recursive walk).",
    )
    group.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="One or more YAML config file paths.",
    )
    group.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Glob pattern for config files (e.g. 'configs/**/*.yaml').",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "slurm"],
        default="local",
        help="Execution backend (default: local).",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of configs to run (for testing).",
    )
    args = parser.parse_args()

    config_paths = collect_configs(args)
    if not config_paths:
        print("No config files found.", file=sys.stderr)
        sys.exit(1)

    if args.max_configs:
        config_paths = config_paths[: args.max_configs]

    print(f"Found {len(config_paths)} config(s). Backend: {args.backend}")
    print("-" * 60)

    summary: list[dict] = []
    global_start = time.time()
    for idx, config_path in enumerate(config_paths, start=1):
        print(f"\n[{idx}/{len(config_paths)}] Running: {config_path}")
        try:
            results = run_experiment(config_path, args.backend)
            summary.append({"config": config_path, "status": "ok", "results": results})
        except Exception as e:
            logging.getLogger(__name__).critical("Experiment failed: %s", e, exc_info=True)
            summary.append({"config": config_path, "status": "failed", "error": str(e)})

    total_duration = time.time() - global_start
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    ok_count = sum(1 for s in summary if s["status"] == "ok")
    fail_count = sum(1 for s in summary if s["status"] == "failed")
    print(f"Total: {len(summary)}  |  Succeeded: {ok_count}  |  Failed: {fail_count}")
    print(f"Total duration: {total_duration:.1f}s")
    if fail_count > 0:
        print("\nFailed configs:")
        for s in summary:
            if s["status"] == "failed":
                print(f"  - {s['config']}: {s['error']}")
    sys.exit(1 if fail_count > 0 else 0)


if __name__ == "__main__":
    main()
