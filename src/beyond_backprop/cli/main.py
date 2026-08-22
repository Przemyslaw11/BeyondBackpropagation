"""Canonical command-line workflows with legacy-compatible validation commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import ConfigValidationError, ExperimentConfig, load_experiment_config
from ..runtime import get_execution_backend, resolve_device


def _config_arguments(
    command: argparse.ArgumentParser, *, dry_run: bool = False
) -> None:
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--base-config", default=Path("configs/base.yaml"), type=Path)
    command.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a resolved configuration value; may be repeated",
    )
    if dry_run:
        command.add_argument("--dry-run", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beyond-backprop")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate-config", "Validate a YAML configuration without loading data"),
        ("inspect-config", "Show the resolved configuration summary"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        _config_arguments(command, dry_run=True)

    experiment = subparsers.add_parser("experiment", help="Run experiments")
    experiment_subparsers = experiment.add_subparsers(
        dest="experiment_command", required=True
    )
    run = experiment_subparsers.add_parser("run", help="Run one experiment")
    _config_arguments(run, dry_run=True)
    run.add_argument("--artifact-dir", type=Path, default=None)

    tune = subparsers.add_parser("tune", help="Run or validate an Optuna study")
    _config_arguments(tune, dry_run=True)
    tune.add_argument("--output-dir", type=Path, default=None)
    tune.add_argument("--study-name", default=None)
    tune.add_argument("--n-trials", type=int, default=None)
    tune.add_argument("--backend", choices=("local", "slurm"), default=None)

    batch = subparsers.add_parser("batch", help="Run or validate a config directory")
    batch.add_argument("--config-dir", required=True, type=Path)
    batch.add_argument("--glob", default="**/*.yaml")
    batch.add_argument("--base-config", default=Path("configs/base.yaml"), type=Path)
    batch.add_argument("--set", dest="overrides", action="append", default=[])
    batch.add_argument("--dry-run", action="store_true")
    batch.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def _load(args: argparse.Namespace) -> ExperimentConfig:
    return load_experiment_config(args.config, args.base_config, args.overrides)


def _print_inspect(config: ExperimentConfig) -> None:
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


def _print_dry_run(config: ExperimentConfig) -> None:
    mapping = config.to_mapping()
    device = resolve_device(mapping)
    backend = get_execution_backend(mapping)
    print("dry_run: true")
    print(f"experiment_name: {config.experiment_name}")
    print(f"algorithm: {config.algorithm.value}")
    print(f"architecture: {config.architecture.value}")
    print(f"dataset: {config.dataset.value}")
    print(f"device: {device}")
    print(f"seed: {config.seed}")
    print(f"batch_size: {config.batch_size}")
    print(f"results_dir: {backend.resolve_results_dir(mapping)}")
    print(f"monitoring: {dict(config.monitoring)}")
    print(f"tracking: {dict(config.tracking)}")
    print(f"config_hash: {config.config_hash}")


def _run_one(config: ExperimentConfig, artifact_dir: Path | None = None) -> int:
    from ..training import ExperimentRunner

    result = ExperimentRunner(artifact_dir=artifact_dir).run(config)
    print(f"status: {result.status.value}")
    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
    return 0 if result.status.value == "succeeded" else 1


def _run_tuning(args: argparse.Namespace, config: ExperimentConfig) -> int:
    """Run the canonical tuning subsystem; the legacy script is only a shim."""
    from ..tuning import run_study

    if args.backend is not None:
        config = load_experiment_config(
            args.config,
            args.base_config,
            tuple(args.overrides) + (f"general.backend={args.backend}",),
        )
    mapping = config.to_mapping()
    output_dir = args.output_dir
    if output_dir is None:
        backend = get_execution_backend(mapping)
        output_dir = Path(backend.resolve_results_dir(mapping)) / "optuna"
    study_name = args.study_name or f"{config.experiment_name}_canonical"
    result = run_study(
        config,
        output_dir=output_dir,
        study_name=study_name,
        n_trials=args.n_trials,
    )
    print(f"study_name: {result.study_name}")
    print(f"trials: {len(result.trials)}")
    print(f"best_value: {result.best_value}")
    return 0 if result.best_value is not None else 1


def _batch_configs(directory: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.name != "base.yaml"
    )


def _run_batch(args: argparse.Namespace) -> int:
    configs = _batch_configs(args.config_dir, args.glob)
    if not configs:
        print(f"No YAML configurations found under {args.config_dir}", file=sys.stderr)
        return 2
    failures = 0
    for path in configs:
        try:
            config = load_experiment_config(path, args.base_config, args.overrides)
        except (ConfigValidationError, FileNotFoundError, OSError, ValueError) as exc:
            print(f"Configuration error in {path}: {exc}", file=sys.stderr)
            failures += 1
            continue
        if args.dry_run:
            print(f"validated: {path} ({config.config_hash})")
            continue
        failures += _run_one(config, args.output_dir / config.experiment_name)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"validate-config", "inspect-config"}:
            config = _load(args)
            if args.command == "validate-config":
                print(f"Valid configuration: {args.config}")
                print(f"config_hash: {config.config_hash}")
            else:
                _print_inspect(config)
            return 0

        if args.command == "batch":
            return _run_batch(args)

        config = _load(args)
        if args.command == "tune":
            if args.dry_run:
                _print_dry_run(config)
                print("tuning: validated; no study started")
                return 0
            return _run_tuning(args, config)

        if args.dry_run:
            _print_dry_run(config)
            return 0
        return _run_one(config, args.artifact_dir)
    except (ConfigValidationError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised by the console command
    raise SystemExit(main())
