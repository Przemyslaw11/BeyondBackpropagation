# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Rationale for individual decisions lives in
`docs/refactoring-decisions.md`; the overall migration narrative lives in
`docs/refactoring-report.md`.

## [Unreleased]

### Removed (Phase E)

- Legacy re-export shims `src/utils/*`, `src/data_utils/*`,
  `src/algorithms/*`, `src/architectures/*`, `src/training/`, `src/tuning/`
  and the migration helper `scripts/_migrate_b2.py` (decision MIG-004).
  Historical `src.*` import paths stop working; use `beyond_backprop.*` and
  the canonical CLI commands instead.

### Added (Phase E)

- `make install-dev` target installing the `[dev]` extra and activating
  pre-commit hooks.
- Widened ruff rule set (`SIM`, `C4`, `N`) with mechanical fixes and targeted,
  documented `noqa`s.
- `CONTRIBUTING.md` with environment setup (including the macOS hidden-`.pth`
  pitfall), Make-target reference, commit conventions, and the
  behavior-preservation rule.
- This changelog.

### Fixed (Phase E)

- Dropped the pinned `python_version` from `[tool.mypy]` so type checking runs
  under the active interpreter (local CPython 3.12 works again; CI on 3.10 is
  unchanged).

## [0.1.0] - Phases A–D (2026-08)

Canonical refactoring foundation, implemented across Phases A–D:

- Canonical package `beyond_backprop` with typed configuration loading
  (deterministic base-plus-experiment merge, strict validation, explicit CLI
  overrides).
- Algorithm adapters and registry for BP, FF, CaFo, and MF delegating to
  dependency-light math modules guarded by numerical regression tests.
- Monitoring protocols with lazy NVML/CodeCarbon adapters and explicit no-op
  snapshots; tracking backends (W&B/local/no-op).
- Shared early stopping, checkpoint manager, tuning subsystem, and artifact
  persistence (`config.resolved.yaml`, `metadata.json`, `metrics.json`,
  `history.csv`, `summary.json`).
- Canonical CLI: `python -m beyond_backprop experiment run|tune|batch|
  validate-config|inspect-config` with `--dry-run` / `--set k=v`.
- Test suite reorganized into `tests/{unit,integration,regression,smoke,
  fixtures}`.
- SLURM array manifest generated exactly once per array job with atomic
  publication, removing regeneration races and per-task revalidation
  (decision OPS-003).
- Retirement of the dual legacy execution path (decision MIG-003); wrapper
  scripts translate to canonical CLI commands.