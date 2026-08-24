# Contributing

Thanks for contributing! This repository backs a published paper
(`arXiv:2509.19063v1`), so behavior preservation outranks style: when in doubt,
change less.

## Environment setup

- Python >= 3.10.
- Runtime install: `make install` (editable install).
- Contributor install: `make install-dev` — installs the `[dev]` extra
  (ruff, mypy, pytest, pytest-cov, pre-commit, tqdm, types-PyYAML) and
  activates the pre-commit hooks.

### macOS note: invisible `.pth` files

If `pip install -e .` appears to be ignored by the interpreter
(`ModuleNotFoundError: beyond_backprop` outside pytest), the editable-install
`.pth` files may carry macOS's `hidden` (`UF_HIDDEN`) flag. CPython >= 3.12.11
skips hidden `.pth` files in `site.py`. Check and repair:

```bash
ls -lO .venv/**/site-packages/*.pth     # look for "hidden" in the flags column
chflags nohidden .venv/**/site-packages/*.pth
```

This project lives under `~/Desktop`, where iCloud Desktop sync has been
observed to re-apply the flag periodically — re-run the two commands above if
the symptom returns.

## Make targets

| Target | Purpose |
| --- | --- |
| `make install` | Install the package in editable mode |
| `make install-dev` | Install package with dev tools and activate pre-commit hooks |
| `make format` | Format canonical package and tests |
| `make format-check` | Check formatting |
| `make lint` | Run Ruff lint checks |
| `make typecheck` | Run mypy on the canonical package |
| `make test` | Complete test suite |
| `make test-fast` | CPU/non-slow tests (CI-equivalent selection) |
| `make test-cov` | Tests with coverage report |
| `make check` | format-check + lint + typecheck + fast tests |
| `make validate-config` | Validate `$(CONFIG)` without loading data |
| `make smoke` | Tiny offline end-to-end smoke test |
| `make clean` | Remove generated build and cache files |

## Commit and PR conventions

- Conventional-Commits-style prefixes: `feat:`, `fix:`, `refactor:`,
  `test:`, `docs:`, `build:`, `chore:` (append `!` for breaking changes).
- One concern per commit; every commit must pass `make check`.
- Keep diffs small and revertible; do not mix formatting with logic changes.

## Behavior preservation (critical)

Any change that could affect reported numbers — accuracy, time, energy,
memory, GFLOPs, CO2e — must be called out explicitly in the PR description
with before/after reasoning. Numerical logic changes require a regression test
first; record intentional corrections in
`docs/refactoring-decisions.md`. Out-of-scope-by-default surfaces include loss
functions, gradient math, optimizer steps, seeding, data splits, early
stopping, metric computation, YAML keys/defaults, CLI flags/exit codes, log-line
formats, and result-artifact schemas.

## Further reading

- `docs/development.md` — development workflow details.
- `docs/reproducibility.md` — run provenance and reproducibility guarantees.
- `docs/refactoring-report.md` — architecture overview and canonical command
  examples.
- `docs/refactoring-decisions.md` — decision register with scientific-impact
  rationale.