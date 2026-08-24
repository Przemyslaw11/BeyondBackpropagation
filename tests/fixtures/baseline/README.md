# Refactoring baseline

This directory records observations made before the canonical package migration.
These files are characterization data, not production configuration.

- Captured: 2026-08-22
- Branch: `main`
- Commit: `61f2bf424cdd14f02b146e474c41151de3c549ee`
- Existing staged user change: `.gitignore` adds `source_tex/`; it is intentionally preserved.

The repository currently has 19 backend-policy tests, approximately 8,143 Python
lines across `src`, `scripts`, and `tests`, and 44 YAML files under `configs/`.
