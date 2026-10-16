# HANDOFF — JPDC extension (Phase 0)

Updated: 2026-10-16. Branch: `jpdc-extension` (from `release/canonical-v2` @ `181ecc4`).

## State
- Phase 0.2 audit: DONE -> `AUDIT.md` (H1 confirmed and quantified; original-MF schedule
  mismatch documented; fairness matrix; open items).
- Phase 0.3a multiplier decomposition: DONE -> `scripts/jpdc/parse_run_logs.py`,
  outputs `jpdc_data/tidy/epochs_by_unit.csv`, `multiplier_decomposition.csv`.
- Phase 0.3b power fits: DONE -> `scripts/jpdc/fit_power_model.py`,
  output `jpdc_data/tidy/power_model_fits.csv`; cross-run fits recorded in AUDIT.md §6.
- Phase 0.3c arXiv headline trace: draft in AUDIT.md §5; needs the Nov-2025 arXiv v1
  protocol check against `jpdc_data/archive2025/bbp_experiments_2025.tar.gz`.
- Phase 0.4 (report to author): this summary + blocking questions below.

## Headline numbers (canonical Protocol A, 20 seeds)
- MF/BP unit-epoch multiplier: CIFAR-10 4.07, CIFAR-100 4.46, MNIST 3.67, Fashion 4.32.
  MF-Joint (joint, no detach): 0.99–1.32. => staging, not detachment, produces the
  multiplier; MF-Simul is the clean single-variable G1 test (and matches the original
  MF paper's schedule more closely than repo MF).
- Stage anatomy CIFAR-10: M0 49.5 (tail 20), W1_M1 99.9 (90% capped), W2_M2 40.4,
  W3_M3 21.5; BP = 32.95 to best + 19 tail.
- Power: MF mean 60.2 W @ 6.4% util; BP 67.7 W @ 13.7% util; cross-run
  P = 51.2 + 1.14·u pooled (R² 0.70).

## Data layout (local)
- `jpdc_data/run_logs/` — per-experiment logs (20 seeds each; M0 only logs per-epoch curves).
- `jpdc_data/monitoring/monitoring/` — 1028 NVML CSVs.
- `jpdc_data/tidy/` — runs_camera_ready.csv (19,944 rows), provenance.csv, plus the
  three derived tables above.
- `jpdc_data/archive2025/bbp_experiments_2025.tar.gz` — 2025-era plots/results/slurm_logs/wandb.
- Git bundles and pre-rewrite history exports live in the OS temp dir; re-extract from
  `../*.bundle` if purged.
- Cluster: `athena:~/BeyondBackpropagation/results/{runs,monitoring,optuna}`, logs,
  `wandb/` (837 MB offline); archives under `/net/pr2/.../plgspyra_bbp_archive/`.
  Helper pattern: scp script -> `ssh athena 'bash -l /tmp/x.sh'`; never train on login nodes.

## Next actions (ordered)
1. Confirm 2025 protocol trace (0.3c) from the archive + arXiv v1; write the two lines.
2. Prepare PREREG.md draft (MF-Simul definition, arms, n=20, seeds, decision thresholds
   A/B/C, ES quantity = final-readout M_L validation CE, unit tests: gradient locality +
   trajectory equivalence, drift controls).
3. Then Phase 1 implementation: add MF-Simul arm to codebase + tests; tiny CPU smoke
   test; only then pilot on Athena (n=5) and full runs (n=20).
4. Budget mapping: 4 configs (BP / MF-Joint / MF-Simul / MF-Seq) x CIFAR-10 + maybe
   CIFAR-100, n=20 — estimate A100-hours from existing trace durations once budget known.

## Resolved questions (author, 2026-10-16)
1. A100 budget: not a planning constraint for now; still report measured usage per session.
2. SLURM: agent may submit jobs directly (pilots included).
3. Power / exclusive-GPU policy: no special replication needed for G1.
4. Original-MF-vs-repo-MF schedule discrepancy: OK to state, as a neutral observation.
5. New analysis code stays in `scripts/jpdc/`.

## Rules of engagement (unchanged)
- Label claims [Paper]/[KB]/[Planning]/[Verified]/[Assumption]; never invent numbers.
- No changes to pre-registered protocol after seeing results (amend PREREG.md with
  timestamps only); pilots are exploratory, final runs are decisive.
- Keep comparability: same codebase, `nn.Linear` primitives, monitor, statistics.
- Report per-session: summary, exact numbers, code/jobs changed, budget, risks, next 3 actions.
