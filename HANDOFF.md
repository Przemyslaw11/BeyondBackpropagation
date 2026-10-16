# HANDOFF — JPDC extension (Phase 0 + Phase 1 start)

Updated: 2026-10-17. Branch: `jpdc-extension` (from `release/canonical-v2` @ `181ecc4`).

## State
- Phase 0.2 audit: DONE -> `AUDIT.md` (H1 confirmed and quantified; original-MF schedule
  mismatch documented; fairness matrix; open items).
- Phase 0.3a multiplier decomposition: DONE -> `scripts/jpdc/parse_run_logs.py`,
  outputs `jpdc_data/tidy/epochs_by_unit.csv`, `multiplier_decomposition.csv`.
- Phase 0.3b power fits: DONE -> `scripts/jpdc/fit_power_model.py`,
  output `jpdc_data/tidy/power_model_fits.csv`; cross-run fits recorded in AUDIT.md §6.
- Phase 0.3c arXiv headline trace: DONE -> AUDIT.md §5 (repro configs from commit
  `869d7b0` are the authoritative source; `archive2025` tarball holds no arXiv v1 PDF).
  Only the "34% faster" pairing remains [Assumption], low priority.
- Phase 1 MF-Simul implementation: DONE -> `src/baselines/bp_ds.py` (`detach_between_layers`
  flag, `train/evaluate_mf_simul_model`), registered in `src/algorithms|baselines/__init__.py`,
  config `configs/mf_simul/cifar10_mlp_3x2000.yaml`, tests `tests/test_mf_simul.py`
  (3 tests) + fairness invariants extended. All 78 repo tests pass locally; CPU smoke
  run (MNIST, 3 epochs) finished at 94.76% test accuracy.
- PREREG.md: drafted 2026-10-17 (arms, thresholds A/B/C, seeds 42-61 n=20, ES quantity,
  unit tests, drift controls, amendments log). Freeze at the decisive commit + record
  hashes before inspecting decisive results.
- Next: port the ladder Optuna driver from `main` (removed on release branch) into
  `jpdc-extension`, add `mfsimul_*_tune` configs (50 TPE trials, same procedure as
  `ladder_mfjoint_*`), tune MF-Simul per dataset, then pilot n=5 (seeds 42-46, CIFAR-10).

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
1. ~~Port ladder tuning machinery~~ DONE (`90e7f95`).
2. ~~Tune MF-Simul per dataset~~ RUNNING (job 3371585, studies `ladder_mfsimul_*`).
   When done: freeze `config_override` from
   `results/optuna/ladder_mfsimul_*_best_params.yaml` into `configs/mf_simul/*`.
3. Prepare and submit pilot n=5 (seeds 42-46, CIFAR-10, four arms) on Athena via SLURM;
   one-page summary for the ~29 Oct author meeting; pilots never enter decisive tables.
   Needs a pilot array SLURM script + a small `run_array`-style driver (port from
   `scripts/slurm_scripts/run_ladder_array.slurm` history if needed).
4. Decisive runs n=20 (seeds 42-61, CIFAR-10 + CIFAR-100, four arms, one commit) after
   PREREG freeze; `scripts/jpdc/analyze_g1.py` @ `8a4ce41` reduces raw results; then the
   G1 verdict (due 8 Nov 2026).
5. Budget mapping: estimate A100-hours from monitoring trace durations (report sacct +
   NVML Wh per session; budget itself is not a planning constraint).

## Resolved questions (author, 2026-10-16)
1. A100 budget: not a planning constraint for now; still report measured usage per session.
2. SLURM: agent may submit jobs directly (pilots included).
3. Power / exclusive-GPU policy: no special replication needed for G1.
4. Original-MF-vs-repo-MF schedule discrepancy: OK to state, as a neutral observation.
5. New analysis code stays in `scripts/jpdc/`.

## Session log
- 2026-10-16 (c76d2ad): phase-0 audit, epoch multiplier decomposition, power model fits.
- 2026-10-17 (284ccac): MF-Simul implemented (bp_ds.py + registrations + config + tests);
  78/78 repo tests pass; local CPU smoke run OK (MNIST 3 epochs, 94.76% test acc, 57 s);
  PREREG.md drafted; scripts/jpdc updated for the `mf_simul` group (parse + power fit);
  AUDIT §5 final (0.3c).
- 2026-10-17 (90e7f95): ladder Optuna tuning stack ported from pre-removal history
  (`src/tuning/*`, `scripts/run_optuna_search.py` + MF_SIMUL dispatch/formatter,
  `configs/tuning/mfsimul_{cifar10,cifar100}_mlp_3x2000_tune.yaml`,
  `scripts/slurm_scripts/mfsimul_tune.slurm`). Local 3-trial smoke of the full tuning
  loop OK (study resume via load_if_exists verified; best-params YAML carries
  `config_override`). Cluster sync method: `git bundle create /tmp/jpdc-extension.bundle
  181ecc4..jpdc-extension` -> scp -> `git fetch /tmp/jpdc-extension.bundle
  jpdc-extension:jpdc-extension` -> checkout (no GitHub push needed).
  Tuning job submitted: `sbatch scripts/slurm_scripts/mfsimul_tune.slurm` = array
  3371585 (2 tasks: CIFAR-10, CIFAR-100; 50 TPE trials each; studies
  `ladder_mfsimul_*`; 12 h wall cap, resumable). Cluster unit tests 12/12 OK.
- 2026-10-17 (ddc195a + next): pilot array script prepared
  (`scripts/slurm_scripts/g1_pilot.slurm`, 4 arms x seeds 42-46 CIFAR-10; submit after
  tuning freezes `configs/mf_simul/*`). Canonical run JSONs and logs archived before any
  overwrite: cluster `results/archive_pre_g1_20261017/` (all `*_run.log` + `runs/`),
  local `jpdc_data/canonical_runs_v2/runs/` (2.8M JSONs). Tuning job 3371585 still
  queued (Priority) at last check.

## Rules of engagement (unchanged)
- Label claims [Paper]/[KB]/[Planning]/[Verified]/[Assumption]; never invent numbers.
- No changes to pre-registered protocol after seeing results (amend PREREG.md with
  timestamps only); pilots are exploratory, final runs are decisive.
- Keep comparability: same codebase, `nn.Linear` primitives, monitor, statistics.
- Report per-session: summary, exact numbers, code/jobs changed, budget, risks, next 3 actions.
