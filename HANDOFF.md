# HANDOFF — JPDC extension (Phase 0 + Phase 1 start)

Updated: 2026-10-22. Branch: `jpdc-extension` (from `release/canonical-v2` @ `181ecc4`).

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
  (3 tests) + fairness invariants extended. Latest suite: 112 passed, 13 skipped.
- PREREG.md: drafted 2026-10-17 (arms, thresholds A/B/C, seeds 42-61 n=20, ES quantity,
  unit tests, drift controls, amendments log). G1 is complete; verdict B (schedule
  artifact). W1 is pre-registered (`PREREG-phase2.md`) and queued; see below.

## 2026-10-21 zero-GPU work package (no GPU-hours consumed)
- **W0b/W0c/W0d/W0e resolved.** M0 is dead compute: 3.99 Wh / 22.9% (C10) and
  3.57 Wh / 22.8% (C100) of MF-Seq energy; W1_M1 is cap-bound 90%/55%; per-stage
  sums equal JSON 40/40. BP is not cap-truncated (full 19-epoch tail, val loss
  rises 40/40, accuracy = PPAM +0.00 pp). Era drift is power + seconds-per-pass,
  no co-tenancy: C100 energy gaps +8.3% BP, +4.1% MF-Joint, +6.9% MF-Seq; BP C10
  is −2.2%. See `jpdc_data/g1/{w0b,w0c,w0d,w0e}/`.
- **Independent G1 verification:** separate reducer, different bootstrap seed, no
  imports from the production analysis; all headline accuracy/pass/energy deltas
  reproduce. 91/96 checks within tolerance; 5 flags are ≤0.25 passes from stored
  1-decimal rounding. `jpdc_data/verification/`.
- **DSS Pareto data:** `jpdc_data/dss/` has Protocol A (all four arms, G1) and
  Protocol B (PPAM BP + MF-Seq only; MF-Joint/MF-Simul explicitly not measured).
  Protocol B reproduces −19.9%/−12.5% energy and −27.5%/−24.6% memory.
- **W4 power model (reframed, 2026-10-21):** the pre-G1 fit is retained only for
  its slope, P = 51.97 [51.33, 52.61] + 1.085 [1.036, 1.133]·u, R²=0.684. W4 now
  reports a **within-session E_arm/E_BP ratio** after one-point intercept
  calibration on BP seed 42 per dataset. C10 mean ratio bias ≤1.1% but per-seed
  MAE 4.0–4.7%; C100 bias +3.5 to +4.4%, MAE 5.4–6.4%. Absolute cross-era use is
  retired; see `jpdc_data/g1/w4_ratio/`. The old absolute table is
  `w4_absolute_legacy.md`.
- **W3 cut communication (redone, 2026-10-21):** the old 69.1 GB/pass is
  **memory traffic** (parameter re-reads + all layer activations), not cut
  communication. New unit: 0.36 GB per epoch per direction per cut
  (45,000×2,000×4 B). BP = 2.16 GB/pass, MF-Simul = 1.08, MF-Seq useful =
  0.36 per prefix cut per stage epoch. Bytes-to-stopping: BP 114.4/94.8 GB
  (C10/C100), MF-Simul 62.9/38.6, MF-Seq useful 30.0/26.5, MF-Seq as
  implemented 83.4/74.4 (dead M0 forward 1.08 GB/pass). Cached-prefix bound saves
  22.1%/25.3% of useful cut bytes; augmentation makes it a counterfactual.
  `scripts/jpdc/w3_cut_communication.py`, `jpdc_data/g1/w3_cut/`.
- **Stage-0 audit:** production `train_mf_matrix_only` calls W1..WL in train and
  validation although M0 reads only a_0. The dead forward is 99.6%/96.0% of M0
  FLOPs, ≥3.97/3.43 Wh of waste. Ex-M0 multipliers are **ESTIMATES** (remove
  the whole stage under the current implementation): energy 2.69×/2.92×,
  passes 3.05×/3.35×, time 3.03×/3.35×; `ERRATA_PPAM.md` row 6,
  `w3_stage0_audit.md`. H-W3d pilot: Spearman **ρ = −0.543** over 6 cells,
  bootstrap 95% CI [−1.000, +0.800], descriptive p=0.266; sign only, small-n
  caveat, no threshold. Output `w3_cut_h_w3d.csv`.
- **Queue/account audit:** jobs 3386582/3386586 are on
  `plglscclass26-gpu-a100` (fairshare 0.000, 44/48 A100 nodes unavailable).
  `plgoncotherapy2-gpu-a100` is authorized for new W1 submissions (team
  `plggfastsim`; fairshare 0.728, 0 usage); `--test-only` estimates 12 Nov vs
  19 Nov for a 4 h GPU job. Existing class-account jobs are not migrated.
  `plgrid-now` (1 h cap) is **not used** because its published policy is
  interactive-only. No submission: `QUEUE_RISK_W1.md`, `W1_CRITICAL_PATH.md`.
- **W1 manifest split (2026-10-22):** frozen core = 200 tasks
  (`w1_family.slurm`: BP, MF-Joint, MF-Simul, MF-Seq, nom0); frozen control =
  40 tasks (`w1_family_controls.slurm`: BP-bestval); BP-fixed = 40
  (`w1_bp_fixed.slurm`: BP-sched lr/wd, 211/191 epochs, no early stopping);
  Seq-Global = 40 (`w1_seqglobal.slurm`); Protocol B = 80
  (`w1_controls_protocol_b.slurm`); BP-sched = 40 (`w1_bp_sched.slurm`).
  **Base total 440**; cap-300 conditional. HPO fallback is in PREREG §9
  (Seq **25 Nov**, BP-sched **2 Dec**, ≥20/dataset; Seq fallback = MF-Seq lr/wd,
  BP-sched fallback = core-BP lr/wd; deadline amendment 2026-10-22). Estimated ~66.5 GPU-h confirm +
  ~51 tuning; details `W1_MANIFEST_ADDITIONS.md`, `W1_CRITICAL_PATH.md`.
  No threshold change. BP-long retired because BP never reaches the 100-epoch cap.
- **W2 (redesigned 2026-10-22):** four-GPU training superseded by a zero-GPU
  link-rate model (`scripts/jpdc/w2_link_rate_model.py`, output
  `jpdc_data/g1/w2_link/`) at 100 Mbit/s, 1 Gbit/s, 10 Gbit/s, using measured
  passes, per-step compute time, per-run energy and cut bytes, plus a short 1 MB
  NCCL P2P diagnostic (`scripts/jpdc/nccl_p2p_benchmark.py`,
  `scripts/slurm_scripts/w2_nccl_p2p.slurm`; prepared, not submitted).
  Pilot: MF-Seq/BP total time is 0.36×/0.39× at 100 Mbit/s and 2.99×/3.30× at
  10 Gbit/s (C10/C100); the slow-link ordering reverses. Seq-Global is flagged
  as a planning proxy until W1. No 4-GPU budget.
- **W6 budget-confound audit (2026-10-22):** headline MF-Seq 205.8× BP is
  confounded (BP gets 420 updates/layer, MF-Seq 420 total = 140/stage).
  Controls: BP-140 103.2×, MF-Seq-1260 **1.71×** (clean 420/layer control),
  Seq-Global-1260 **0.23×** (not clean: 420–1260 cumulative updates/layer).
  The toy remains exploratory and must not be used to argue staging helps.

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
2. ~~Tune MF-Simul per dataset~~ DONE (job 3371597; 50/50 trials each; best val acc
   60.02 C10 trial 21 / 31.50 C100 trial 37). Frozen into `configs/mf_simul/*` (C10
   lr 2.338e-4, wd 2.660e-6, λ 0.0207; C100 lr 3.079e-4, wd 1.922e-5, λ 0.2166);
   best-params YAMLs copied to `jpdc_data/optuna/`; PREREG §2 clarified + §10 row.
3. ~~Prepare pilot array script~~ DONE (`8e99dbb`, `scripts/slurm_scripts/g1_pilot.slurm`);
   pilot ran (3371868). One-page pilot summary → `jpdc_data/pilot/`; pilots never enter
   decisive tables.
4. ~~Decisive runs n=20~~ DONE (jobs 3372435/3373251/3373266/3376200; 160/160 valid run
   JSONs, seeds 42-61, four arms both datasets, one commit). Reduced with
   `scripts/jpdc/analyze_g1.py` (@ `328b573`) -> **G1 verdict = B (schedule artifact)**:
   - C100: MF-Simul - BP = **-0.27 pp** [-0.50, -0.03], p=0.076; passes ratio R=0.81;
     contemporaneous MF-Seq - BP = **+5.04 pp** [+4.82, +5.25], p<0.001.
   - C10 (secondary): MF-Simul - BP = -0.86 pp [-1.16, -0.56]; MF-Seq +3.90 pp.
   - MF-Joint C100 = +1.86 pp over BP => staging (not detachment) carries the whole
     MF advantage; detachment alone is ~BP or slightly below.
   Outputs: `results/g1/g1_summary.md` + `g1_per_seed.csv` (cluster), `jpdc_data/g1/`.
5. ~~G1 report + Phase-2 planning~~ DONE as drafts: `PREREG-phase2.md` (W1 ladder, decision
   rules frozen for review), `G1_ATTRIBUTION_SUMMARY.md` (author one-pager),
   `ERRATA_PPAM.md` (claim-level corrections).
6. Budget mapping: ~**28.7 A100-GPU-hours** for the whole G1 chain (tuning 2x50 trials,
   pilot 20, decisive 160 + incident reruns 112; sacct 2026-10-17). NVML Wh per run in
   `g1_per_seed.csv` (final C100: BP 4.48, MF-Joint 5.11, MF-Simul 3.55, MF-Seq 16.67 Wh).
7. ~~Pull repair job 3376703 + refresh G1 tables~~ DONE (2026-10-21, login nodes restored):
   all 12 repaired runs were complete (Oct 17 14:34-14:47, before the lockout). Pulled
   160/160 JSONs + 160 monitoring CSVs + 248 task `.out` files. `analyze_g1` and `w0a`
   regenerated: **160/160 energies first-party** (`trace_from = json` in all cells),
   accuracy verdict unchanged (B). Final paired C100 energies: MF-Joint +0.62
   [+0.47, +0.79], MF-Simul -0.93 [-1.13, -0.71], MF-Seq +12.18 [+11.69, +12.69] Wh.
   W0c cap anatomy (`w0c_cap_anatomy.py`, new): MF-Seq W1_M1 cap-bound 90% C10 / 55%
   C100 (PPAM 90%/50%), M0 23.4%/23.2% of passes, per-stage sums = JSON in 40/40 runs.
   W0e refreshed against the same tree; one-pager/errata updated.
8. **QUEUED (2026-10-22, no GPU-hours consumed):**
   - Seq-Global tuning: job **3386582** (2 tasks, 50 trials each, 24 h wall;
     studies `ladder_mfglobal_{cifar10,cifar100}_mlp_3x2000_tune`). If it runs
     before 25 Nov, freeze the best trial with ≥20 completed per dataset;
     otherwise use the MF-Seq fallback. Record the branch in §9 and commit.
     BP-sched tuning is separate on the 2 Dec deadline; its fallback is core-BP
     lr/wd, not MF-Seq's.
   - H-W1cap Step 1: job **3386586** (5 tasks, C100 cap-300 seeds 42-46).
   - Both remain on `plglscclass26-gpu-a100`, pending `ReqNodeNotAvail`; they are
     not migrated. New W1 work uses `plgoncotherapy2-gpu-a100` (team `plggfastsim`).
   - W1 is split: `w1_submit.sh core` (200), `controls` (40 BP-bestval),
     `fixed` (40 BP-fixed), `seqglobal` (40), `protocolb` (80), `sched` (40).
     No W1 job submitted; author gate applies.
9. Freeze record to write in `PREREG-phase2.md` §9: code-freeze commit hash,
   SHA-256 of `scripts/jpdc/analyze_w1.py`, and the frozen-config manifest hash
   from `scripts/jpdc/w1_config_hash.py` (12 core/BP-bestval configs; HPO-gated
   configs excluded). The analysis reducer implements the 2x2, H-W1a..d, the
   C100 Holm family, paired Wilcoxon/10k bootstrap, co-primary costs, and the
   descriptive probe tables; tests include a planted effect and missing seeds.
10. Account decision done (2026-10-22): new W1 submissions use
    `plgoncotherapy2-gpu-a100`; existing pending jobs stay on the class account.
    Do not cancel or migrate them.
11. HPO fallback approved and committed (`f877b92`); deadlines amended on
    2026-10-22 to Seq **25 Nov** and BP-sched **2 Dec** (outage, not data).
    Do not insert any different fallback wording.
12. Queue starts 12/19/26 Nov are mapped in `W1_CRITICAL_PATH.md`; cut order:
    W3 comms first, W4 second, **never W1 core**. No `plgrid-now`.
13. DSS data are ready in `jpdc_data/dss/`; build the figure after W1, keeping
    Protocol A and B on separate axes (era drift is real).
14. Pre-W1 freeze package: secondary endpoints + stage epochs, BP-fixed, HPO
    deadline amendment (BP-sched fallback corrected to core-BP lr/wd), W2 link
    model, H-W3d sign/caveat, W6 budget audit, `analyze_w1.py` + tests, and
    `INSTRUMENTATION_EVIDENCE.md`. The cluster checkout is still at `813430e`;
    sync to the final frozen commit before any submission.

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
  local `jpdc_data/canonical_runs_v2/runs/` (2.8M JSONs). Tuning job 3371585 failed in
  1 s on both tasks (`ModuleNotFoundError: No module named 'src'` — sbatch script lacked
  PYTHONPATH); fixed and resubmitted as 3371597 (pending, Priority).
- 2026-10-17 (a9fd8b8): `g1_pilot.slurm` + tuning fix committed; cluster synced.
- 2026-10-17 (next): `g1_decisive.slurm` prepared (8 configs x 20 seeds = 160 runs;
  submit after the §10 freeze). Run order fixed: tuning (3371597) -> freeze MF-Simul
  params -> pilot (g1_pilot.slurm) -> decisive (g1_decisive.slurm) -> analyze_g1.
- 2026-10-17 (next, tuning run): 3371597 finished, studies 50/50 trials each (best val
  acc 60.02/31.50; MF-Joint reference 59.70/31.86). MF-Simul hyperparameters frozen
  into both `configs/mf_simul/*` configs (C100 config created); PREREG §2 λ wording
  clarified (tuned like MF-Joint; λ=1.0 stays the unit-test anchor) + §10 row; 12
   fairness/MF-Simul tests pass. Pilot submitted next.
- 2026-10-17 (00db091): pilot array `g1_pilot.slurm` submitted = job 3371868 (20 tasks).
  Partition congested (13 nodes fail*, 10 down*); SLURM start estimate 07:35 CEST.
  `scripts/jpdc/summarize_pilot.py` added (exploratory C10 pilot one-pager; fixture-tested).
- 2026-10-17 (bb75bb7): pilot finished (20/20 COMPLETED by 05:06). Exploratory C10 n=5:
  BP 62.69+-0.26 (49.4 passes, ~4 Wh) | MF-Joint 62.45 (51.8) | MF-Simul 61.70+-0.30
  (59.0, ratio 1.19, ~5 Wh) | MF-Seq 66.39 (219.0, ratio 4.43, ~17 Wh). Bug found and
  fixed: run logs are append-only, so pass sums double-counted canonical+pilot blocks;
  `analyze_g1.run_passes` now takes the latest run per seed (fixture-unchanged, hash
  re-recorded in PREREG §10). Summary file: `results/jpdc_g1_pilot_summary.md` (cluster)
  + `jpdc_data/pilot/`. Decisive array submitted next.
- 2026-10-17 (next): decisive array `g1_decisive.slurm` submitted = job 3372435
  (160 tasks; 8 configs x seeds 42-61). Analysis next: `python3 -m scripts.jpdc.analyze_g1
  --results-dir results --out-dir results/g1` on the cluster.
- 2026-10-17 (INCIDENT): 3372435 ran 73 OK / 87 FAILED. Root cause chain: the pilot and
  decisive sbatch scripts did NOT export `BBP_DISABLE_CHECKPOINTS=1`, so BP runs saved
  per-seed checkpoints (~150-200 MB each, `checkpoints/` grew to 3.0 GB) -> $HOME hit its
  10 GiB quota at ~10:05 -> site quota watchdog killed 87 running tasks mid-epoch
  (exit 0:53) and sbatch rejected resubmissions. Fixes: deleted `checkpoints/*`
  (HOME back to 70.8%), both g1 scripts now export `BBP_DISABLE_CHECKPOINTS=1` (also
  required by PREREG §3: final weights scored) and redirect `WANDB_DIR` to `$SCRATCH`.
  Consequence: today's BP results (all pilot + 17 decisive completions) were scored with
  best-val-accuracy restore = protocol deviation -> ALL 40 BP decisive tasks must rerun;
  the 56 completed MF*/MF-Joint/MF-Simul results are valid (their configs have
  `checkpoint_dir: null`). Rerun = failed IDs + BP-completed IDs (104 tasks), same commit.
- 2026-10-17 (6a06fa4): incident fix committed + synced. Reruns submitted: 3373251
  (87 failed IDs) + 3373266 (17 BP-completed IDs). BP reruns double-checked against the
  fixed script (`BBP_DISABLE_CHECKPOINTS=1`, wandb -> `$SCRATCH/bbp_wandb`). MF*/MF-Joint/
   MF-Simul completions from 3372435 remain valid.
- 2026-10-17 (328b573): five decisive run JSONs + three MF JSONs were found truncated
  (quota-window partial writes, tasks had exited 0) — 8 tasks rerun (job 3376200);
  parser hardened against glued log lines and empty NVML CSVs. Final reduction:
  160/160 runs, n=20 per arm/dataset. **G1 verdict = B (schedule artifact)**.
  Session budget: ~28.7 A100-GPU-h across all jobs (incl. incident reruns).
- 2026-10-17/19 (W0 session, commits d90c7be..HEAD): post-verdict artifact audit found
  12/160 decisive run JSONs declaring NVML traces that never landed (quota window); the
  old glob fallback had silently used September traces for those energies. PREREG §10
  amended: passes from run-JSON `epochs_completed` (shared log lost lines: e.g. MF-Seq
  C100 seed42 W1_M1 100 per log vs 81 per `.out`+JSON), energy strictly from each JSON's
  `monitoring_csv_path`. Repair job 3376703 = the 12 (arm, seed) reruns at the decisive
  commit (per-seed log files, forced `WANDB_DIR=$SCRATCH/bbp_wandb`, quota precheck).
  New analysis: `w0a_energy_memory.py` (cost table), `w0e_drift.py` (G1 vs PPAM regime:
  no co-tenancy anywhere, drift ≤±5% arm-specific), `w0b_m0_energy.py` (M0 = 22.9%/22.8%
  of MF-Seq Wh), `parse_task_outs.py` (per-task `.out` anatomy; log-independent).
  W1 implemented: `src/algorithms/mf_global.py` (Seq-Global) + 5 gradient tests +
  `test_m0_dead_stage.py` + `configs/mf_global/`, `configs/mf_nom0/*`; CPU smoke OK
  (MNIST 98.08%). Drafts: `PREREG-phase2.md`, `G1_ATTRIBUTION_SUMMARY.md`,
  `ERRATA_PPAM.md`. **Incident: Athena login nodes restricted** (site security patch,
  admin notice 2026-10-17 evening) — 3376703 keeps running on compute nodes; pulls and
  new submissions blocked until access returns.
- 2026-10-21: Athena login restored (no jobs lost). Repair job 3376703 had finished Oct 17
  14:47, before the lockout; pulled it plus all stored `BBP_G1*.out` (248 files). Fresh
  reduction: 160/160 runs with their own NVML trace; G1 verdict B unchanged; C100 energy
  deltas -0.93/+0.62/+12.18 Wh. New `w0c_cap_anatomy.py`: MF-Seq cap anatomy (90%/55%
  W1_M1, M0 ~23%) matches PPAM; 40/40 stage sums equal JSON. `parse_task_outs` extended
  for the repair-script header. `G1_ATTRIBUTION_SUMMARY.md`/`ERRATA_PPAM.md` refreshed.
  Note: 56 of the 160 decisive task `.out` files were lost when the quota watchdog killed
  job 3372435 (buffers not flushed) — passes/energy never depended on them.
- 2026-10-21 (zero-GPU package): W0e drift decomposition finalized (power + sec/pass, no
  co-tenancy); W0d BP audit added; independent G1 reducer agrees on every headline number;
  DSS Protocol A/B Pareto data generated; W4 pre-G1-only power fit under-predicts G1 by
  3–7%; W3 hook accounting completed (M0 20% of bytes, cache bound 11–12%); Athena
  account/queue audit found 44/48 nodes unavailable and a second allocation. W1 manifest
  (14 configs/280 tasks) confirmed; `PREREG-W2-W3.md` draft + `QUEUE_RISK_W1.md` memo added.
  No GPU jobs submitted.
- 2026-10-21 (six-item package): W3 redone as cut-boundary communication (BP 2.16 GB/pass,
  old 69.1 GB/pass relabelled memory traffic); stage-0 production path audited (dead forward
  99.6%/96.0% of M0 FLOPs; ex-M0 multipliers 2.69×/2.92× energy); W1 controls added
  (BP-sched, BP-bestval, Protocol-B MF-Seq/nom0; base 440 tasks, ~109 GPU-h; later
  superseded by BP-fixed and ~118 GPU-h) with an
  in-memory best-state restore and scheduler-aware BP tuning; C100 test provenance fixed
  (Wilcoxon p=0.0759 vs paired-t CI [-0.525,-0.008], p=0.0437); W4 reframed as a calibrated
  ratio model; submission variants and 4×5 h resumable tuning chain prepared, not submitted.
  `plgrid-now` rejected on policy (interactive-only). No GPU jobs submitted or cancelled.

- 2026-10-22 (pre-submission continuation): fallback HPO rule added to
  `PREREG-phase2.md` §9 at commit `f877b92` before any tuning trial completed
  (Seq 4 Nov / BP-sched 11 Nov; ≥20 trials/dataset; the BP-sched fallback was
  later corrected to core-BP lr/wd, PREREG §9, 2026-10-22; disclose).
  W1 split into independent arrays; `w1_submit.sh` modes
  `core|controls|fixed|seqglobal|protocolb|sched`. Author authorized
  `plgoncotherapy2-gpu-a100` (team `plggfastsim`) for new W1 work; the two pending
  class-account jobs were not migrated. `W1_CRITICAL_PATH.md` maps 12/19/26 Nov
  starts (cut W3 comms first, W4 second, never W1 core). W3 per-cut and stage-0
  numbers and W4 ratio reframing are confirmed in `STATUS_2026-10-22.md`. W6 toy
  CPU numerics are exploratory only (`scripts/jpdc/w6_toy_sim.py`,
  `jpdc_data/g1/w6_toy/`). No W1 job submitted.
- 2026-10-22 (pre-W1 freeze package, this commit): the HPO deadlines were moved
  to **25 Nov / 2 Dec** for the outage (not data). BP-long was retired and
  replaced by **BP-fixed** (`w1_bp_fixed.slurm`, BP-sched lr/wd, fixed
  211/191 epochs, no early stopping); frozen controls are 40 BP-bestval and
  BP-fixed is a separate 40-task array, so the base is 440 and confirm cost is
  ~66.5 GPU-h. The freeze instrumentation (`secondary_endpoints.py` + stage
  epochs) is integrated after the NVML window and unit-tested to leave final
  weights unchanged. W2 was redesigned as the zero-GPU link-rate model plus a
  1 MB NCCL P2P diagnostic (prepared, not submitted). H-W3d now reports the sign
  of rho with a small-n caveat and ex-M0 multipliers are labelled estimates.
  The W6 toy has budget-confound controls: MF-Seq-1260 is 1.71× BP-420 (clean
  equal-update), so the headline 205.8× is not a staging effect. `W1_SYNC_AND_SUBMIT.md`
  contains the bundle sync and one-command blocks; none executed. The W1 reducer
  and instrumentation evidence are in `scripts/jpdc/analyze_w1.py`,
  `tests/test_analyze_w1.py`, and `INSTRUMENTATION_EVIDENCE.md`. Tests 112
  passed, 13 skipped; Ruff and `bash -n` clean.

## Rules of engagement (unchanged)
- Label claims [Paper]/[KB]/[Planning]/[Verified]/[Assumption]; never invent numbers.
- No changes to pre-registered protocol after seeing results (amend PREREG.md with
  timestamps only); pilots are exploratory, final runs are decisive.
- Keep comparability: same codebase, `nn.Linear` primitives, monitor, statistics.
- Report per-session: summary, exact numbers, code/jobs changed, budget, risks, next 3 actions.
