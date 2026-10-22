# PREREG-phase2.md — JPDC extension, W1: what produces the MF-Seq edge?

Freeze rule: this file is committed **before** any W1 confirmatory run. Changes after that point
are date-stamped additions to §9 only. Pilots (n=5) are exploratory and never enter decisive
tables. Same discipline as `PREREG.md`: every arm runs contemporaneously at n=20, paired seeds
42–61, on one commit; config diffs between arms are limited to algorithm name and hyperparameter
values.

## 1. Question

G1 (`PREREG.md`) settled that **detachment alone is ~BP** (MF-Simul − BP = −0.27 pp on CIFAR-100,
p = 0.076) while the staged MF-Seq keeps **+5.04 pp** at 4.4× passes and 3.7× energy
[Verified, PREREG.md §10 amendment of 2026-10-17]. What remains is attribution: *which
component of the staged schedule* produces that edge and cost?

Three candidates, separable by one missing 2×2 cell:

- **S — schedule:** fresh optimizer + independent early stopping per stage, greedy per-stage
  objective (MF-Seq vs the joint arms).
- **L — frozen prefix:** each stage trains against a prefix whose weights never move
  (layer-locality of credit assignment). MF-Seq confounds S and L; **Seq-Global** removes L
  while keeping S.
- **Z — the dead M0 stage:** phase 1 trains M_0 on the raw input; nothing downstream or at
  evaluation reads it (pinned by `tests/test_m0_dead_stage.py`), yet it owns ≈23% of MF-Seq's
  passes and energy [Verified, `scripts/jpdc/w0b_m0_energy.py`].

## 2. Arms

| arm | schedule | prefix gradients | readout | optimizer | ES scope | status |
|---|---|---|---|---|---|---|
| BP | 1 joint unit | full backprop | output layer | AdamW | global, Protocol A | existing |
| MF-Joint | 1 joint unit | full backprop through aux heads | M_L | Adam | global | existing |
| MF-Simul | 1 joint unit | detached inputs per layer | M_L | Adam | global | existing |
| MF-Seq | M0 + 3 stages | frozen prefix | per-stage M_i, final M_L | Adam | per stage | existing |
| **Seq-Global** | M0 + 3 stages | **trainable prefix** | per-stage M_i, final M_L | Adam | per stage | `src/algorithms/mf_global.py` |
| **MF-Seq-nom0** | 3 stages (no M0) | frozen prefix | per-stage M_i, final M_L | Adam | per stage | `configs/mf_nom0/*` (flag exists) |

Seq-Global, exact (`src/algorithms/mf_global.py`): stage `s` unfreezes `W_1..W_s` and `M_s`,
and one backward pass per batch reaches all of them through the live prefix;
`W_{s+1}..W_L` and every other `M_j` stay frozen (unit-tested in
`tests/test_mf_global_gradients.py`). Stage count, per-stage ES, and the readout mirror MF-Seq.

2×2: {MF-Joint, MF-Simul} × {MF-Seq, Seq-Global} separate *where the loss is attached*
(M_L vs every stage) from *whether the prefix moves*. MF-Seq-nom0 and a cap-300 MF-Seq
(§3, H-W1c/H-W1cap) test the two cost levers.

## 3. Hypotheses and frozen decision rules

All primary thresholds on **CIFAR-100** (G1's decisive lens) at n=20, seeds 42–61; CIFAR-10 is
the consistency check. Nothing here can retroactively change G1's verdict.

- **H-W1a (locality carries the edge):** Seq-Global − MF-Seq ≤ −2.0 pp on C100, and
  Seq-Global − MF-Simul ≤ +1.0 pp. Reading: without the frozen prefix the staged schedule
  collapses toward the joint arms; "MF = local learning" survives.
- **H-W1b (schedule carries the edge):** Seq-Global − MF-Seq ≥ −1.0 pp on C100. Reading: the
  edge is the stopping/schedule structure, not locality; G1's interpretation sharpens to
  "staging, with or without frozen prefixes".
- **H-W1c (M0 is pure overhead):** MF-Seq-nom0 − MF-Seq ≥ −0.5 pp on C100 while saving ≥15% of
  passes **and** ≥15% of energy. If instead accuracy drops ≥1.5 pp, M0 is functionally
  load-bearing and the dead-stage claim is withdrawn (energy share ≈23% [Verified] sets the
  saving side; the accuracy side is the test).
- **H-W1cap (cap binding, two-step):** Step 1 pilot (n=5, exploratory, seeds 42–46): MF-Seq at
  `max_epochs: 300`. If the pilot shows a mean accuracy gain ≥ +0.5 pp on C100 or mean W1_M1
  epochs > 130, run the full n=20 cap-300 arm (Step 2) and report the head-to-head; otherwise
  the 100-epoch cap stands and the cap-sensitivity table (W0c) is reported as the defense.
- **H-W1d (strong-BP control, amended 2026-10-22; §9):** **BP-fixed**, using the
  BP-sched lr/wd and a cosine schedule over the fixed budget, no early stopping, and a fixed budget of
  **211 epochs (CIFAR-10) / 191 epochs (CIFAR-100)** = `round(mean MF-Seq total passes per
  dataset from G1)`; final weights are scored. If BP-fixed − MF-Seq ≥ −1.0 pp on C100 at
  ~1× the passes of MF-Seq, the "compute-advantage" framing of MF is retired. BP-long
  (`max_epochs: 300`, early stopping) is retired because BP never reaches the 100-epoch cap
  [Verified, W0d], so it duplicated core BP. This control is now a same-pass accuracy check,
  not a truncation check.

Thresholds for A/B-style readings are the ones above; intermediate outcomes are reported as
intermediate with both readings and no third relabeling of G1.

## 4. Endpoints and statistics

- **Primary:** test accuracy of the final readout, final weights, single evaluation, no
  checkpoint restore (`BBP_DISABLE_CHECKPOINTS=1`).
- **Co-primary (cost):** passes (JSON `epochs_completed`), NVML energy (Wh, strict
  `monitoring_csv_path`), wall-clock. Per-arm means with 95% CI; paired-by-seed differences
  with Wilcoxon + 10k bootstrap, Holm within the family {H-W1a, H-W1b, H-W1c}; H-W1cap and
  H-W1d are reported with raw CIs (controls).
- **Secondary (diagnostic only; never in a decision rule):** final full-train loss/accuracy
  in eval mode, final validation loss/accuracy, per-stage epochs and stopping epoch,
  per-layer linear-probe test accuracy on `a_1..a_L`, local-head test accuracy where heads
  exist, and entropy effective rank of each `a_i` on a fixed 5,000-row probe set. The probe
  is fitted on the validation split (no augmentation) and scored on the first 5,000 test
  rows; the same ridge procedure is used for every arm. These evaluations run after the
  NVML monitor has stopped, record their own seconds in the run JSON, and must not change
  final weights. They do not enter the Holm family or any W1 verdict.
- **Analysis source of truth:** per-run JSON + own trace; the shared append-only run log is
  never used for numbers (2026-10-17 amendment, `PREREG.md` §10).

## 5. Materials and run plan

- Hyperparameters: Seq-Global and MF-Seq-nom0 inherit the tuned MF-Seq values
  (`configs/mf/cifar10_mlp_3x2000.yaml`, `configs/mf/cifar100_mlp_3x2000.yaml`) so the arms
  differ in one thing only. **Decision required before any W1 confirmatory run:** either
  (a) keep inherited values and record that choice here, or (b) give Seq-Global its own
  50-trial ladder-Optuna run (same protocol as MF-Simul) and freeze the result in
  `configs/mf_global/*` first. Option (b) is the parity-maximizing default unless the author
  prefers speed.
- Runs: 6 main arms × 2 datasets × 20 seeds = 240; controls are specified in the
  2026-10-21 §9 amendment (base total 440 tasks after controls; cap-300 remains conditional).
  One commit for the whole family; verify `src/` identical across arms.
- Cluster hygiene: per-seed run logs via generated configs (shared-log contention observed
  2026-10-17), forced `WANDB_DIR` on scratch, HOME-quota precheck, per-task `.out` retained.
- Budget: reported per session from `sacct`/NVML; not a planning constraint ([author, 2026-10-16]).

## 6. What this phase will not claim

- No new datasets/architectures inside W1 (CNN and communication accounting are separate
  workstreams with their own prereg addenda before their confirmatory runs).
- Staggered-MF-Simul hybrids and non-first-order optimizers are deferred: they need their own
  hypotheses; they cannot be added to W1 after freeze.
- MNIST/Fashion results, tuning curves, and pilots are exploratory and never pooled with W1.

## 7. Deliverables and gates

- One-page W1 summary for the author meeting (join G1's existing one-pager), including the
  2×2 table, the M0 overhead line, and cap/BP-control verdicts.
- `ERRATA_PPAM.md` gains the W0/W1 rows it is owed (15 Nov DSS data deadline).
- G2 scope gate: 20 Dec 2026 — select which of W2–W6 (comms accounting, cost model, CNN,
  theory numerics) enter the JPDC claims. Experiments frozen 8 Jan 2027.

## 8. Freeze

Commit this file before scheduling any W1 confirmatory job. Record the commit hash, the
`analyze_w1.py` hash, and the frozen-config manifest hash from
`scripts/jpdc/w1_config_hash.py` in §9 at launch. Until §5's HPO decision is recorded, W1
jobs may only be pilots.

## 9. Amendments log

| date | change | by |
|---|---|---|
| 2026-10-17 | PREREG-phase2 drafted (W1 attribution ladder: Seq-Global implemented + unit-tested; mf_nom0 configs added; CPU smoke of Seq-Global OK). No W1 run scheduled yet | agent |
| 2026-10-21 | §5 HPO decision recorded: **option (b)**, parity-maximizing. Seq-Global gets its own 50-trial ladder-Optuna study per dataset before any confirmatory run (`configs/tuning/mfglobal_*`, studies `ladder_mfglobal_{cifar10,cifar100}_mlp_3x2000_tune`); search space is the MF-family breadth lr/wd (no auxiliary weight exists — each stage has its own local loss). Best params frozen into `configs/mf_global/*`; the tuning commit hash and best-params YAMLs are recorded here when the studies finish. | agent |
| 2026-10-21 | **W1 config manifest confirmed (no protocol change).** The 14 configs are the six main arms × 2 datasets — BP, MF-Joint, MF-Simul, MF-Seq, Seq-Global, MF-Seq-nom0 — plus the H-W1d BP-long control × 2 datasets: 7 arms × 2 datasets × 20 seeds = **280 tasks** in `scripts/slurm_scripts/w1_family.slurm`. BP-long is the seventh arm/control, not one of the six main arms; `mf_nom0` is the "MF-Seq without stage 0" arm. Cap-300 remains a conditional two-step arm and is **not** in the 280-task array. **Superseded later this date by the 440-task row below** (controls split into their own arrays; no threshold change). | agent |
| 2026-10-21 | **W2/W3 prereg addendum drafted** as a separate file, `PREREG-W2-W3.md` (status: draft, not frozen; no confirmatory W2/W3 run scheduled). This keeps the W1 freeze untouched, as §7 requires separate addenda for W2–W6. The G1 W3 byte accounting in `jpdc_data/g1/w3/` is labelled an exploratory engineering pilot, not confirmatory evidence. | agent |
| 2026-10-21 | **W3 redone as cut-boundary communication** (`scripts/jpdc/w3_cut_communication.py`, `jpdc_data/g1/w3_cut/`). The unit is one activation tensor crossing one layer cut: 45,000×2,000×4 B = 0.36 GB per epoch per direction per cut. BP = 2.16 GB/pass (forward + backward at 3 cuts); MF-Simul = 1.08; MF-Seq useful = 0.36 per prefix cut per stage epoch. The old 69.1 GB/pass is **relabelled memory traffic**, not communication: it counted 19.95 GB forward parameter reads and 39.90 GB backward parameter-gradient reads plus all activation reads/writes. The old table is retained only under the legacy heading. Cached-prefix and augmentation cases are separate; CIFAR augmentation makes cached-prefix a counterfactual. | agent |
| 2026-10-21 | **Stage-0 audit.** Production `train_mf_matrix_only` was instrumented on a tiny model; W1 and W2 are called in both the two training batches and the validation batch although M0 reads only a_0. On the canonical 3×2000 MLP the dead forward is 99.6% (C10) / 96.0% (C100) of M0-stage FLOPs, a lower bound of 3.97 / 3.43 Wh. Ex-M0 multipliers (remove the whole stage; `ERRATA_PPAM.md`): energy 3.45×→2.69× (C10), 3.72×→2.92× (C100); passes 3.99×→3.05×, 4.36×→3.35×; wall time 3.93×→3.03×, 4.33×→3.35×. The G1 M0 energy share is 21.9%/21.4% (W0b's 22.9%/22.8% is of the PPAM-era trace total). | agent |
| 2026-10-21 | **W1 additions after the six core arms (controls only; no threshold or Holm change).** (a) **BP-sched**: fixed 100-epoch cosine schedule, no early stopping, 50 lr/wd trials per dataset, 40 confirm tasks; the Optuna BP objective now honours `training.scheduler` so tuning and confirmation use the same schedule. (b) **BP-bestval**: in-memory best-validation-accuracy restore, no disk checkpoint, 0 new tuning trials, 40 tasks. (c) **MF-Seq-nom0 Protocol B**: `S=L=3`, `ceil(Ebar_BP/3)` per stage (planning 18/15), 40 tasks. (d) **Contemporaneous Protocol B**: MF-Seq `S=4`, `ceil(Ebar_BP/4)` (planning 14/11), 40 tasks; BP Protocol B is the core BP arm, not rerun. Controls total 160 tasks; BP-sched adds 40 after its study, for a **revised base of 240 core + 200 controls = 440 tasks** (cap-300 still conditional). Core arrays launch first. Equal disclosed tuning: Seq-Global 100 trials, BP-sched 100 trials; other controls inherit frozen values. Budgets are frozen from the actual core-BP mean before controls run; G1 planning values are 14/11 for MF-Seq Protocol B and 18/15 for nom0 Protocol B. | agent |
| 2026-10-21 | **Reporting hygiene for C100 MF-Simul vs BP.** The p=0.0759 is the pre-registered two-sided paired Wilcoxon signed-rank test (exact, n=20, W=57). The 95% interval is a percentile bootstrap of the mean. The paired t-test gives −0.2665 pp [−0.5247, −0.0083], p=0.0437; this is reported as a sensitivity check, not a replacement. The interval/test disagreement is an estimand difference (mean vs signed ranks), not an arithmetic error; verdict B is unchanged. | agent |
| 2026-10-21 | **W4 reframed as within-session ratio prediction** (`scripts/jpdc/w4_power_extrapolation.py`, `jpdc_data/g1/w4_ratio/`). Absolute cross-era energy predictions are retired. The pre-G1 slope is combined with a one-point intercept calibration on BP seed 42 per dataset; the endpoint is the paired E_arm/E_BP ratio. C10 mean ratio bias ≤1.1% but per-seed MAE 4.0–4.7%; C100 bias +3.5 to +4.4%, MAE 5.4–6.4%. Calibration-seed sensitivity is reported. Do not use across eras without a fresh same-session BP calibration. | agent |
| 2026-10-21 | **Queue/account preparation, no W1 submission.** Default account remains `plglscclass26-gpu-a100`; account change is one line in `scripts/slurm_scripts/w1_submit.sh`. Prepared core/controls/BP-sched arrays and 5-hour resumable Seq-Global tuning segments. `plgrid-now` is **not** used: the published Cyfronet definition restricts it to interactive jobs (1 h, 1 job), not batch tuning; Athena's current docs do not document a separate policy. No job was submitted or cancelled under this amendment. | agent |
| 2026-10-22 | **Fallback HPO rule, approved by the author before any tuning trial completed.** Seq-Global tuning deadline: **4 Nov 2026 23:59 CET**. BP-sched tuning deadline: **11 Nov 2026 23:59 CET**. At each deadline, freeze the best-validation trial among completed trials if **≥20 trials are completed per dataset**; otherwise reuse MF-Seq's tuned lr/wd for that dataset. Disclose which branch was used, per dataset and per arm. This rule changes no W1 outcome, threshold, seed, or Holm family. Job 3386582 had consumed 0 GPU-hours and no tuning trial had completed when this row was committed. | author |
| 2026-10-22 | **W1 manifest split for independent submission (no threshold/Holm change).** Frozen core = BP, MF-Joint, MF-Simul, MF-Seq, MF-Seq-nom0 (200 tasks, `w1_family.slurm`); frozen controls = BP-long, BP-bestval (80, `w1_family_controls.slurm`); Seq-Global (40, `w1_seqglobal.slurm`) waits on the 4 Nov HPO branch; Protocol B = MF-Seq + nom0 (80, `w1_controls_protocol_b.slurm`) waits on the actual core-BP `Ebar_BP`; BP-sched (40, `w1_bp_sched.slurm`) waits on the 11 Nov HPO branch. Base total remains 440; cap-300 remains conditional. `w1_submit.sh` exposes `core|controls|seqglobal|protocolb|sched`. **Superseded later on the same date by the BP-fixed and HPO-deadline rows below** (controls 40 BP-bestval + 40 BP-fixed; Seq 25 Nov; BP-sched 2 Dec). | agent |
| 2026-10-22 | **Account decision by the author.** New W1 submissions use `plgoncotherapy2-gpu-a100` (team `plggfastsim`). Existing pending jobs 3386582/3386586 remain on `plglscclass26-gpu-a100`; they are not cancelled or migrated. No W1 job was submitted. | author |
| 2026-10-22 | **W6 toy numerics scoped as exploratory.** A CPU-only deep-linear toy compares BP, MF-Simul, MF-Seq and Seq-Global on one synthetic task (`scripts/jpdc/w6_toy_sim.py`, output `jpdc_data/g1/w6_toy/`). No W6 preregistration exists; results are labelled exploratory and are not pooled with W1–W4 or used as a hypothesis test. | agent |
| 2026-10-22 | **Freeze-commit instrumentation (diagnostic only; no threshold, seed, or Holm change).** Every W1 run JSON now also records, as exploratory secondary endpoints: final full-train loss/accuracy in eval mode with the eval transform; final validation loss/accuracy; the primary test loss/accuracy; per-layer linear-probe test accuracy on frozen `a_1..a_L` (ridge probe fitted on the fixed 5,000-row validation split, scored on the first 5,000 test rows); local-head test accuracy for every existing `M_i`; per-stage epochs and stopping epoch; and the entropy effective rank of each `a_i` on the fixed 5,000-row probe set. The computation runs after the NVML monitor has stopped, records its own seconds, and is unit-tested not to change training weights (same seed, same final state). | agent |
| 2026-10-22 | **H-W1d amended before any W1 result: BP-long is replaced by BP-fixed.** BP-long duplicated core BP because BP never reaches the 100-epoch cap. BP-fixed uses the BP-sched hyperparameters/cosine schedule, no early stopping, and fixed epochs **211 (C10) / 191 (C100)** = `round(mean MF-Seq total passes per dataset from G1)`; final weights are scored. Task count is unchanged: frozen controls are now BP-bestval only (40, `w1_family_controls.slurm`), BP-fixed is 40 (`w1_bp_fixed.slurm`), and the 440-task base is unchanged. BP-fixed is gated on the same BP-sched HPO branch and refuses to run while `configs/bp_sched/*` contains `PLACEHOLDER`. | agent |
| 2026-10-22 | **HPO deadlines moved for the A100 outage, not for data (before any tuning trial completed).** Seq-Global deadline: **25 Nov 2026 23:59 CET**. BP-sched deadline: **2 Dec 2026 23:59 CET**. The ≥20-completed-trials-per-dataset rule and the MF-Seq lr/wd fallback are unchanged; the branch must be recorded here and committed before the corresponding confirm array. Reason: jobs 3386582/3386586 remained `PENDING` with 0 GPU-hours and `ReqNodeNotAvail`; no trial had completed. This row supersedes the 4 Nov/11 Nov dates in the earlier fallback row. | author |
| 2026-10-22 | **BP-sched fallback corrected before any tuning trial completed; no outcome change.** The earlier fallback sentence is superseded for **BP-sched only**. If fewer than 20 trials per dataset are completed by **2 Dec 2026 23:59 CET**, BP-sched uses the **core BP** tuned lr/wd with the fixed cosine schedule (CIFAR-10: `lr=7.67408805303038e-05`, `wd=3.67072980229754e-05`; CIFAR-100: `lr=6.681923669912236e-05`, `wd=2.9979680868782076e-06`, both from `configs/bp_baselines/*`), not MF-Seq's lr/wd. BP-fixed inherits the same branch. Seq-Global's fallback remains MF-Seq's tuned lr/wd (`configs/mf/*`). Jobs 3386582/3386586 were rechecked on 2026-10-22 and remained `PENDING` with `Elapsed=00:00:00`, so no tuning trial had completed. This changes no W1 outcome, threshold, seed, task count, or Holm family. | author |
| 2026-10-22 | **Analysis implementation clarifications fixed before the freeze; no outcome definition changed.** (1) The wall-clock endpoint is `training_duration_sec`, so the post-training probes are outside the cost window; `total_run_duration_sec` is retained descriptively. (2) H-W1a and H-W1c are conjunctions: their p-value is the conservative intersection-union maximum of their component paired tests; the declared family remains exactly {H-W1a, H-W1b, H-W1c}. (3) Holm is applied on CIFAR-100 only; CIFAR-10 repeats the same thresholds as a consistency check with raw p-values and cannot be labelled confirmatory support. (4) A verdict requires all 20 paired seeds; missing seeds are listed and never imputed; an arm with fewer than 20 is `incomplete`/`not run`, not supported. (5) The 2x2 effects are paired-by-seed factorial contrasts on accuracy and each co-primary cost; H-W1d's ~1x pass condition is reported as the observed pass ratio, with no extra threshold. (6) Secondary probe/instrumentation tables are descriptive only and never enter Holm or a verdict. | agent |
| 2026-10-22 | **Pre-W1 freeze record.** Code-freeze commit: **`16ff4c44044571006ea977a8fb714a2f76837c85`**. `scripts/jpdc/analyze_w1.py` SHA-256: **`c424a0eebe81e8e48ec3e3ad5371e9343e6a7650e0ca9d4221c04c74a5a61ff1`**. Frozen-config manifest SHA-256 from `scripts/jpdc/w1_config_hash.py`: **`76f188472a6fb0ab481185b5446c66c1b047371a1933582b36e6f80e65c39d80`** (12 files: core BP/MF-Joint/MF-Simul/MF-Seq/nom0 plus BP-bestval, two datasets each; HPO-gated `bp_sched`, `mf_global`, and Protocol-B planning configs excluded until their dated branches). This §9 row is a metadata-only commit directly following the code-freeze commit; the code and config bytes are identical to the recorded commit. No W1 job was submitted, and no pending job was cancelled or migrated. Instrumentation inertness/overhead evidence is in `INSTRUMENTATION_EVIDENCE.md`; tests 112 passed, 13 skipped; Ruff and `bash -n` clean. | agent |
| 2026-10-22 | **Scheduling-order only, no protocol change.** The core and BP-bestval arrays were arm-major (`CONFIG_INDEX = INDEX / NUM_SEEDS`): tasks 1–20 were all BP/CIFAR-10, so paired arms would have started on different nodes and days and G1's measured 8% machine drift would enter the arm differences. The array index is now seed-major (`SEED_INDEX = INDEX / NUM_CONFIGS`) and the core config list groups CIFAR-10 before CIFAR-100, so each (seed, dataset) cell occupies five adjacent indices and each seed a contiguous ten-index block. `w1_family.slurm` and `w1_family_controls.slurm` also echo `SLURMD_NODENAME` and an ISO start timestamp per task, so node and start time are in every `.out` (no `src/` change). No config, seed, arm, task count, throttle or threshold changed. Commit **`7932875033e06aef729148d76f2070c381e4d16f`**. `analyze_w1.py` and the frozen-config manifest re-verified at their §9 values after the change. | agent |
| 2026-10-22 | **Submission plumbing only, no protocol change.** `w1_submit.sh` gains the `schedtune` case that its usage line already advertised: it submits `bp_sched_tune.slurm` (2 tasks, 50 trials per dataset). The path reads `configs/tuning/bp_sched_*` only and therefore has no `configs/bp_sched/*` placeholder gate; that gate still blocks `sched` and `fixed`. Same commit **`7932875033e06aef729148d76f2070c381e4d16f`**. No frozen quantity changed. | agent |
| 2026-10-22 | **Administrative note — account approved (not a protocol change).** The author confirmed in session that Dzwinel has approved charging **`plgoncotherapy2-gpu-a100`** (team `plggfastsim`) for the W1 family. Recorded as "author-confirmed in session". Pending jobs 3386582/3386586 remain on `plglscclass26-gpu-a100` and were not cancelled, held, migrated or requeued. | author-confirmed in session |
| 2026-10-22 | **Tag and launch record.** Annotated tag **`w1-freeze`** → **`16ff4c4`**, the code-freeze commit itself (not `815e002`, which only adds the §9 record). Annotated tag **`w1-launch`** → the commit carrying this row, i.e. `7932875` plus this metadata edit; the cluster checkout is moved to `w1-launch`. `git diff --stat w1-freeze w1-launch -- src/ scripts/jpdc/analyze_w1.py configs/` must be empty, and the three §9 hashes were re-verified on the cluster before submission. Future blocks re-check the same three hashes against `w1-freeze` and record the result here. | agent |
| 2026-10-22 | **Throttle only (submission hygiene), no protocol change.** `w1_family.slurm` and `w1_family_controls.slurm` now carry `--array=1-200%20` and `--array=1-40%20`, so at most 20 tasks start at once (the 200/40 task counts, configs, seeds and arms are unchanged). `bp_sched_tune.slurm` stays unthrottled at 2 tasks. Commit **`e21981df2f14ebcc8f2e1b1d3da4cbcbadb97f85`**; `python3 -m pytest tests/test_w1_launchers.py` = 5 passed. The `w1-launch` tag was re-pointed from `e12278d` to the metadata commit carrying this row **before any job was submitted**, so no submission exists under the earlier tag point. | agent |
