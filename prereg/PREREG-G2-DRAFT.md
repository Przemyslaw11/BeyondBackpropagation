# PREREG-G2 (DRAFT, NOT IN FORCE): per-pass cost under saturation, measurement hardening, initialisation control

Status: **FROZEN 2026-11-11 (tag `g2-freeze`); scope A approved by the author on 2026-11-10** (see section 7);
thresholds in section 4 are still proposals until the freeze. Originally written written 2026-11-10 on branch
`phase3-prereg-draft` (from `jpdc-extension` @ `f1896db`). Nothing here is frozen,
submitted or allowed on the Athena tree `~/BeyondBackpropagation` while any W1 task
can still start. No W1 outcome was read to write it. Every threshold below is a
**proposal**; the author fixes them, and the frozen version gets its own commit,
tag and §9-style amendments log before any confirmatory run.

## 0. What G2 does not repeat

G1 and W1 already answer the memo's (`research_direction.md`) schedule questions:
detachment vs staging (G1, verdict B), which part of the staged schedule carries the
edge (W1 core, Seq-Global, no-M0), and the stronger-BP question (W1 BP-bestval,
BP-sched, BP-fixed). G2 does not rerun them. It reuses their pass counts `N` and adds
what they cannot give: the per-pass energy `e` under conditions that do not leave
the GPU idle, and a cleaner energy measurement.

## 1. Question

The cost model (`manuscript/jpdc/theory_cost_model.tex`) writes end-to-end energy
as `E = N e`, with per-pass advantage `a = e_b / e_f` and pass multiplier
`m = N_f / N_b`. All existing `e` values come from runs in which the GPU is mostly
idle: utilisation 6.0% (MF) and 11.5% (BP) [PPAM Sec. 5.2, verified in the
camera-ready]. A reviewer can therefore say that `a` measures host-loader waiting,
not the arithmetic a forward-only method saves. G2 asks:

- **Q1.** How does `a` (MF-Simul vs BP, MF-Seq-nom0 vs BP) change as the GPU moves
  from host-bound to saturated (batch x width)?
- **Q2.** How much of the existing per-pass energy differences survive an
  energy-counter measurement, an idle-power correction and same-GPU pairing?
- **Q3.** Does the BP vs MF initialisation difference (Section 2.3) move BP's
  accuracy or pass count?

## 2. Findings of 2026-11-10 that shape the design (verified, no W1 outcome read)

1. **MF-Simul equals the published per-batch rule up to two per-layer constants.**
   With detached inputs, one Adam over `L_readout + λ Σ L_aux` gives exactly the
   same parameters as per-layer Adams on the unweighted losses (MF Algorithm 1,
   arXiv:2501.09238, Adam per its Sec. 4) whose hidden layers use weight decay
   `wd/λ` and Adam epsilon `ε/λ`. Checked numerically (max parameter difference
   1.1e-16 after 50 steps, float64). So under Adam, `λ` acts only as a layer-wise
   L2 and epsilon ratio: frozen values give hidden-layer L2 of 1.29e-4 vs 2.66e-6
   for the top layer (CIFAR-10, λ=0.0207) and 8.87e-5 vs 1.92e-5 (CIFAR-100,
   λ=0.217). MF-Simul may be called "the authors' per-batch MF update with a tuned
   layer-wise weight decay"; at λ=1 it is the update exactly.
2. **MF-Simul also trains an `M_0` head on the raw input.** Detached, it influences
   no other parameter; it costs compute only (3072x10 / 3072x100 per sample).
3. **Initialisation differs between BP and every MF arm.** BP (`engine.py`,
   `nn.Sequential`) keeps PyTorch's default `nn.Linear` init; `MF_MLP` uses
   `kaiming_uniform_(a=√5, nonlinearity="relu")`, where PyTorch ignores `a` for
   ReLU, i.e. He-uniform, plus zero biases. Hidden-weight std on a 3072→2000
   layer: BP 0.0104, MF 0.0255 (**2.45x**); BP biases U(±0.018), MF 0. This
   confounds every BP-vs-MF contrast in G1 and W1; within-MF contrasts are clean.
   It cannot be changed in W1 (frozen) and must be disclosed in the paper.

## 3. Elements, ordered by value per GPU-hour

| ID | Element | What it adds | GPU-h (estimate) |
|---|---|---|---|
| E-I | Instrumentation: `nvmlDeviceGetTotalEnergyConsumption` at epoch / window boundaries as primary energy (5 Hz trace kept as cross-check); 5-min idle baseline per job, gross and net reported; GPU UUID, host, driver, job ID in every record | Q2 | ~0 (code only) |
| E-P | All arms of one repeat in one job on one GPU, randomised order | Q2: cancels GPU unit offset | ~0 (scheduling) |
| E-D | Input data resident on the GPU, GPU-side crop/flip; bridge cell on the host loader | Removes the idle confound for Q1 | ~1 (bridge) |
| E-S | Saturation sweep: batch {128, 512, 2048, 8192} x width {2000, 4000, 8000}, depth 3; arms BP, MF-Simul, MF-Seq-nom0 (one stage at a time); fixed number of optimiser steps after warm-up, not to convergence | Q1: `e` per pass per cell; combined with W1 `N` via `E = N e` | ~6 (12 cells x 3 arms x 5 repeats x ~2 min) |
| E-B | Initialisation control: BP with `MF_MLP` init (He-uniform, zero bias), W1 core-BP hyperparameters unchanged, CIFAR-10/100, seeds 42-61 | Q3 | ~3 (40 x ~4 min) |
| E-M | Memory breakdown (allocator snapshot per tensor class) at the E-S cells | Peak-memory claim | ~1 |
| E-X | Measured layers-on-separate-GPUs runs (1, 2, 4 GPUs) | Tests the W2/W3 link model | ~40; optional per editor's scope answer |

GPU-h estimates are arithmetic from G1 wall times (BP CIFAR-100 220 s per run,
`G1_ATTRIBUTION_SUMMARY.md`), not measured on the new pipeline. A pilot of E-D/E-S
(exploratory, ~1 GPU-h) is needed before the step count and repeat count are fixed.

## 4. Hypotheses (thresholds approved by the author 2026-11-10; repeat count open until the pilot)

| ID | Statement | Metric | Proposed test |
|---|---|---|---|
| H-G2a | `a` (MF-Simul vs BP) increases with batch x width | net counter energy per optimiser step, per cell | one-sided test on the slope of `log a` against `log(batch x width)`, paired by repeat within a job |
| H-G2b | In the saturated corner (batch 8192, width 8000) MF-Simul's net energy per pass is below BP's | same | one-sided, paired, `log a > 0` |
| H-G2c | The per-pass energy ordering of the host-bound bridge cell agrees in sign with the PPAM measurement | same, host loader, batch 128, width 2000 | sign agreement, descriptive |
| H-G2d | BP accuracy with MF init is equivalent to BP with default init | test accuracy, paired by seed | TOST, ±0.5 pp (to align with H-W1 thresholds; author to confirm) |

Multiplicity: Holm over {H-G2a, H-G2b, H-G2d}; H-G2c descriptive. Bootstrap
10,000 resamples, as in W1. End-to-end projections `N e` that combine W1 pass counts
with E-S per-pass energies are labelled **derived**, never measured.

## 5. Scope options for the author

| Option | Contents | GPU-h | Needs | Feasible on `plgoncotherapy2` (ends 2026-11-23)? |
|---|---|---|---|---|
| **A (recommended)** | E-I, E-P, E-D, E-S, E-B | ~11 + pilot | scope fixed by ~2026-11-13; code + tests + freeze by ~2026-11-18 | only if W1 finishes first and nodes return; tight |
| B | A + E-M + depth sweep (3, 6, 12) | ~40 | as A | very tight |
| C | B + E-X (measured multi-GPU) | ~80 | multi-GPU nodes | no; needs a successor allocation |

Without a successor allocation, anything not finished by 2026-11-23 is not measured.
The class grant `plglscclass26` is not an option until its research use is resolved
(`CLASS_JOB_FACT_SHEET.md`).

## 5a. Implementation (scope A, on this branch; not on Athena)

| Element | Where | Status |
|---|---|---|
| E-I counter | `src/utils/monitoring.py::get_gpu_energy_counter_j`; `engine.py` writes `counter_training_energy_j` over the `training_duration_sec` window | done, tested on CPU (NaN without NVIDIA) |
| E-I idle, identity | `scripts/jpdc/g2_sweep.py` (`idle_baseline`, `gpu_identity`: UUID, name, driver, host, job ID) | done |
| E-P pairing | `g2_sweep.py`: one task = one repeat = all units on one GPU, order shuffled by repeat seed | done |
| E-D GPU-resident data | `g2_sweep.py::gpu_augment` (reflect-pad 4, random crop, flip; tested against torchvision semantics); `--pipeline host` bridge uses the production transforms | done |
| E-S sweep | `g2_sweep.py`; arms `bp`, `mf_simul` (production forward/loss from `bp_ds.py`), `mf_stage1..3` (MF-Seq-nom0 stage step mirroring `mf.py`); optimiser type/lr/wd from each arm's W1 config | done; launcher `g2_sweep.slurm` |
| E-B init control | `bp_init: mf` in `engine.py` (`_apply_mf_mlp_init`); configs `configs/g2_bp_mfinit/*` = core BP configs + flag; launcher `g2_bp_mfinit.slurm` (40 tasks) | done; MNIST smoke (3x64 widths, local, not a result) through `run_experiment.py` OK |
| Submission | `g2_submit.sh {pilot,sweep,bridge,init}`; `ACCOUNT` has no default; the class grant is refused | done |
| Analysis script + synthetic fixture test | `scripts/jpdc/analyze_g2.py`, `tests/test_analyze_g2.py` (planted effect, null, TOST, incomplete, duplicate/pipeline guards) | written; draft hash below; the **freeze hash** is recorded after the author fixes the section-4 constants |
| Pilot | `g2_submit.sh pilot` (1 task, short windows; exploratory) | not run; needs the new allocation and W1 finished |

Measurement unit: per (cell, arm), the driver counter difference over a step count
fixed after a 10 s warm-up (target 30 s window), divided by samples and scaled to one
45,000-sample pass. MF-Seq per-pass energy is the stage energies weighted by W1
stage epochs (derived). Updated cost: sweep 10 tasks x ~50 min = ~8.5 GPU-h; bridge
~1.6; E-B ~3; pilot ~0.5: **~14 GPU-h**.

## 6. Rules carried over from W1

- Pilots and smoke runs are exploratory and never enter a decisive table.
- The analysis script is written, tested on a synthetic fixture and hashed before any
  confirmatory run.
- G2 code lives on this branch. It reaches Athena only in a **separate checkout**,
  and only after every W1 task has finished (W1 imports `src/` from
  `~/BeyondBackpropagation` at run time).
- Every protocol step gets a timestamped row in the amendments log below.

## 7. Amendments log

| date | change | by |
|---|---|---|
| 2026-11-10 | Draft written for the author's scope decision; not in force | agent |
| 2026-11-10T09:59:04+01:00 | **Author decisions (in session).** (1) Scope **A** approved (E-I, E-P, E-D, E-S, E-B). (2) The author states a new GPU allocation will be available by 2026-11-23; its account name is not yet known, so no G2 job can be submitted and the W1 §9 allocation-expiry row stays a draft. (3) The "15 Nov DSS" item in `ERRATA_PPAM.md` is not relevant. Thresholds in section 4 remain proposals until the freeze. | author-confirmed in session |
| 2026-11-10T09:59:04+01:00 | Scope-A code, configs, launchers and tests added on this branch (section 5a); suite 122 passed, 13 skipped. Nothing on Athena. No W1 outcome read. | agent |
| 2026-11-10T10:20:28+01:00 | **Pilot approved by the author to run before W1 finishes.** Athena checkout `$SCRATCH/bbp_g2` @ `e4a963f` (separate from the W1 tree); account `plgoncotherapy2-gpu-a100` (the author states it will be prolonged). Exploratory: fixes the measurement window and repeat count only. | author-confirmed in session |
| 2026-11-10T12:07:28+01:00 | **Analysis script drafted (no G2 or W1 data read).** `scripts/jpdc/analyze_g2.py` draft SHA-256 `1127d5887393e365258566fd44d316e9a25314d865373ee4d33a21b443a42ff7`; reuses `analyze_w1.py` helpers by import (unmodified). Proposed constants as in section 4: primary CIFAR-100 with Holm over {H-G2a, H-G2b, H-G2d}, CIFAR-10 raw consistency, TOST ±0.5 pp, corner (8192, 8000), 5 repeats, net energy = counter − idle W × window. Suite 127 passed, 13 skipped. Not frozen. | agent |
| 2026-11-10T12:10:26+01:00 | **Author approved the section-4 values as they stand (in session).** H-G2a one-sided t on per-repeat slopes, slope > 0; H-G2b one-sided t at (8192, 8000); H-G2c descriptive at (128, 2000); H-G2d paired TOST ±0.5 pp over seeds 42-61; α = 0.05; CIFAR-100 confirmatory with Holm over {H-G2a, H-G2b, H-G2d}, CIFAR-10 raw-p consistency; net energy = counter − idle W × window. **Open:** the repeat count (`N_REPEATS`, draft 5), to be fixed after the pilot; the freeze hash of `analyze_g2.py` is recorded then. | author-confirmed in session |
| 2026-11-11T19:10+01:00 | **Pilot 3534024 done (exploratory).** CIFAR-10 repeat 0, 60 units, 22 min, exit 0, t0017 (A100-SXM4-40GB). Counter/trace 0.990-1.031; idle 56.1 W; net/gross >= 0.65 (no non-positive net); corner peak 4.4 GiB, >= 26 steps per 10 s. One repeat cannot estimate between-repeat variance; the repeat count is set on cost and power grounds (author decision pending). | agent |
| 2026-11-11T19:10:29+01:00 | **Frozen and submitted (author: "do the recommended approach", 2026-11-11).** Repeats: **10 per dataset** (sweep and bridge 20 tasks each). Freeze commit `cf79f046417235657a2d8a65ee9351e3fd2cd5fb`, tag `g2-freeze` (local + Athena). SHA-256: `analyze_g2.py` `486fed67c1b6ab68d15dc4ef4e610b72089911fd74f46325db3bd0c978d772c3`; `g2_sweep.py` `c8b49b3d9ef71280c62091032499827b23aa429c45c4c8a8cdb368da51664c26`; `configs/g2_bp_mfinit/cifar10_*` `d67a379d…2c947`, `cifar100_*` `befdc6dc…5351f`; `engine.py` `af443635…a4a2`; `monitoring.py` `330298fa…6364`. Athena checkout `$SCRATCH/bbp_g2` fast-forwarded to the tag, hashes equal. Account `plgoncotherapy2-gpu-a100`. Jobs: sweep **3545830** `[1-20]`, bridge **3545831** `[1-20]`, E-B **3545832** `[1-40%20]`; `sbatch --test-only` start estimate 2026-12-21 (23 failed + 13 reserved nodes; fairshare factor 5e-6), past the 2026-11-23 allocation end unless prolonged. Pilot data not used in any decisive table. | author-confirmed in session |
| 2026-11-11T19:36:34+01:00 | Time limits shortened on the pending arrays (author-approved; scheduling only): E-B and bridge 0:30:00, sweep 1:15:00. Launcher files unchanged. | author-confirmed in session |
| 2026-11-12T23:49:00+01:00 | **G2 complete; copied out (no analysis run, no outcome read).** 80/80 tasks COMPLETED, exit 0:0, no requeues: sweep 3545830 20/20 (2743-2766 s), bridge 3545831 20/20 (498-504 s), E-B 3545832 40/40 (214-357 s). Integrity only: 20 sweep files x 60 units, 20 bridge files x 5 units, one per (dataset, repeat); counter energy present in every unit; counter/trace 0.984-1.022 (sweep), 0.930-1.053 (bridge); 40 E-B result JSONs with accuracy and counter energy (the only `nan` in logs is the FLOP-estimate field). Copy-out `$SCRATCH/g2_copyout_20261112.sh` to `plggfastsim/plgspyra_bbp_archive/g2_copyout/2026-11-12/`: 322 files (results minus pilot, plus slurm logs), 322/322 OK at destination and locally (`athena_g2/copyout_2026-11-12/`); manifest `619aacb781209b56…`, committed as `manuscript/jpdc/copyout_manifests/2026-11-12/g2_SHA256SUMS`. | agent |
| 2026-11-13T00:21:20+01:00 | **Frozen G2 analysis: run authorised and about to execute once (author: "Please run it, and push branch", 2026-11-13, after being told G2 is complete).** Code: `scripts/jpdc/analyze_g2.py` SHA-256 `486fed67c1b6ab68d15dc4ef4e610b72089911fd74f46325db3bd0c978d772c3`; `git diff --stat g2-freeze -- src/ scripts/ configs/` empty; `tests/test_analyze_g2.py` 5 passed (local Python 3.13.7, numpy 2.3.5, pandas 3.0.5, scipy 1.17.0). Input: symlink tree `jpdc_data/g2/analysis_input_20261113/` → `sweep` = copy-out `g2_sweep` (20 files, jobs 3545830/3545831 only, no pilot file), `bridge` = copy-out `g2_sweep_bridge` (20), `results/runs/g2_bp_mfinit_{cifar10,cifar100}_mlp_3x2000` (20 + 20) from the 2026-11-12 copy-out (322/322 match manifest `619aacb7…`), and `results/runs/bp_{cifar10,cifar100}_mlp_3x2000` = the W1 core BP runs (40/40 match `copyout_manifests/2026-10-25/core_tasks_1-200.SHA256SUMS` on the `jpdc-extension` branch). Pilot 3534024 is not in the input. Output: `jpdc_data/g2/analysis_20261113/`. No G2 outcome read before this row. | author-confirmed in session |
| 2026-11-13T00:22:01+01:00 | **Frozen G2 analysis run once.** `PYTHONPATH=. python3 scripts/jpdc/analyze_g2.py --sweep-dir ../jpdc_data/g2/analysis_input_20261113/sweep --bridge-dir …/bridge --results-dir …/results --out-dir ../jpdc_data/g2/analysis_20261113` (local, commit `6564e88`, script `486fed67…`), started 2026-11-13T00:21:24+01:00, exit 0. 1,200 sweep units, 100 bridge units, 0 non-positive or missing net energy; 18 distinct GPUs; idle 54.9-63.1 W; 20/20 seed pairs per dataset. 7 output files copied unchanged to `analysis/g2_frozen_20261113/`; manifest `analysis/g2_frozen_20261113.SHA256SUMS` (`07aa6749…`). Reducer labels (quoted), CIFAR-100 confirmatory, Holm over {a, b, d}: **H-G2a supported** (slope of log a on log(batch x width) +0.0539 [+0.0524, +0.0551], Holm p 9.4e-14); **H-G2b supported** (log a at (8192, 8000) +0.368 [+0.359, +0.378], Holm p 9.4e-14; a = 1.44, derived); **H-G2d not supported** (BP-MFinit − BP −0.81 pp [−0.96, −0.64], TOST ±0.5 pp p 0.999). CIFAR-10 raw-p consistency: H-G2a and H-G2b consistent (+0.0522; +0.374), H-G2d not consistent (−1.08 pp [−1.30, −0.85]). H-G2c descriptive at (128, 2000) on the host pipeline, MF-Simul log a +0.029 (C100) / +0.079 (C10), all four MF arms "MF cheaper". The frozen script is not modified. | agent |
