# PREREG.md — JPDC extension, G1: detachment vs schedule

Freeze rule: this file is committed **before** any decisive G1 result is inspected. Changes
after that point are date-stamped additions to §10 only. Pilots (n=5) are exploratory and
never enter decisive tables. Source facts are [Verified] in `AUDIT.md` unless marked otherwise.

## 1. Question

Is the MF family's efficiency/accuracy edge over backpropagation a property of **local
credit assignment** (detachment), or of the **staged-greedy schedule**?

- `H_G1a` (detachment survives): MF-Simul keeps an accuracy edge over BP at joint-schedule cost.
- `H_G1b` (schedule artifact): the edge disappears without staging; MF-Seq's advantage is a
  stopping/schedule artifact, and the honest "published-schedule" arm is MF-Simul.

Context [Verified]: BP→MF-Joint→MF changes five things at once; the 3.7–4.5× unit-epoch
multiplier is produced by staging, not by locality (AUDIT §2–3). The original MF paper is a
per-batch in-order schedule with all layers evolving (AUDIT §4), which MF-Simul reproduces
more faithfully than staged MF-Seq.

## 2. Arms and operational definitions

| arm | schedule | gradient path | readout | optimizer | ES scope |
|---|---|---|---|---|---|
| BP | 1 joint unit | full backprop | output layer | AdamW | global, Protocol A |
| MF-Joint | 1 joint unit | full backprop through aux heads | M_L projection | Adam | global, Protocol A |
| MF-Simul | 1 joint unit | detached inputs (below) | M_L projection | Adam | global, Protocol A |
| MF-Seq | M0 + 3 sequential stages, frozen prefix, fresh optimizer + ES per stage | detached inputs per stage | per-stage M_i, final M_L | Adam | per stage, Protocol A |

MF-Simul, exact (implemented at `src/baselines/bp_ds.py:69,482`):

```
a_0 = x
a_i = ReLU( W_i · a_{i-1}.detach() )            i = 1..L
loss = CE(M_L a_L, y) + λ · Σ_{i<L} CE(M_i a_i, y),   λ = aux_weight = 1.0
```

One backward pass per batch sums all local losses; every local loss reaches only its own
`W_i` and `M_i`. Single Adam, single global early stopping on the validation CE of M_L.
`λ = 1.0` keeps the summed objective identical to MF-Seq's per-unit local losses, so the
only difference from the staged algorithm is the schedule, not the objective weights.
Equivalence anchor: one MF-Simul step = one step of each MF-Seq unit's objective
(`tests/test_mf_simul.py`, seeded unit tests).

## 3. Endpoints

- **Primary:** test accuracy of the arm's final readout (BP: output layer; MF family: M_L),
  final weights, single test evaluation, no checkpoint restore (`BBP_DISABLE_CHECKPOINTS`).
- **Co-primary (cost):** training-unit epochs ("passes", Protocol A counting); measured
  run energy (Wh) and wall-clock from NVML/monitoring; report all three per arm.
- **ES quantity** [frozen]: validation CE of the final readout (BP: output layer; MF family:
  M_L), Protocol A values: `metric val_loss, min_delta 0, patience 20, cap 100` per unit.

## 4. Materials

- Datasets: CIFAR-10, CIFAR-100; 3×2000 ReLU MLP; val_split 0.1; existing pipeline unchanged.
- Seeds: **42–61, n=20, paired across arms**, per dataset; reused from the camera-ready runs.
- Hyperparameters: BP, MF-Joint, MF-Seq at camera-ready tuned values (commit `181ecc4`).
  MF-Simul is tuned per dataset with the **same ladder Optuna procedure** as MF-Joint:
  50 TPE trials, maximize `val_accuracy`, no pruner (study `ladder_mfsimul_<dataset>_*`,
  driver ported from `main:scripts/run_optuna_search.py`); best-params YAML frozen under
  `configs/mf_simul/` before decisive runs. Until then, `configs/mf_simul/*.yaml` carry
  provisional MF-Joint values (clearly commented).
- Decisive runs: one commit for all four arms; config diffs between arms limited to
  algorithm name and hyperparameter values.

## 5. Decision rule (frozen thresholds)

Let Δ100 = MF-Simul − BP test accuracy on CIFAR-100, R = passes(MF-Simul)/passes(BP).

- **A (detachment survives):** Δ100 ≥ +2.0 pp **and** R ≤ 1.5.
- **B (schedule artifact):** Δ100 ≤ +0.5 pp **and** MF-Seq retains ≥ +2.0 pp over BP on
  CIFAR-100 in the contemporaneous runs.
- **C:** anything else → report as intermediate; interpretation chosen only after seeing C
  (allowed by this PREREG), with A/B labels applied to CIFAR-10 as the consistency check.

Statistics: paired-by-seed comparisons (Wilcoxon signed-rank, two-sided, α = 0.05, Holm
across the two datasets for the accuracy comparison); effect size = mean paired difference
with 95% bootstrap CI (10k resamples, fixed bootstrap seed 0). Full per-seed table reported.

## 6. Required evidence before launch

1. `python3 -m unittest tests.test_mf_simul tests.test_deep_supervision_gradients tests.test_fairness_invariants`
   passes at the decisive commit (gradient locality, trajectory equivalence, config fairness).
2. Tiny CPU smoke test of MF-Simul end-to-end via `scripts/run_experiment.py` (done
   2026-10-17; rerun if the training code changes).
3. Analysis script `scripts/jpdc/analyze_g1.py` committed; reduces raw results to every
   number used in §5; its hash recorded in §10.

## 7. Drift control

All four arms are **rerun contemporaneously at n=20 on the decisive commit** (budget is not
a constraint, author 2026-10-16). Stored camera-ready results are used only as a sanity
reference, never as the comparison baseline. If MF-Seq cannot be rerun, fall back to n=5
calibration (seeds 42–46) against stored per-seed results: |Δmean acc| ≤ 0.3 pp and
|Δpasses| ≤ 5% required, else full n=20 rerun before any decisive claim.

## 8. Budget and energy accounting

Per session, report: SLURM job ids, GPU-hours from `sacct`, NVML Wh per run, wall-clock.
Budget is not a planning constraint [author, 2026-10-16]; measured usage still reported.
Power model context only: `P = 51.2 + 1.14·u` pooled (AUDIT §6).

## 9. Pilots

Pilot = n=5, seeds 42–46, CIFAR-10, all four arms: sanity of the MF-Simul code path, pass
and wall-clock estimates, variance estimate for sizing. Pilot results are exploratory
(summary for the ~29 Oct author meeting) and never pooled with decisive results.

## 10. Amendments log

| date | change | by |
|---|---|---|
| 2026-10-17 | PREREG created; MF-Simul implemented + tested (same commit) | agent |
| 2026-10-17 | analysis script added: `scripts/jpdc/analyze_g1.py` @ `8a4ce41` (dry-run on synthetic fixture passed); tuning stack ported (`90e7f95`), MF-Simul studies `ladder_mfsimul_*` launched (SLURM 3371585) | agent |
