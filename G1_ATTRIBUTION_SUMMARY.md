# G1 attribution summary (one page)

**Question.** Is MF's accuracy/energy edge over BP a property of **local credit assignment
(detachment)** or of the **staged-greedy schedule**? G1 is the pre-registered separator
(`PREREG.md`): MF-Simul = MF-Joint's architecture and readout **without staging** (per-batch
detached inputs, one Adam, one global early stop), tuned under the same ladder-Optuna protocol
as every other arm.

**Design.** Four arms × {CIFAR-10, CIFAR-100} × 20 seeds (42–61), run contemporaneously on one
commit (`08118ca`; `src/` identical to HEAD), Protocol A, no checkpoint restore. 160 decisive
runs + 12 artifact repairs (job 3376703). Verdict was frozen before reduction; the artifact
audit (below) changed pass/energy *sources*, not endpoints or thresholds.

## Headline (accuracy, paired by seed)

| arm | C10 Δ vs BP (pp) | C100 Δ vs BP (pp) | verdict-relevant |
|---|---|---|---|
| MF-Joint | −0.19 [−0.44, +0.05], p=0.19 | **+1.86** [+1.66, +2.06], p=1.9e−6 | joint MF readout alone ≈ +2 pp on C100 |
| MF-Simul | **−0.86** [−1.16, −0.56], p<0.001 | **−0.27** [−0.50, −0.03], p=0.076 (Wilcoxon; paired-t CI [−0.525, −0.008], p=0.044) | detachment alone ≈ BP; no edge |
| MF-Seq | **+3.90** | **+5.04** [+4.82, +5.25] | staged schedule reproduces the paper edge |

Decision rule: C100 Δ(MF-Simul − BP) ≤ +0.5 pp with MF-Seq ≥ +2.0 pp ⇒ **Decision B (schedule
artifact)**. Detachment is not where the accuracy comes from; the staged schedule is.

Brackets are 95% percentile-bootstrap CIs for the mean (production reducer: 10,000 resamples,
seed 0); p-values are exact two-sided paired Wilcoxon tests. The C100 MF-Simul disagreement
between the interval and the rank test is documented in the test-provenance note below.

## What the difference costs (paired, same seeds; C100)

| arm | passes | energy (Wh) | wall (s) | peak alloc (MiB) |
|---|---|---|---|---|
| BP | 43.9 | 4.48 | 220 | 302 |
| MF-Joint | 50.2 (×1.14) | +0.62 [+0.47, +0.79] | +30 | +16 |
| MF-Simul | 35.7 (**×0.81**) | **−0.93** [−1.13, −0.71] | **−40** | +16 |
| MF-Seq | 191.3 (×4.36) | +12.18 [+11.69, +12.69] | +731 | **−74** |

- C10 is the cautionary dataset: MF-Simul there costs **+5.3 passes and +0.50 Wh (+9%)** for
  −0.86 pp. The energy saving of the unstaged MF is C100-specific, not a general property.
- MF-Seq's edge costs 3.7× energy and 4.4× time on C100; its allocator memory is *lower* than
  BP's (228 vs 302 MiB).

## Cost structure (W0b): the dead M0 stage

M0 contributes nothing to any prediction (phase 2 never reads it; evaluation uses a_L, M_L;
`tests/test_m0_dead_stage.py`) yet owns **22.9% (C10) / 22.8% (C100)** of MF-Seq's energy and
~23% of its passes (`scripts/jpdc/w0b_m0_energy.py`). A no-M0 arm is in the W1 ladder.

## Artifact integrity (why the numbers moved since the first reduction)

- **12/160 runs** declared Oct-17 NVML traces that never landed (HOME-quota window). The old
  reduction's glob fallback had silently borrowed September PPAM traces for them; energies for
  those (arm, seed) pairs are excluded and rebuilt by a fresh same-commit rerun.
- **Passes now come from each run's own JSON** (`epochs_completed`): the shared append-only log
  lost header/mid-block lines that day (MF-Seq C100 seed42: log says W1_M1=100, the task's own
  `.out` and JSON show 81; totals 204 vs 185). Corrections: BP C10 50→53, C100 40→43.9;
  MF-Seq C10 240→211, C100 213→191 (the latter matches the PPAM-era 191.4).
- **Drift audit (W0e):** G1 vs PPAM-era is machine-state level only — no co-tenancy on any GPU
  (max compute_processes = 1 in all 1,000+ traces); power/temp deltas are single-digit
  percent and arm-specific, and the energy drift decomposes into passes (BP only, +1 epoch),
  power (−5.5% to +3.8% log; negative only for BP C10) and seconds-per-pass (+0.5% to +3.7%
  log). BP C10 actually got cheaper (−2.2%); the C100 gaps are +8.3% (BP), +4.1% (MF-Joint)
  and +6.9% (MF-Seq). It is not a schedule change.
- **Cap anatomy (W0c):** per-task stdout reproduces the PPAM stage profile: MF-Seq's W1_M1
  early stop is cap-bound in **90% (C10) / 55% (C100)** of runs (PPAM 90%/50%), M0 owns
  23.4%/23.2% of passes, and per-stage sums equal the JSON `epochs_completed` in **40/40**
  runs. The decisive reruns are faithful re-executions, not a different schedule regime.
- **BP convergence audit (W0d):** the BP baseline is not cap-truncated. It stops on validation
  loss after a full 19-epoch patience tail (best epoch 33.0/23.9; last epoch 52.0/42.9), and
  validation loss rises in 40/40 runs. Its accuracy reproduces the PPAM value exactly
  (+0.00 pp on both datasets), so the baseline is a genuine early-stopping result.
- **Independent re-derivation (2026-10-21):** a separate reducer with no import of
  `analyze_g1.py` or `w0a_energy_memory.py` (bootstrap seed 20261021) reproduces every
  headline accuracy, pass and energy delta, including the decisive MF-Simul values
  (−0.86 pp C10, p=0.0003; −0.27 pp C100, p=0.0759). 91/96 cross-checks agree within
  tolerance; the five flags are ≤0.25 passes and come from the stored table's 1-decimal
  rounding, not a substantive mismatch (`jpdc_data/verification/`).
- **C100 test provenance:** the reported p=0.0759 is the pre-registered two-sided paired
  Wilcoxon signed-rank test (exact, n=20, W=57), not a bootstrap p-value. The 95% CI is a
  percentile bootstrap of the mean. A paired t-test gives −0.2665 pp [−0.5247, −0.0083],
  p=0.0437. The interval/test disagreement is an estimand difference (mean vs signed ranks),
  not an arithmetic error; the t result is reported as sensitivity only and does not replace
  the pre-registered verdict.
- G1 accuracy verdicts are unaffected by all of the above (accuracy never depended on logs).

## Implications

1. **Paper-facing correction:** the PPAM ablation's "detach step = largest single gain
   (+4.10/+3.18 pp)" conflates detachment with the staged schedule. G1: detachment alone ≈ BP
   (−0.86/−0.27 pp); the +3.18 pp reappears only when staging returns. Reads as a *sharpening*,
   not a refutation: the measured ladder numbers stand, the interpretation moves to "schedule".
2. **Honest arm for "MF as published":** MF-Simul reproduces the original paper's joint,
   all-layers-evolving schedule; on C100 it matches BP accuracy at ×0.81 passes and −0.93 Wh —
   a legitimate efficiency story without the accuracy gain.
3. **W1 ladder (pre-registered, `PREREG-phase2.md`):** Seq-Global (staged, trainable prefix,
   implemented + unit-tested) isolates schedule vs frozen prefix; MF-Seq-noM0, BP-fixed and cap-300
   controls bound the cost levers.

Sources: `PREREG.md` §10 (amendments), `g1_per_seed.csv`, `jpdc_data/g1/w0a/`,
`w0b_summary.md`, `w0c_summary.md`, `w0d_summary.md`, `w0e_per_seed.csv`,
`jpdc_data/verification/`; jobs 3372435/3373251/3373266/3376200/3376703.
