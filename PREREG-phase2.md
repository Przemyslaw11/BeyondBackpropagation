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
- **H-W1d (strong-BP control):** BP with `max_epochs: 300` (same protocol otherwise). If
  BP-long − MF-Seq ≥ −1.0 pp on C100 at ≤2× the passes of MF-Seq, the "compute-advantage"
  framing of MF is retired. Expected defensible outcome [prior]: BP-long does not close the
  gap; BP never hits the 100-epoch cap in any existing run [Verified, W0d], so this control
  is about time-to-accuracy, not truncation.

Thresholds for A/B-style readings are the ones above; intermediate outcomes are reported as
intermediate with both readings and no third relabeling of G1.

## 4. Endpoints and statistics

- **Primary:** test accuracy of the final readout, final weights, single evaluation, no
  checkpoint restore (`BBP_DISABLE_CHECKPOINTS=1`).
- **Co-primary (cost):** passes (JSON `epochs_completed`), NVML energy (Wh, strict
  `monitoring_csv_path`), wall-clock. Per-arm means with 95% CI; paired-by-seed differences
  with Wilcoxon + 10k bootstrap, Holm within the family {H-W1a, H-W1b, H-W1c}; H-W1cap and
  H-W1d are reported with raw CIs (controls).
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
- Runs: 6 main arms × 2 datasets × 20 seeds = 240; plus cap-300 MF-Seq and BP-long
  conditional arm families per §3. One commit for the whole family; verify `src/` identical
  across arms.
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

Commit this file before scheduling any W1 confirmatory job. Record the commit hash and the
`analyze_w1.py` hash in §9 at launch. Until §5's HPO decision is recorded, W1 jobs may only
be pilots.

## 9. Amendments log

| date | change | by |
|---|---|---|
| 2026-10-17 | PREREG-phase2 drafted (W1 attribution ladder: Seq-Global implemented + unit-tested; mf_nom0 configs added; CPU smoke of Seq-Global OK). No W1 run scheduled yet | agent |
| 2026-10-21 | §5 HPO decision recorded: **option (b)**, parity-maximizing. Seq-Global gets its own 50-trial ladder-Optuna study per dataset before any confirmatory run (`configs/tuning/mfglobal_*`, studies `ladder_mfglobal_{cifar10,cifar100}_mlp_3x2000_tune`); search space is the MF-family breadth lr/wd (no auxiliary weight exists — each stage has its own local loss). Best params frozen into `configs/mf_global/*`; the tuning commit hash and best-params YAMLs are recorded here when the studies finish. | agent |
| 2026-10-21 | **W1 config manifest confirmed (no protocol change).** The 14 configs are the six main arms × 2 datasets — BP, MF-Joint, MF-Simul, MF-Seq, Seq-Global, MF-Seq-nom0 — plus the H-W1d BP-long control × 2 datasets: 7 arms × 2 datasets × 20 seeds = **280 tasks** in `scripts/slurm_scripts/w1_family.slurm`. BP-long is the seventh arm/control, not one of the six main arms; `mf_nom0` is the "MF-Seq without stage 0" arm. Cap-300 remains a conditional two-step arm and is **not** in the 280-task array. | agent |
| 2026-10-21 | **W2/W3 prereg addendum drafted** as a separate file, `PREREG-W2-W3.md` (status: draft, not frozen; no confirmatory W2/W3 run scheduled). This keeps the W1 freeze untouched, as §7 requires separate addenda for W2–W6. The G1 W3 byte accounting in `jpdc_data/g1/w3/` is labelled an exploratory engineering pilot, not confirmatory evidence. | agent |
