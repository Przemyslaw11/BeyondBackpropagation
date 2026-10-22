# PREREG-W2-W3.md — JPDC extension, communication and byte accounting

**Status: draft, revised 2026-10-22. Not frozen. No confirmatory W2/W3 run has
been scheduled.** This is a separate addendum to `PREREG-phase2.md`, as required by
its §7 ("W2–W6 have their own prereg addenda before their confirmatory runs").
It does not alter W1's hypotheses, arms, thresholds, or freeze.

Shared discipline: paired seeds 42–61, contemporaneous arms on one commit,
Holm within a declared metric family, means with 95% CI and medians reported,
per-run JSON + own NVML trace as the source of truth, never the shared
append-only log. Pilots are exploratory and never pooled with decisive tables.
Amendments to this addendum go in its §5 log, dated, before the affected run.

---

## W2 — communication accounting

### W2.1 Question

Does the credit-assignment structure that produces MF-Seq's accuracy edge also
change the **cut-boundary communication** bill enough to alter the time or energy
ranking at a given link rate? W1 varies locality on one GPU; W2 asks whether
locality is a communication advantage, an accuracy advantage, or both.

This is not a scaling study. It is a link-rate accounting model over the measured
W1/G1 runs, plus one short NCCL latency diagnostic. The old four-rank training
experiment is superseded (§5, 2026-10-22).

### W2.2 Design (revised 2026-10-22; old design superseded)

The old design — four ranks, 160 layer-parallel training runs, measured NCCL
counters — is **superseded** and retained only in §5 as a planning record. It
was too expensive for the queue and answered a scale question the paper does not
need. W2 is now a **link-rate accounting model plus one short latency
diagnostic**:

**(a) Link-rate model (zero GPU; primary W2 evidence).** For BP, MF-Simul,
MF-Seq and Seq-Global, use the measured G1/W1 per-seed artifacts:

- measured total passes from each run JSON `epochs_completed`;
- measured per-step compute time = `training_duration_sec / (passes × 352)`;
- measured GPU energy from each run's own NVML trace;
- cut-byte bill from the W3 cut-boundary counter (useful cuts; the dead M0
  forward is excluded);
- link rates **100 Mbit/s, 1 Gbit/s, 10 Gbit/s**.

The model serializes compute and communication:

```
T_to_final_accuracy = T_compute + B_cut / R_link
E_to_final_accuracy = E_GPU + P_idle × (B_cut / R_link) + E_NIC
```

where `B_cut` is cut bytes to the run's stopping point and `P_idle` is the W4
measured intercept (51.97 W). `E_NIC` is reported at 1 pJ/bit with a 0 W port
default and a sensitivity for a nonzero port power; at these byte counts the
pJ/bit term is negligible while the idle-GPU term is not. The target is each
arm's **own measured final test accuracy**, because G1 did not log a per-epoch
test trajectory; this is a time/energy-to-final-accuracy frontier, not a
common-threshold crossing. Seed uncertainty is propagated by computing every
quantity per seed and reporting paired means, medians and 95% CIs.

Seq-Global has not run in G1. Until W1 produces it, the model emits an explicit
**planning proxy**: MF-Seq passes and accuracy, BP per-pass compute/energy, and
the analytic Seq-Global cut bill (W1 0, W2 0.72, W3 1.44 GB/pass, reflecting
forward plus backward across the trainable prefix). Proxy rows are marked in
the output and are never presented as measurements.

**(b) NCCL point-to-point microbenchmark (short diagnostic).** One 2-rank job
on two A100s exchanges a 1 MB float32 tensor in a ping-pong loop (20 warm-up,
200 timed iterations). It reports one-way latency and achieved bandwidth. This
is the latency floor for the 10 Gbit/s end of the model, not a confirmatory W2
run and not a substitute for a distributed training experiment.

**(c) Optional n=3 layer-parallel demo.** If a reviewer or author wants a
concrete pipeline sanity check, run a **three-rank, n=3 seed** demo of the
existing `MF_MLP` with forward activations and backward gradients across cuts.
It is descriptive only, is not pooled with W1, and cannot support a scaling or
accuracy claim. It is not scheduled.

### W2.3 Hypotheses and frozen decision rules

CIFAR-100 is primary; CIFAR-10 is the consistency check. All quantities are
per-seed model outputs from (a), paired by seed. No new 4-GPU training runs are
required for W2.

- **H-W2a (slow-link crossover, time):** at 100 Mbit/s, MF-Seq's total time to
  its own final accuracy is lower than BP's (mean ratio <1.0, 95% CI upper bound
  <1.0); at 10 Gbit/s, the ordering reverses (mean ratio >1.0, 95% CI lower
  bound >1.0). Reading: the cut-byte saving is large enough to matter only when
  the link is slow; at modern interconnect speeds MF-Seq's extra passes dominate.
- **H-W2b (slow-link crossover, energy):** the same reversal holds for total
  energy (compute plus idle-GPU communication energy), with the same CI
  directions. Reading: if true, the energy ranking is link-dependent and the
  paper must state the link assumption; if false, MF-Seq's energy cost is robust
  to the link model.
- **H-W2c (latency floor, diagnostic):** the measured 1 MB one-way NCCL latency
  is ≤100 µs and ≤10% of the 1 MB transfer time at 10 Gbit/s (838.9 µs). This
  is a diagnostic for whether the 10 Gbit/s model is bandwidth- or
  latency-bound; it is not a confirmatory hypothesis and gets no p-value.

Holm is applied within the time family {H-W2a at 100 Mbit/s, H-W2a at 10
Gbit/s} and within the energy family {H-W2b at 100 Mbit/s, H-W2b at 10 Gbit/s};
the two families are reported separately. No threshold is moved after seeing the
model output.

### W2.4 Endpoints and statistics

- **Primary:** total time and total energy to each arm's own final accuracy at
  100 Mbit/s and 10 Gbit/s, expressed as a paired ratio to BP; mean, median,
  95% bootstrap CI over seeds 42–61.
- **Secondary:** cut bytes, compute seconds, communication seconds, the
  communication-time share, and the communication-energy share at 1 Gbit/s;
  descriptive only.
- **Diagnostic:** the 1 MB NCCL one-way latency/bandwidth and its comparison
  with the 10 Gbit/s transfer time.
- **Exclusions:** no optimizer-state communication (ZeRO/FSDP), no multi-node
  scaling, no CNN, no overlap model. Compute and communication are serialized;
  any overlap would only reduce the communication penalty, so the model is
  conservative for slow links.
- **Source of truth:** per-run JSON for passes and `training_duration_sec`;
  each run's own NVML trace for energy; W3 cut accounting for bytes; no shared
  log.

### W2.5 Materials and run plan

- CPU model: `scripts/jpdc/w2_link_rate_model.py`; outputs in
  `jpdc_data/g1/w2_link/` (`w2_link_per_seed.csv`, `w2_link_summary.csv`,
  `w2_link_summary.json`, `w2_link_summary.md`). This is the primary W2
  deliverable and needs no GPU.
- NCCL diagnostic: `scripts/jpdc/nccl_p2p_benchmark.py` and
  `scripts/slurm_scripts/w2_nccl_p2p.slurm`; one short job, prepared but not
  submitted.
- Optional demo: design only; not implemented and not in the W1/W2 budget.
- No 4-GPU training budget. The old 160-run plan is retired; W2 cannot consume
  W1 core GPU hours.

### W2.6 What W2 will not claim

- No measured multi-GPU training result, no scaling efficiency, no interconnect
  comparison, no FSDP/ZeRO.
- No claim that communication causes the MF-Seq accuracy edge; W2 models cost,
  W1 measures attribution.
- No claim that the Seq-Global proxy is a measurement.
- No CNN or transformer results; those are W5.

---

## W3 — cut-boundary communication

### W3.1 Question

How many bytes cross a layer cut per data pass, and to stopping, at the
canonical 3×2000 CIFAR MLP? This is **cut communication**, not total memory
traffic. The old "69 GB/pass" table counted parameter re-reads and every layer
activation read/write; it is retained only as a relabelled legacy table
(`jpdc_data/g1/w3/w3_summary.md`, `w3_byte_accounting.py`) and must never be
cited as communication.

W3 is an accounting/engineering workstream, not a new accuracy experiment. Its
confirmatory data are the W1 runs; the G1 cut accounting already produced
(`jpdc_data/g1/w3_cut/`, 2026-10-21) is an **exploratory pilot** and is not used
for confirmatory p-values.

### W3.2 Estimand and accounting rules

- **Unit:** one cut tensor = `45,000 samples × 2,000 × 4 B = 0.360 GB` per
  epoch per direction per cut (decimal GB; 0.335 GiB). A forward activation and
  the corresponding backward gradient are separate directions.
- **BP:** each cut k=1..L carries a_k forward **and** dL/da_k backward:
  2×0.36 = 0.72 GB/pass per cut, 2.16 GB/pass total.
- **MF-Simul:** each cut carries a forward activation only; local heads never
  backprop across a cut: 0.36 GB/pass per cut, 1.08 total.
- **MF-Seq:** stage W_k recomputes the frozen prefix a_1..a_{k-1}; those prefix
  cuts cross into the stage. Current-stage a_k and its local head stay local and
  are not cut traffic. Stage W1 = 0, W2 = 0.36, W3 = 0.72 GB/pass. Cut 3 is
  therefore zero in useful MF-Seq.
- **M0:** production runs a full no-grad forward through W1..W3 before reading
  a_0. That is 3×0.36 = 1.08 GB/pass of pure implementation waste. Report it
  separately; include it only in the "as implemented" total.
- **Per pass** = one epoch over 45,000 training samples at batch 128 = 352
  batches. **Bytes-to-stopping** = per-pass bytes × measured passes: JSON
  `epochs_completed` for BP/MF-Simul. For W1 MF-Seq/Seq-Global, per-stage epochs
  come from the run JSON `stage_epochs` added by the 2026-10-22 instrumentation;
  G1's frozen artifacts fall back to per-stage stdout (a missing W1_M1
  early-stopping line means the 100-epoch cap; the stage sum is asserted against
  JSON). Never infer a pass count from the shared log.
- **Cached prefix:** canonical config is `activation_cache: recompute`; CIFAR
  random crop/flip makes cached frozen activations stale. The counterfactual is
  one fill per stage (0.36, 0.72 GB for W2/W3) plus one cached a_{k-1} read per
  epoch (0.36 GB); no backward crosses a cut. Report separately, never as the
  primary number.
- **Instrumentation:** forward hooks on ReLU outputs a_k; full backward hooks on
  the same tensors for BP. One batch of 128 is scaled to 45,000 samples and
  checked against the closed form `N_train × hidden_dim_k × 4`; mismatch >1e-9
  GB fails. Optimizer/validation traffic is excluded by design and stated as a
  bound.

### W3.3 Hypotheses and frozen decision rules

Confirmatory on the W1 family (six arms × two datasets × 20 seeds), after W1
finishes. Thresholds on CIFAR-100; CIFAR-10 is the consistency check. All
intervals are paired-bootstrap 95% CIs over seeds; W3 is a cost accounting and
does not add accuracy p-values.

- **H-W3a (cut saving survives the pass count):** MF-Seq's useful
  bytes-to-stopping are ≤0.50× BP's on both datasets, while its passes are
  ≥3.0× BP's; at the same time its per-pass useful cut bytes are ≤0.50× BP's.
  Supported only if both ratios are below 0.50, the bytes-to-stopping ratio's
  95% CI upper bound is below 0.50, and the pass-ratio CI lower bound is above
  3.0. This is the quantitative version of "the per-pass saving is not fully
  eaten by extra passes."
- **H-W3b (dead M0 forward is a material cut cost):** M0's full forward through
  W1..W3 is ≥50% of MF-Seq's **as-implemented** bytes-to-stopping on C100.
  MF-Seq-nom0 should remove that block, reducing as-implemented cut bytes by
  ≥50%; accuracy is governed separately by H-W1c (<0.5 pp drop = supported,
  0.5–1.5 pp = intermediate, ≥1.5 pp withdraws the dead-stage claim). The pilot
  value is 64%, so a much lower confirmatory share would itself be informative.
- **H-W3c (prefix caching is intermediate, not dominant):** caching the frozen
  prefix with augmentation disabled would save **15–50%** of MF-Seq's useful
  bytes-to-stopping on C100. Below 15% caching is second-order; above 50% it is
  first-order and must be discussed as such. The pilot is 25.3% (C100) and
  22.1% (C10). The canonical CIFAR run cannot use this cache because random
  crop/flip makes stale prefixes; the counterfactual is reported separately.
- **H-W3d (cut bytes are not an energy proxy; descriptive, no threshold):**
  across the available arm×dataset cells, report the **sign and magnitude** of
  Spearman ρ between arm-level cut bytes-to-stopping and energy-to-stopping,
  with a paired seed bootstrap 95% CI and both mean and median inputs. The seed
  bootstrap conditions on the fixed six cells; because a common seed rescaling
  can preserve the cell ordering, a cell bootstrap is also reported as a
  sensitivity. There is **no one-sided threshold** and no "ρ < 0.5" decision
  rule. With 12 cells at most
  (six arms × two datasets), and with two datasets sharing the same seed block,
  the effective sample is small and the cells are not independent; the result is
  a descriptive check, not a confirmatory test. A negative sign means higher cut
  bytes do not track higher energy — the expected direction, because MF-Seq has
  the smallest cut bill but the largest energy. The pilot (three arms × two
  datasets, n=6) gives ρ = −0.543; this is a pilot value only and is not used for
  a p-value.

### W3.4 Endpoints and statistics

- **Primary:** useful and as-implemented bytes-to-stopping, per-pass bytes by
  cut k=1..L and by MF-Seq stage, and stage shares; arm means with 95%
  bootstrap CIs; paired-by-seed ratios where arms share seeds.
- **Secondary:** M0 dead-forward share, cached-prefix saving, and the **sign of
  Spearman ρ** between cut bytes and measured energy, with a small-n caveat. The
  W1 panel has at most 12 cells and only two datasets, so ρ is descriptive; no
  one-sided threshold, no confirmatory p-value.
- **Legacy cross-check:** the old total-memory-traffic table is retained under
  an explicit "memory traffic, not communication" heading; it is never mixed
  with cut bytes. Parameter reads, parameter-gradient reads, optimizer state
  and validation traffic are outside the cut definition.
- **No new significance claims on the G1 pilot.** G1 cut numbers are descriptive
  engineering checks (BP 2.16 GB/pass; MF-Simul 1.08; MF-Seq useful 0.36 per
  prefix cut; M0 1.08; cache saving 22–25%); they may be cited as pilot values
  only, never as confirmatory p-values.
- **Source of truth:** W1 per-run JSON + own trace for passes/energy; W1 run
  JSON `stage_epochs` for per-stage epochs (G1 stdout only as the frozen
  fallback); `scripts/jpdc/w3_cut_communication.py` (commit hash recorded at
  freeze) for cut bytes.

### W3.5 Materials and run plan

- Zero new GPU runs: CPU hooks on the canonical model + W1 artifacts.
- Pilot (already done, exploratory): G1 160 runs, output in
  `jpdc_data/g1/w3_cut/`.
- Confirmatory: re-run the same script on W1 artifacts after the W1 freeze;
  record the script hash and the W1 commit in this addendum's §5.
- Cached-prefix case: a separate counterfactual with augmentation disabled and
  one fill per stage; it is never merged into the canonical bill.
- Validation: a GPU/NCCL smoke is **not** required for W3 because the estimand
  is logical cut bytes, not measured interconnect traffic; if added later it is
  a diagnostic, not a confirmatory endpoint.

### W3.6 What W3 will not claim

- Not total memory traffic, not a Roofline model, not a hardware-counter
  validation, and not an energy predictor. The old 69.1 GB/pass number is
  memory traffic, not communication.
- No CNN/transformer cut claims (W5), no multi-GPU communication claims (W2).
- No claim that cut bytes cause accuracy; W3 is a cost decomposition.
- No claim that caching is free in the canonical augmentation pipeline; the
  cached case is a counterfactual, not a deployable default.

---

## 4. Open decisions for the author (before freeze)

1. **W2 target:** use each arm's own final measured accuracy (current default)
   or wait for a common-threshold trace that G1 does not contain. A common
   threshold would require per-epoch test evaluation and is out of scope for the
   frozen W1 runs.
2. **W2 overlap assumption:** serialized compute/communication (current
   conservative default) vs an overlap fraction. Overlap is not measured at this
   scale; any overlap would only reduce the slow-link communication penalty.
3. **W3 optimizer sensitivity:** add a one-off sensitivity that includes Adam
   state traffic (≈6× parameter bytes per step), or keep it excluded? Default:
   excluded from the primary, reported as a single sensitivity number.
4. **W3 validation pass:** include the per-epoch validation forward (~10% of a
   training forward, no backward) in the primary, or exclude it? Default:
   exclude, state the bound; it is common to all arms.

## 5. Amendments log

| date | change | by |
|---|---|---|
| 2026-10-21 | Draft created; no confirmatory W2/W3 run scheduled. W3 G1 accounting labelled exploratory pilot. | agent |
| 2026-10-21 | **W3 redefined as cut-boundary communication** (0.36 GB per epoch per direction per cut; BP 2.16 GB/pass, MF-Simul 1.08, MF-Seq useful 0.36 per prefix cut, M0 dead forward 1.08). The old 69.1 GB/pass is relabelled **memory traffic** and retained only as a legacy table. Hypotheses H-W3a–d rewritten for cut bytes; H-W3d now predicts cut bytes are *not* an energy proxy. Cached-prefix case separated; CIFAR augmentation makes it a counterfactual. Pilot outputs `jpdc_data/g1/w3_cut/`, script `scripts/jpdc/w3_cut_communication.py`. | agent |
| 2026-10-22 | **W2 four-GPU training design superseded.** Replaced by (a) the zero-GPU link-rate model `scripts/jpdc/w2_link_rate_model.py` over measured passes, per-step compute time, per-run energy and W3 cut bytes at 100 Mbit/s, 1 Gbit/s and 10 Gbit/s; (b) one short 1 MB NCCL point-to-point diagnostic; (c) an optional n=3 layer-parallel demo that is not scheduled. No 4-GPU training runs, no 160-run W2 budget. Seq-Global is explicitly a planning proxy until W1. Pilot model output: `jpdc_data/g1/w2_link/`. | agent |
| 2026-10-22 | **H-W3d changed before any W1 result** from a one-sided `ρ < 0.5` threshold to reporting the **sign** of Spearman ρ with a 95% bootstrap CI and an explicit small-n caveat (≤12 non-independent arm×dataset cells). Pilot n=6 ρ = −0.543; pilot only. | author |
| 2026-10-22 | **Ex-M0 multipliers labelled estimates.** The stage-0 audit values (energy 2.69×/2.92×, passes 3.05×/3.35×, time 3.03×/3.35×) are implementation-dependent estimates from the G1 pilot, not measurements of a production pipeline; the W3 output and paper text must say so. | agent |
| 2026-10-22 | **W3 per-stage source updated before any W1 result.** W1 per-stage epochs are read from the run JSON `stage_epochs` added by the freeze instrumentation, not from stdout. G1's already-frozen artifacts keep their stdout fallback; the sum is still asserted against JSON `epochs_completed`. No outcome or threshold change. | agent |
