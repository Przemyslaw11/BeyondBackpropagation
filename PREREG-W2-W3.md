# PREREG-W2-W3.md — JPDC extension, communication and byte accounting

**Status: draft, 2026-10-21. Not frozen. No confirmatory W2/W3 run has been
scheduled.** This is a separate addendum to `PREREG-phase2.md`, as required by
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
change the **inter-worker communication** bill when the 3×2000 MLP is split
across GPUs? W1 varies locality on one GPU; W2 asks whether locality is a
communication advantage, an accuracy advantage, or both.

This is not a scaling study. It is a controlled accounting experiment at the
smallest distribution that exposes every layer boundary (3–4 ranks).

### W2.2 Design

- **Deployment:** layer-parallel pipeline, one hidden layer per rank:
  rank 0 holds the input adapter + W1, rank 1 W2, rank 2 W3, rank 3 the readout
  head. Activations are sent forward between ranks; gradients are sent backward
  only where the graph requires them. Micro-batch = 128; one optimizer step per
  micro-batch; no activation checkpointing.
- **Arms:** BP, MF-Simul, MF-Seq, Seq-Global. Same configs as W1, same seeds.
- **Datasets:** CIFAR-10 and CIFAR-100.
- **Instrumentation:** (i) analytic hook counts from the W3 counter, mapped onto
  rank boundaries; (ii) measured NCCL point-to-point/all-reduce bytes from
  `torch.distributed` counters; (iii) CUDA-event communication time and total
  step time. The analytic and measured bills must agree within 5%; if they do
  not, the measured bill is authoritative and the discrepancy is reported.
- **Runs:** n=20 seeds per arm × dataset for accuracy and bytes-to-accuracy;
  communication microbenchmarks repeated 5× per seed and summarised by median.
  Total 160 training runs, 4-GPU jobs. A n=5 pilot (seeds 42–46) is exploratory.

### W2.3 Hypotheses and frozen decision rules

All thresholds on CIFAR-100; CIFAR-10 is the consistency check. Bytes are
inter-worker bytes per optimizer step, summed over the run; "matched passes"
means both arms use their own Protocol A stopping, and the comparison is also
repeated at BP's realised pass count.

- **H-W2a (locality lowers communication):** at matched passes, BP's
  inter-worker bytes are ≥2.0× MF-Seq's, with the 95% CI lower bound >1.5×.
  Reading: frozen prefixes remove backward traffic across layer boundaries and
  are a genuine communication lever, not only an accuracy lever.
- **H-W2b (communication-adjusted edge):** MF-Seq reaches BP's final CIFAR-100
  accuracy with **no more** total inter-worker bytes than BP spent to reach its
  own final accuracy (ratio ≤1.0, CI upper bound <1.2). Reading: even after
  paying 3.5–4.5× the passes, the per-pass communication saving offsets the
  pass count; if the ratio is >1.2, the accuracy edge is communication-expensive
  and must be reported as such.
- **H-W2c (scale boundary):** at 4 ranks and this width, communication time is
  <20% of step time for every arm (CI upper bound <25%). Reading: at the paper's
  scale, single-GPU energy/pass results are not communication-limited; W2 is a
  boundary statement, not a multi-node scaling claim. If the share is ≥25% for
  any arm, that arm's W2 numbers are reported descriptively and no scale-free
  claim is made.

Decision matrix: H-W2a and H-W2b can both hold (locality saves bytes and the
accuracy edge is cheap); H-W2a without H-W2b means the saving is real but eaten
by extra passes; neither means communication is not a differentiator at this
scale. No relabeling of W1.

### W2.4 Endpoints and statistics

- **Primary:** inter-worker bytes per pass and bytes-to-BP-accuracy, measured
  NCCL counters; paired by seed, Wilcoxon + 10k bootstrap; Holm over
  {H-W2a, H-W2b}.
- **Secondary:** communication-time share, step time, NVML energy, wall time,
  final accuracy; descriptive CIs, no Holm.
- **Diagnostics:** analytic-vs-measured byte agreement; per-boundary byte
  breakdown (forward activation, backward gradient, all-reduce); overlap
  fraction.
- **Exclusions:** no optimizer-state communication (ZeRO/FSDP is out of scope);
  no multi-node runs; no CNN.

### W2.5 Materials and run plan

- Implementation: a `src/distributed/` layer-parallel wrapper around the
  existing `MF_MLP`; one process per GPU, `torch.distributed` NCCL, no
  checkpointing (same `BBP_DISABLE_CHECKPOINTS=1` hygiene).
- First deliverable is the analytic bill on CPU (no GPU): map W3's per-layer
  tensor traffic onto the 4-rank cut and publish the predicted boundary bytes.
  Only if the analytic bill shows a ≥1.5× separation is the GPU experiment
  worth scheduling.
- GPU stage: n=5 pilot; freeze this addendum and the wrapper commit; then n=20.
- Budget/gate: W2 enters the JPDC claims only if the G2 scope gate (20 Dec 2026)
  selects it and 4-GPU time is available. Otherwise it is reported as a
  boundary calculation with the analytic bill only.

### W2.6 What W2 will not claim

- No multi-node scaling efficiency, no interconnect comparison, no FSDP/ZeRO.
- No claim that communication causes the MF-Seq accuracy edge; W2 measures cost,
  W1 measures attribution.
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
  `epochs_completed` for BP/MF-Simul, per-stage stdout epochs for MF-Seq (a
  missing W1_M1 early-stopping line means the 100-epoch cap; the stage sum is
  asserted against JSON).
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
- **H-W3d (cut bytes are not an energy proxy):** across the 12 arm×dataset
  cells, arm-level cut bytes-to-stopping and energy-to-stopping have Spearman
  ρ < 0.5. If ρ ≥ 0.8, cut bytes are a useful energy proxy and the paper says
  so; 0.5–0.8 is intermediate. This hypothesis is deliberately the opposite of
  the old total-memory-traffic hypothesis: cut bytes exclude parameter reads
  and dead M0 work, which are exactly the terms that make MF-Seq expensive.

### W3.4 Endpoints and statistics

- **Primary:** useful and as-implemented bytes-to-stopping, per-pass bytes by
  cut k=1..L and by MF-Seq stage, and stage shares; arm means with 95%
  bootstrap CIs; paired-by-seed ratios where arms share seeds.
- **Secondary:** M0 dead-forward share, cached-prefix saving, and Spearman ρ
  between cut bytes and measured energy.
- **Legacy cross-check:** the old total-memory-traffic table is retained under
  an explicit "memory traffic, not communication" heading; it is never mixed
  with cut bytes. Parameter reads, parameter-gradient reads, optimizer state
  and validation traffic are outside the cut definition.
- **No new significance claims on the G1 pilot.** G1 cut numbers are descriptive
  engineering checks (BP 2.16 GB/pass; MF-Simul 1.08; MF-Seq useful 0.36 per
  prefix cut; M0 1.08; cache saving 22–25%); they may be cited as pilot values
  only, never as confirmatory p-values.
- **Source of truth:** W1 per-run JSON + own trace for passes/energy; W1 stage
  stdout for per-stage epochs; `scripts/jpdc/w3_cut_communication.py` (commit
  hash recorded at freeze) for cut bytes.

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

1. **W2 rank map:** 4 ranks (input+W1 / W2 / W3 / head) vs 3 ranks (W1 / W2 /
   W3+head). Default in this draft: 4 ranks, so every boundary is explicit.
2. **W2 baseline:** include a data-parallel BP baseline as a reference, or only
   the layer-parallel arms? Default: layer-parallel only; data-parallel is a
   different question.
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
