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

## W3 — byte accounting

### W3.1 Question

How many bytes does each training rule move through the network per data pass,
and to stopping, at the canonical 3×2000 CIFAR MLP? Which part is unavoidable
gradient/activation traffic, which part is parameter streaming, and which part
is the dead M0 stage or frozen-prefix recomputation?

W3 is an accounting/engineering workstream, not a new accuracy experiment. Its
confirmatory data are the W1 runs; the G1 accounting already produced
(`jpdc_data/g1/w3/`, 2026-10-21) is an **exploratory pilot** and is not used for
confirmatory p-values.

### W3.2 Estimand and accounting rules

- **Logical tensor traffic** through Linear, ReLU, and M-head operations:
  - forward: activation input read + activation output write + parameter read;
  - backward: grad-output read + grad-input write (only where the graph needs
    it) + parameter-gradient accumulate read+write;
  - no optimizer state/update traffic, no validation pass, no allocator
    metadata, no host-device copies.
- **Per pass** = one epoch over 45,000 training samples at batch 128 = 352
  batches. **Bytes-to-stopping** = per-pass bytes × measured passes: JSON
  `epochs_completed` for BP/MF-Simul, per-stage stdout epochs for MF-Seq (a
  missing W1_M1 early-stopping line means the 100-epoch cap; the stage sum is
  asserted against JSON).
- **Cuts k=1..L:** for MF-Seq, stage M0, W1, W2, W3 each get their own per-pass
  bill; BP and MF-Simul are full-network bills. The k-th cut is the stage that
  trains W_k against the frozen prefix W_1..W_{k-1}.
- **GB = 1e9 bytes.** Report total, activation-only, and parameter-only columns.
- **Cached-prefix assumption:** the canonical config is
  `activation_cache: recompute`; CIFAR random crop/flip makes a cached frozen
  activation stale, and the code rejects a cache strategy under stochastic
  transforms. The primary bill therefore recomputes the prefix every epoch. A
  cached-prefix counterfactual (augmentation disabled, one fill per stage) is
  reported separately as a bound, never as the primary number.
- **M0:** the code runs a full no-grad forward through W1..W3 before taking
  a_0 in `train_mf_matrix_only`; the primary bill includes that traffic. It is
  dead compute: no prediction reads M0 (pinned by `tests/test_m0_dead_stage.py`).

### W3.3 Hypotheses and frozen decision rules

Confirmatory on the W1 family (six arms × two datasets × 20 seeds), after W1
finishes. Thresholds on CIFAR-100; CIFAR-10 is the consistency check.

- **H-W3a (per-pass saving survives the pass count):** MF-Seq's per-pass bytes
  are ≤0.75× BP's on both datasets, while its passes are ≥3.0× BP's, and its
  bytes-to-stopping ratio is **below its pass ratio** (i.e., the per-pass saving
  is not fully eaten by extra passes). Supported if all three hold with
  non-overlapping 95% CIs.
- **H-W3b (M0 is a material byte cost):** M0 accounts for ≥15% of MF-Seq's
  bytes-to-stopping on C100; MF-Seq-nom0 reduces bytes-to-stopping by ≥15% and
  passes by ≥15% versus MF-Seq, with an accuracy drop <0.5 pp. A drop of 0.5–1.5 pp
  is intermediate (report both readings); a drop ≥1.5 pp withdraws the dead-stage
  claim, exactly as in H-W1c. This is the byte-side counterpart of H-W1c.
- **H-W3c (prefix caching is second-order):** caching the frozen prefix
  (augmentation disabled) would save ≤15% of MF-Seq's **total** bytes-to-stopping
  on C100. If the saving is >25%, prefix recomputation is a first-order cost and
  must be discussed as such; 15–25% is reported as intermediate with both
  readings.
- **H-W3d (bytes track energy):** across the 12 arm×dataset cells, arm-level
  bytes-to-stopping and energy-to-stopping have Spearman ρ ≥0.8. If ρ <0.5,
  bytes are not a useful energy proxy and the paper reports them as orthogonal
  cost axes; 0.5–0.8 is intermediate.

### W3.4 Endpoints and statistics

- **Primary:** bytes-to-stopping (total and activation-only), per-pass bytes,
  stage shares; arm means with 95% CI, paired-by-seed differences where the arms
  share seeds.
- **Secondary:** parameter-only traffic, M0 share, cached-prefix saving,
  Spearman ρ with energy.
- **No new significance claims on the G1 pilot.** G1 W3 numbers are descriptive
  engineering checks (they already show M0 ≈20%, cache saving ≈12%); they may be
  cited as pilot values only, never as confirmatory p-values.
- **Source of truth:** W1 per-run JSON + own trace for passes/energy; W1 stage
  stdout for per-stage epochs; `scripts/jpdc/w3_byte_accounting.py` (commit hash
  recorded at freeze) for bytes.

### W3.5 Materials and run plan

- Zero new GPU runs: CPU hooks on the canonical model + W1 artifacts.
- Pilot (already done, exploratory): G1 160 runs, output in `jpdc_data/g1/w3/`.
- Confirmatory: re-run the same script on W1 artifacts after the W1 freeze;
  record the script hash and the W1 commit in this addendum's §5.
- Validation: one GPU smoke that measures one batch's NCCL/PCIe traffic is not
  required for W3 (single-GPU logical traffic); if added later, it is a
  diagnostic, not a confirmatory endpoint.

### W3.6 What W3 will not claim

- Not a Roofline model, not a hardware-counter validation, not an energy
  predictor; optimizer state/update traffic is excluded by design.
- No CNN/transformer byte claims (W5), no multi-GPU communication claims (W2).
- No claim that bytes cause accuracy; W3 is a cost decomposition.

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
