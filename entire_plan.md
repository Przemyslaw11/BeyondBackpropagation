## Plan: Camera-Ready Rebuild — Fair Re-Run From Scratch (v3)

Re-execute the entire experimental matrix under a harmonised fairness protocol, build a plotting pipeline from scratch, and rewrite the paper for the 15-page LNCS camera-ready by **2 November** (58 days). Ordering follows reviewer criticality, but the fairness fixes land first because every number depends on them.

Changed in v3: the activation cache becomes an explicit **time–memory frontier** (Option B), which grows the ablation ladder from four rungs to six and reframes the $O(L^2)$ fix from a patch into a finding — caching is *enabled by* gradient locality and is structurally unavailable to BP, BP-DS and MF-Joint.

Budget: **~111 GPU-h (~155 with contingency)** — 62 h of hyperparameter search, 49 h of final runs.

Headline risk, stated plainly: harmonised early stopping will likely *reduce* MF's 40.78 % energy advantage while the cache fix *increases* it. The two are attributed separately in step 20 or the result is uninterpretable.

---

### Phase 1 — Zero-compute fixes and safety net (Week 1, parallel with Phase 2)

1. **Fix the Figure 3 subfloat label swap** in `main.tex` — `CaFo_2.png` is Mean Validation *Loss* but labelled Accuracy, and vice versa. A factual error in the accepted PDF.
2. **Delete two unsupported caption claims** — the `WiMj` convention sentence on Fig 2 (the legend contains only BP and FF) and the "step-like pattern" assertion on Fig 4(c).
3. **Back-port thesis text** from `main.tex`: NVML expansion (closes R3a verbatim), the "hardware-validated" definition (R4a), Appendix A architecture and optimiser details (R2b), Appendix B environment including **driver 570.86.15**, which governs NVML power semantics.
4. **Back-port the CaFo results table** from `4_ExperimentsAndResults.tex` lines 125–210 as a *safety net*. Closes R3b immediately; Phase 4 overwrites it. If the CaFo re-runs slip, the paper is still complete.
5. **Add `lee2015deeply`** to the bibliography — cited in §6.4, absent from every bib file in the repo.

---

### Phase 2 — Fairness infrastructure (Week 1–2, blocks all runs)

Nothing may run until this is merged and tested.

6. **Harmonise early stopping** in `configs` — one metric (validation loss), `min_delta` 0.0, one patience, one max-epoch cap per configuration group. Currently BP patience ranges 5→40, FF stops on *accuracy* with `min_delta` 0.01, and MF uses patience 4 against BP's 20 on the CIFAR-10 headline. Move shared keys into `base.yaml` and strip per-config overrides.
7. **Harmonise the pruner** to `None` for all algorithms in `tuning`, offset by a reduced HPO epoch budget applied identically. Median-for-all is rejected: truncating MF mid-layer is not the same event as truncating BP mid-epoch.
8. **Derive the Optuna sampler seed per study** from a hash of algorithm + dataset + architecture in `run_optuna_search.py` (~line 132). The shared seed 42 is why four sets of byte-identical hyperparameters exist across supposedly independent searches.
9. **Equalise search spaces** — add `weight_decay` to the MF space (currently pinned at 0.0), widen FF's learning-rate range from ~1.4 to BP's 3 decades. *Depends on step 8.*
10. **Fix the BP baseline asymmetry** in `_create_mlp_model` at `engine.py:68-73` — the MF-family BP baseline instantiates `MF_MLP` regardless of `is_bp_baseline`, carrying 90,720 dead `projection_matrices` that receive AdamW state and weight decay. Mirror the `nn.Sequential` reconstruction already used for the FF family at `engine.py:44-66`.
11. **Extend `GPUEnergyMonitor`** in `monitoring.py` with `nvmlDeviceGetUtilizationRates`, `nvmlDeviceGetTemperature`, `nvmlDeviceGetClockInfo`, `nvmlDeviceGetComputeRunningProcesses`, **plus `psutil` process RSS**, and emit a per-run CSV. Today it samples only power and memory, so three of four hardware traces in the paper are W&B system panels. RSS is mandatory under Option B — without it the host-cache variant appears free and a reviewer will correctly call that cheating. *Parallel with steps 6–10.*
12. **Implement three MF cache strategies behind one config flag** at `mf.py:418-423`: `recompute` (current $O(L^2)$ behaviour, unchanged), `cache_device`, `cache_host`. The flag is the only thing that varies across ladder rungs 4–6. Caching is only sound because detachment freezes layers $0 \ldots i-1$ during layer $i$'s training — encode that precondition as an assertion so the flag cannot be enabled for a joint-gradient algorithm.
13. **Fix the SLURM array bugs** in [scripts/slurm_scripts/run_array.slurm](scripts/slurm_scripts/run_array.slurm) — `--array=1-25` against 27 config entries silently dropped the last two MF configs, and `mnist_mlp_3x1000_SGD_ref.yaml` does not exist. Convert to a 2-D array over config × seed via `EXPERIMENT_SEED` ([src/training/engine.py](src/training/engine.py#L177)), which needs no Python change.
14. **Write `tests/test_fairness_invariants.py`** — assert early-stopping settings, pruner policy, batch size, max epochs and data split are uniform within each config group, and that no two configs share tuned hyperparameters unintentionally. Turns fairness from a prose claim into a CI check. *Depends on steps 6–9.*

---

### Phase 3 — The critical experiments: six-rung ladder at n = 20 (Week 2–4)

This is R2c and R2d — *is the efficiency advantage caused by locality, or by confounds?*

| Rung | Readout | Aux losses | Gradients | Can cache? | Isolates |
|---|---|---|---|---|---|
| 1. BP | `output_layer` | none | global | no | reference |
| 2. BP-DS | `output_layer` | each layer via $M_i$ | global | no | auxiliary supervision |
| 3. MF-Joint | $M_L$ | each layer via $M_i$ | global | no | readout / extra parameters |
| 4. MF-recompute | $M_L$ | each layer via $M_i$ | **local, sequential** | — | **locality, direct effect** |
| 5. MF-cache-device | $M_L$ | each layer via $M_i$ | local, sequential | yes | **locality's enabled optimisation** |
| 6. MF-cache-host | $M_L$ | each layer via $M_i$ | local, sequential | yes | same, GPU-memory-neutral |

Rungs 1–3 cannot cache because joint gradients make any cached activation stale within one step. Rung 4 is the accepted paper's implementation, which keeps the comparison against rung 3 honest. BP → rung 6 is the full practical benefit.

15. **Add `src/baselines/bp_ds.py`** with `train_bp_ds_model` / `evaluate_bp_ds_model` mirroring the existing `train_bp_model` signature, importing `mf_local_loss_fn` from [src/algorithms/mf.py](src/algorithms/mf.py) **verbatim** so the loss is provably identical. Register in [src/algorithms/__init__.py](src/algorithms/__init__.py); branch in `get_model_and_adapter` at [src/training/engine.py](src/training/engine.py#L129).
16. **Write the gradient unit test before any cluster time is spent** — assert layer-0 weights receive non-zero gradient from the layer-$L$ auxiliary loss. If `forward_with_intermediate_activations` at `mf_mlp.py:138` detaches internally, BP-DS silently equals MF and the entire ablation is void. **Run this in week 1**, not week 2, so a failure has three weeks of slack. *Blocks step 17.*
17. **Tune four rungs independently** on the four MLP configurations — 16 studies, 50 trials, `pruner: None`. Rungs 5 and 6 **share rung 4's hyperparameters**: caching is mathematically identical to recomputation, so the optimisation trajectory is unchanged and separate searches would be meaningless. *Depends on Phase 2 and step 16.* ≈ 13 GPU-h.
18. **Run the ladder at n = 20** — 4 configs × 6 rungs × 20 seeds = 480 runs. ≈ 27 GPU-h. Twenty seeds buys a 0.25 pp equivalence margin at 80 % power and makes the internal comparison "does BP-DS equal MF?" decidable — that comparison is the Outcome B/C trigger, and at n = 7 it could not distinguish "equal" from "underpowered".
19. **Pre-register the analysis** before looking at results: paired-by-seed as primary (the shared seed set fixes the train/val split and data order; pairing roughly halves the required n), **Welch** rather than Student, **Holm** correction within each metric family, bootstrap 95 % CIs alongside every p-value, and **TOST at $\Delta = 0.25$ pp** for every parity claim. A non-significant p-value is never reported as evidence of equivalence.
20. **Run two diagnostic conditions on CIFAR-10** — harmonised early stopping with rung 4, and original early stopping with rung 5 — to attribute the change in the headline number to protocol versus implementation. ≈ 1 GPU-h.
21. **Two-stage $\sigma$ check.** After step 18, recompute $\sigma$ from 20 real observations and top up only if it demands more. All published effect sizes derive from n = 3, where the SD's 95 % CI spans roughly $[0.5\hat{\sigma},\ 5.7\hat{\sigma}]$, and the paper and thesis disagree by 5.6× on the CIFAR-10 SD. This resolves T11 automatically.

---

### Phase 4 — Remaining matrix (Week 4–5, parallel with Phase 5)

22. **Re-tune and re-run FF at n = 7** — the 13× claim is a headline and is affected by the early-stopping metric mismatch. Effects are large ($d \approx 6.6$ on accuracy), so 7 suffices. ≈ 13 h HPO + 5 h finals.
23. **Re-tune and re-run CaFo** — 8 configs, 30 trials, n = 5. Replaces the Phase 1 back-ported table. ≈ 24 h HPO + 10 h finals.
24. **Re-tune and re-run BP CNN and remaining BP MLP baselines** at n = 5–7. ≈ 12 h HPO + 6 h finals.
25. **Export W&B histories** from `przspyra11/BeyondBackpropagation` — used to confirm the instrument diagnosis and cross-check new against published numbers, not as a data source.

---

### Phase 5 — Plotting pipeline (Week 5–6, parallel with Phase 4)

No plotting code exists anywhere in the repo; the current figures are W&B UI exports, which is why they cannot be regenerated at higher DPI. Fig 4 currently renders at roughly **1.7 pt** effective type at `0.32\linewidth`.

26. **Data layer** — the extended monitor from step 11 emits a per-run time series (timestamp, power, GPU memory, utilisation, temperature, SM clock, process RSS); add a per-run summary record (config, algorithm, cache strategy, dataset, architecture, seed, test accuracy, wall time, energy, peak GPU memory, peak RSS, epochs run, stop reason). Write an aggregator producing one tidy long-format table that every figure and table draws from, so no number in the paper is transcribed by hand.
27. **Style contract in a single rcParams module** — generate at exact LNCS textwidth (122 mm) so `includegraphics` never scales; 8 pt base with a 7 pt tick floor; Okabe–Ito palette; linestyle *and* marker cycling so panels survive greyscale; no in-image titles (LaTeX captions only); legends outside the axes or shared across a figure; zero-based or explicitly broken axes (Fig 4b currently spans 23–30 °C, making 3 °C look dramatic); vector PDF output.
28. **The ladder waterfall (new, centrepiece)** — decomposes the total BP → MF-cache energy saving across five transitions, showing how much each of auxiliary supervision, readout choice, locality and caching contributes. This is the direct visual answer to R2c.
29. **The time–memory frontier (new)** — a scatter with BP as a single reference point and rungs 4, 5 and 6 tracing the frontier, with GPU memory on one axis and wall time on the other, and marker size or a second panel carrying host RSS. Three points make a frontier; two make a line segment. Visually the most striking figure the paper would have.
30. **The equivalence forest plot (new)** — per-configuration mean accuracy differences with bootstrap 95 % CIs and the $\pm 0.25$ pp equivalence band shaded. Converts the paper's weakest statistical claim into its most rigorous figure.
31. **Rebuild the four existing figures** — FF (2 panels), CaFo (2 panels, labels correct), MF (2 panels, NVML power trace replacing temperature), MF-vs-BP convergence against wall-clock time rather than epochs. Bootstrap CI bands over 20 seeds replace ±SD ribbons.
32. **Diagnostic plots not for the paper** — early-stopping and cache-strategy attribution from step 20, for the response letter.

---

### Phase 6 — Manuscript and response letter (Week 6–8)

33. **Rewrite §4 fairness protocols 2, 3 and 4** with an early-stopping settings table and a search-space table. Protocol 4's "consistent early stopping" and protocol 3's "applied to all algorithms for every configuration" are both unsupported by the configs.
34. **Add the ablation-ladder subsection** to §5, with the time–memory frontier presented as a consequence of locality rather than an implementation note.
35. **Update §6.4 limitations** — remove "three runs are insufficient", add the achieved equivalence margin, and add the **GPU-only energy accounting gap**: NVML cannot see the CPU and PCIe energy that rung 6 shifts off-instrument. Optional secondary cross-check via `codecarbon` 3.0.1, already in the stack.
36. **Write the response letter**, mapping every reviewer comment to a section, table or figure, and voluntarily disclosing the self-found defects. Reviewers did not raise them; disclosing is far stronger than being caught.

---

**Verification**

1. `tests/test_fairness_invariants.py` passes across all configs — the machine-checkable definition of "fair".
2. The BP-DS gradient test asserts non-zero layer-0 gradients before any job is submitted.
3. **Accuracy-identity check across rungs 4–6.** The three cache strategies must agree on accuracy to within run-to-run GPU nondeterminism at a fixed seed. Any real deviation is a cache bug. This is the single best correctness test for step 12.
4. The cache-strategy assertion rejects enabling a cache for any joint-gradient algorithm.
5. Re-run one published configuration under the *original* settings and confirm it reproduces the accepted numbers within CI — proves the from-scratch pipeline is not silently different.
6. Cross-check new NVML energy against exported W&B histories for the same configuration.
7. Confirm the SLURM array launches exactly `n_configs × n_seeds` jobs and all complete.
8. Every number in every table traces to the aggregated tidy table — no hand transcription.
9. Inspect every figure at 100 % zoom for ≥ 8 pt type, and in greyscale.
10. Compile with `llncs.cls`; confirm Fig 3's panels now match their captions.

---

**Decisions**

- **n = 20** for the ladder; **n = 7** for FF and remaining MLP baselines; **n = 5** for CaFo and CNN baselines.
- Parity claims use **TOST at $\Delta = 0.25$ pp**, paired-by-seed, Welch, Holm-corrected, with bootstrap CIs.
- Pruning disabled for **all** algorithms; equal trial budget with an identically reduced HPO epoch budget.
- Activation cache is **Option B**: three strategies reported as an explicit frontier. Rungs 5 and 6 share rung 4's hyperparameters. Process RSS is a first-class reported metric. fp16 caching rejected (precision confound).
- Both BP-DS and MF-Joint run; Outcome B/C reframing pre-approved.
- Phase 1 back-ports the thesis CaFo table as a safety net even though Phase 4 replaces it.
- **Page budget deferred** — revisit once real tables exist.
- **Excluded**: new datasets, new architectures, transformer or ImageNet-scale work, any claim about non-A100 hardware.

---

**Timeline**

| Week | Dates | Content |
|---|---|---|
| 1 | Sep 5–12 | Phases 1 and 2 in parallel; **gradient test early** |
| 2 | Sep 12–19 | Phase 2 completes; Phase 3 code; HPO opens |
| 3–4 | Sep 19 – Oct 3 | HPO campaign; ladder finals at n = 20; two-stage $\sigma$ check |
| 5 | Oct 3–10 | Phase 4 remaining matrix; Phase 5 data layer and style module |
| 6 | Oct 10–17 | Figures built; analysis tables generated |
| 7 | Oct 17–24 | Manuscript rewrite; response letter |
| 8 | Oct 24 – Nov 2 | Buffer, LNCS compliance, proofread, submit |

---

**Further Considerations**

1. **HPO is the only real bottleneck** — 62 of 111 GPU-h, sitting on the critical path in weeks 2–4. If the A100 queue is contended, the cut is to drop CaFo re-tuning (24 GPU-h) and keep the thesis hyperparameters for CaFo alone, disclosed in a footnote. CaFo is not a headline claim and the Phase 1 back-port already covers R3b.
2. **Step 16 can void Phase 3 entirely.** If `forward_with_intermediate_activations` detaches internally, BP-DS collapses onto MF and the ladder needs a different implementation path — hence running it in week 1.
3. **The cache work makes MF look better than the accepted paper claimed.** That is an unusual position in a rebuttal and should be framed deliberately: the previously reported advantage was *understated* because the original implementation recomputed the forward pass $O(L^2)$ times, which is itself the answer to R2d's "is the speedup an implementation artefact?"