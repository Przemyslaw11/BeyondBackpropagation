# End-to-End Validation Report (post-refactor series @ `9dcff31`)

**Date:** 2026-08-25 · **Branch:** `refactor/pspyra/refactoring-foundation`
**Validated tree:** `9dcff31` · **Golden baseline:** worktree of `e061711`
(`/tmp/bbb-baseline`, last pre-refactor commit)

## Verdict summary

| Item | Scope | Status | Evidence |
| --- | --- | --- | --- |
| R-A | FF stacked inference, cross-tree | **PASS** (CPU + MPS) / cuBLAS **BLOCKED** | 3 seeds × 2 devices: 100% prediction agreement, Δaccuracy = 0.0 pp |
| R-B | Real NVML paths | **BLOCKED** (no CUDA/NVML on this host); degraded path **PASS** | `measured=False`, source `nvml-unavailable`, R2 warning fires |
| R-C | Early-stopping timing vs legacy | **PASS** | stop epochs identical across trees, all loops × patience 1–4 |
| R-D | CaFo epoch metrics (P3) | **PASS** | bit-exact recomputation; cross-tree bit-equality to first validation |
| R-E | Checkpoints (R1) | **PASS** | legacy filenames/payloads intact; atomicity tests green |
| R-F | Logging redirection (D5) | **PASS** | new pinned regression test `tests/unit/test_setup_logging_force.py` |

**No production code changes were required.** No refactor-introduced defect was
found on any check executable on this host.

## Hardware / software

- Host: Apple M2 (macOS, darwin), 10-core GPU — **no NVIDIA driver, no NVML,
  `torch.cuda.is_available() == False`**
- Python 3.13.7 · torch 2.13.0 · pynvml installed but `NVMLError_LibraryNotFound`
- Accelerators actually exercised: CPU and **MPS** (Metal). MPS is a genuinely
  different matmul backend with its own rounding behavior, so it serves as a
  *same-risk-class proxy* for the cuBLAS concern in R-A — it does **not**
  substitute for it (see handoff below).

## Methodology

All comparisons ran the same probe scripts against both trees, selecting the
tree via `PYTHONPATH` with an import-root assertion (`--expect-root`) so a
wrong-tree result is impossible:

1. **Black-box FF evaluation probe:** prediction marginals recovered exactly by
   evaluating once per constant-label loader (`acc_v ⇒ #(pred=v)`), plus one
   natural-label accuracy run, per seed/device/tree. Evaluation goodness depends
   only on RNG-free positive inputs, so outputs are deterministic given
   weights+data (the P4 RNG-stream deviation cannot mask differences here).
2. **White-box row-invariance probe:** stacked pass vs verbatim candidate loop,
   same weights, measuring max abs goodness diff and argmax flips directly.
3. **Scripted-evaluator early-stopping probe:** evaluator patched to a flat
   constant stream; expected stop at exactly `patience + 1` evaluator calls
   (legacy `bad >= P` boundary with an initial improvement from the ±inf seed).
4. **Real-run probe:** 2–3 genuine training epochs per algorithm in both trees
   with `checkpointing.checkpoint_dir` set; every `log_metrics` dict captured to
   JSONL; checkpoint directory inventoried. Comparison rule follows documented
   deviation note 2: bit-equality up to and including the first validation
   entry (wall-clock duration keys excluded), structural equality after.

## Measurements

### R-A. FF stacked inference

White-box row-invariance (5 seeds × 64×10 candidates each):

| Device | argmax flips | max abs goodness diff | bitwise-equal batches |
| --- | --- | --- | --- |
| CPU | 0/3200 | 0.0 | 5/5 |
| MPS | 0/3200 | 0.0 | 5/5 |

Bitwise row-invariance holds even on Metal — the stacked pass and the candidate
loop produce identical goodness tensors, not merely equal argmaxes.

Black-box cross-tree `evaluate_ff_model` (prediction histograms + accuracy;
seeds 0–2; current tree vs `e061711` baseline):

| Seed | Device | Histogram (cur) | Histogram (base) | Acc cur | Acc base |
| --- | --- | --- | --- | --- | --- |
| 0 | cpu | [1, 0, 45, 2] | [1, 0, 45, 2] | 29.1667 | 29.1667 |
| 1 | cpu | [0, 38, 10, 0] | [0, 38, 10, 0] | 29.1667 | 29.1667 |
| 2 | cpu | [0, 2, 31, 15] | [0, 2, 31, 15] | 18.75 | 18.75 |
| 0–2 | mps | identical to cpu rows | identical | = | = |

Agreement 100% everywhere; accuracy delta 0.0 pp (seed noise band observed:
29.17 → 18.75 across seeds, far above any tree delta).

### R-B. NVML paths

This host has no NVML. Verified the degraded path end-to-end:
`NvmlResourceMonitor.start()` logs the new R2 warning
(`NVML monitoring unavailable (NVMLError_LibraryNotFound ...)`), and `stop()`
returns `ResourceSnapshot(measured=False, source='nvml-unavailable')`.
`Peak_*_Mem_Epoch_MiB` population from real sampling, plausibility vs device
memory, log-boundary-only per-batch keys, and `measured=True` remain **to be
verified on the GPU host** (handoff below).

### R-C. Early stopping timing

Stop epochs (evaluator-call counts), flat metric stream:

| Algorithm | P=1 | P=2 | P=3 | P=4 | Legacy boundary (`bad >= P`) |
| --- | --- | --- | --- | --- | --- |
| ff (max) | 2=2 | 3=3 | 4=4 | 5=5 | match ×4 |
| mf_matrix_only (min) | 2=2 | 3=3 | 4=4 | 5=5 | match ×4 |
| cafo_predictor (min) | 2=2 | 3=3 | 4=4 | 5=5 | match ×4 |

Format: `current = baseline`. All 12 combinations identical across trees and
equal to the legacy rule. (MF W-phase equivalence remains pinned by
`tests/integration/test_early_stopping_equivalence.py`; the ±inf unification
deviation stays intentional and test-pinned.)

### R-D. CaFo epoch metrics

Current-tree run (seed 0, predictor phase, 4 batches/epoch):
batch losses `[1.1243259906768799, 1.1521719694137573, 1.1198253631591797,
1.1287740468978882]`; logged `Train_Loss_EpochAvg = 1.1312743425369263` —
**bit-exact** match with local recomputation (plain left-to-right mean).
Accuracies `[12.5, 25.0, 25.0, 25.0]` → logged `21.875`, bit-exact. All values
finite. Cross-tree: metrics.jsonl entries bit-identical through the first
validation entry; key sets structurally identical afterwards (0 mismatches).

### R-E. Checkpoints

Real runs with `checkpointing.checkpoint_dir` set; inventories identical across
trees; `torch.load` succeeds on every file; zero dotfile leftovers.

- FF: `ff_checkpoint_epoch_{1..N}.pth` = full dict `{epoch, state_dict,
  optimizer, best_metric_value, val_accuracy}` ✓; `ff_<exp>_best.pth` = raw
  state_dict ✓ (re-loaded by the trainer's post-training best-model reload)
- MF: `mf_matrix_M0_complete.pth`, `mf_layer_1_complete.pth` =
  `{state_dict, layer_trained_index}` ✓
- CaFo: `cafo_predictor_0_complete.pth` =
  `{state_dict, predictor_index, epochs_trained}` ✓

Failure behavior is pinned by existing unit tests
(`tests/unit/test_save_checkpoint.py`): raise-instead-of-swallow and
failed-best-save-leaves-existing-checkpoint-intact are green.

### R-F. Logging redirection

`setup_logging(log_file=A)` then `setup_logging(log_file=B, force=True)`:
A receives only pre-switch records, B only post-switch records, B is not even
created without `force`. Pinned by `tests/unit/test_setup_logging_force.py`
(new in this change).

## Handoff: what still must run on the CUDA host

Two items are **blocked here**, not failed. Exact pass/fail procedure:

```bash
git worktree add /tmp/bbb-baseline e061711   # if not present
pip install -e ".[dev,gpu]"

# R-A on cuBLAS (black-box, ~1 min): run the FF histogram probe
# (see "Probe scripts" below) on both trees:
for s in 0 1 2; do
  python3 probe_eval_ff.py --device cuda --seed $s --out /tmp/ffa_cur_s$s.json
  PYTHONPATH=/tmp/bbb-baseline/src python3 probe_eval_ff.py \
    --device cuda --seed $s --out /tmp/ffa_base_s$s.json --expect-root /tmp/bbb-baseline
done
# Pass: prediction_counts identical pairwise; accuracy delta within seed noise.
# If agreement < 99% or systematic deltas appear: implement the commented
# row-chunking upgrade in evaluate_ff_model (chunk boundaries must never split
# a sample's candidates), keeping tests/integration/test_ff_eval_batching.py green.

# R-B (one FF + one CaFo config with energy monitoring enabled):
# shrink epochs/hidden_dims in a copy of the config first.
# Check: no get_gpu_memory_usage exceptions; Peak_GPU_Mem_Epoch_MiB > 0 and
# <= device memory; per-batch mem keys only at log boundaries;
# NvmlResourceMonitor.stop().measured == True; epochs shorter than
# log_interval yield exactly one sample (last batch) and a plausible peak.
```

### Probe scripts

The four standalone probes (~100 lines each, no repo changes needed) used for
this validation are available in the validation session under `/tmp/e2e-val/`:

- `probe_eval_ff.py` — black-box prediction-histogram extraction from
  `evaluate_ff_model` (device/tree selectable)
- `probe_row_invariance.py` — stacked-vs-loop goodness diff and argmax flips
- `probe_train.py` — scripted-ES stop epochs / real-run JSONL capture +
  checkpoint inventory
- `compare_runs.py` — bit-equality-up-to-first-validation comparator

Copy them onto the GPU host next to the two worktrees to repeat R-A/R-B there.

## Conclusion

Everything executable outside a CUDA host passes, most of it at bit-level
strictness against the golden baseline. The refactor holds. Remaining exposure
is concentrated in exactly the two items the risk register predicted (cuBLAS
rounding under P4, real-NVML sampling under R-B); both have ready-to-run,
pass/fail-crisp procedures above. No resolution-log status changes and no
protocol decisions arise from this validation.


