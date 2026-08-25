# Repository Optimization Review

**Date:** 2026-08-25 · **Branch:** `refactor/pspyra/refactoring-foundation` @ `e061711`
**Scope:** Full review of `src/beyond_backprop/`, `scripts/`, tests, and tooling for
possible optimizations, conducted after the canonical refactoring series and the
four follow-up cleanup commits (`0daffac`..`e061711`).

**Ground rule applied throughout:** this repository backs a published paper.
Every recommendation below is classified by whether it can change reported
numbers. Findings marked **[safe]** are behavior-preserving by construction;
**[verify]** items are plausible wins that require an equivalence check before
adoption; **[do-not-do]** items look like wins but must stay untouched.

---

## Executive summary

| ID | Area | Finding | Impact | Class |
| --- | --- | --- | --- | --- |
| P1 | Performance | Scalar NaN/finite checks allocate CPU tensors on hot paths | Small, free | [safe] |
| P2 | Performance | Per-batch NVML polling inside training steps duplicates the background sampler | Small–medium | [verify] |
| P3 | Performance | Per-batch `.item()` accumulation forces GPU sync every step | Small–medium | [verify] |
| P4 | Performance | FF multi-pass inference runs `num_classes` sequential forwards per batch | Large (eval) | [verify] |
| D1 | DRY / fairness | Early stopping re-implemented inline in FF/MF/CaFo despite a shared tested class | Maintainability risk | [verify] |
| R1 | Robustness | Legacy `save_checkpoint` swallows all exceptions → silent checkpoint loss | Data-loss risk | [safe] to fix |
| R2 | Robustness | `NvmlResourceMonitor.start()` swallows failures without logging | Diagnosability | [safe] |
| R3 | Robustness | TOCTOU in `create_directory_if_not_exists` | Trivial | [safe] |
| T1 | Testing | Coverage gap concentrated in the four full training loops (56% overall) | Highest-leverage test work | — |
| Q1–Q5 | Code quality | Dead locals, same-module wrappers, global logging sentinel, module grab-bag | Hygiene | [safe] |

---

## 1. Performance

### P1. Tensor-allocating scalar checks on hot paths **[safe]**

`torch.isnan(torch.tensor(x))` / `torch.isfinite(torch.tensor(x))` construct and
destroy a CPU tensor to answer a plain-float question:

- `algorithms/ff.py:336, 359, 431, 471` (line 336 runs **once per training batch**
  when NVML is active)
- `algorithms/mf.py:203`
- `algorithms/bp.py:105, 126, 138, 154, 160`

Replace with `math.isnan(x)` / `math.isfinite(x)`. Numerically identical
(`torch.isnan` on a 0-d CPU float tensor returns exactly what `math.isnan`
returns for the same double). Zero-risk micro-win; also removes ~9 pointless
tensor allocations per epoch.

### P2. Inline NVML polling duplicates the background monitor **[verify]**

The FF/MF trainers query `get_gpu_memory_usage(gpu_handle)` once per batch
(`ff.py:333-337`, `mf.py:~200`) even though `monitoring/nvml.py` already runs a
dedicated sampling thread at a fixed interval for the runner-level resource
snapshot. The inline poll adds an NVML round-trip into the step loop.

Options, cheapest first:
1. Throttle the inline poll to `log_interval` boundaries (memory *logging*
   already only happens there; only peak tracking uses every sample).
2. Drop the inline sampling entirely and take `peak_memory_mib` from the
   runner's `NvmlResourceMonitor` snapshot.

Caveat: option 2 changes the sampling rate of the trainer's reported
per-epoch peak memory (0.2 s thread cadence vs. per-batch), which can lower
observed peaks. This field feeds `FF_Hinton/Peak_GPU_Mem_Epoch_MiB`; if any
reported number depends on it, treat as a measurement-definition change and
document it — otherwise adopt.

### P3. Per-step `.item()` synchronization **[verify]**

In `ff.py` (lines 322-352 region) and similarly in CaFo/MF, per-batch losses
are converted with `.item()` every step to accumulate epoch sums, and again at
each `log_interval`. Each `.item()` forces a device sync. Accumulating
`loss.detach()` tensors and reducing at epoch end removes most syncs.

Caveat: the NaN/Inf guard (`ff.py:265-276`) intentionally reads loss values
per batch to skip bad batches; keep that check (it can use `torch.isfinite`
on-device without `.item()`, branching before the sync). Floating-point
accumulation order changes if you switch from Python floats to tensor sums —
epoch-average metrics could differ in the last ULP. Given the reproducibility
contract, only adopt if bit-equality of logged epoch metrics is verified or
explicitly waived.

### P4. FF inference: one forward pass per class candidate **[verify]**

`evaluate_ff_model` (`ff.py:576-644`) loops `for label_candidate in
range(num_classes)` and performs a full forward goodness pass per candidate —
10× the forward work per batch on CIFAR-10, 100× on CIFAR-100. Candidates
could be stacked along the batch dimension and evaluated in a single forward,
since layers act row-wise.

This is the single largest eval speedup available, but it changes peak
activation memory by ~`num_classes`× and interacts with the custom
`ReLU_full_grad` autograd function. It must be validated by asserting
bit-identical predicted labels against the loop version on a fixed seed
before adoption. Until then, leave as-is.

### P5. Duplicate artifact writes **[safe, leave as-is]**

`training/runner.py` writes byte-identical `summary.json` + `run_summary.json`
and config pairs every run. This is the documented compatibility-alias policy
(OBS decision register); cost is kilobytes per run. Do not "optimize" away.

## 2. DRY and code quality

### D1. Early stopping exists twice **[verify — highest-value refactor]**

BP uses the canonical, unit-tested `training/early_stopping.EarlyStopping`
(`bp.py:69-78`). FF (`ff.py:121-155, 430-470`), MF and CaFo (`cafo.py:364-531`)
re-implement patience/mode/min-delta logic inline — verbatim-preserved legacy
code. The methodology requires early stopping to be *identical across
algorithms* for fairness; right now it is identical-by-history, not
identical-by-code.

Recommended path (three steps, each its own commit):
1. Write characterization tests that feed identical synthetic metric streams
   (including NaN epochs, exact-tie improvements, first-epoch values) through
   both the inline logic and `EarlyStopping`, asserting equal stop decisions.
2. Swap the inline implementations for `EarlyStopping` behind those tests.
3. Delete the duplicated branches.

Do **not** fold this into any other change: if equivalence fails, the
discrepancy is itself a finding about the published protocol.

### D2. Same-module compatibility wrappers with no callers **[safe]**

`generate_ff_hinton_inputs` and `get_linear_cooldown_lr` (`ff.py:45-68`) wrap
`ff_math` functions and have no callers outside `ff.py` (verified by grep over
`src/` and `tests/`). Call the `ff_math` functions directly and delete the
wrappers, or move them behind the legacy-shim boundary if history matters.

### D3. Dead locals in `profiling.py` **[safe]**

`conv_hook` assigns then immediately deletes `output_shape`
(`profiling.py:46-47`); nothing ever uses it. Remove the dead locals. (The
`forward_gflops` / `estimated_fwd_gflops` key duplication is an intentional
artifact alias — keep it.)

### D4. `utils/training_support.py` is a grab-bag **[safe, low priority]**

Logging setup + W&B setup + metric formatting + NVML queries + checkpointing
in one module (verbatim-migrated legacy). The canonical boundaries
(`monitoring/`, `tracking/`, `checkpointing/`) already exist; the long-term
direction is to shrink this module to nothing as trainer bodies modernize. No
urgent action; do not migrate blindly mid-experiment.

### D5. Process-global logging sentinel **[safe]**

`setup_logging` gates reconfiguration on `os.environ["LOGGING_SETUP_COMPLETE"]`
(`training_support.py:199-227`). This breaks log-file redirection when several
runs execute in one process (e.g., in-process Optuna trials). Replace the env
sentinel with a module-level flag or an explicit `force` parameter.

### D6. `scripts/tuning_utils` sys.path shim **[safe, acceptable]**

The `_common.py` helper is imported via a small `sys.path.insert` shim so the
scripts run standalone. Acceptable for research utilities; if a fifth script
ever appears, promote `_common` into `beyond_backprop` (write-side config
tooling) instead of growing the shim.


## 3. Robustness

### R1. Silent checkpoint loss in the legacy saver **[safe] to fix**

`utils/training_support.save_checkpoint` catches **all** exceptions from
`torch.save` and continues after logging (`training_support.py:63-77`). A full
disk silently yields a missing best-model checkpoint while the run reports
success. `CheckpointManager` already does this correctly (atomic temp-file +
`os.replace` + raise). Migrate FF/MF/CaFo checkpoint writes onto
`CheckpointManager` (or at minimum re-raise). Constraint: legacy best-checkpoint
filenames are load-bearing for restart paths (decision MIG-002) — preserve them.

### R2. NVML start failures are invisible **[safe]**

`NvmlResourceMonitor.start()` reduces any failure to `self._nvml = None`
(`nvml.py:45-47`) without logging. A driver hiccup then looks identical to
"no GPU". Add one warning log — measurement semantics are unchanged (the
snapshot still reports `measured=False` with source `nvml-unavailable`).

### R3. TOCTOU in directory creation **[safe]**

`create_directory_if_not_exists` does `exists` + `makedirs`
(`training_support.py:28-36`). Use `os.makedirs(path, exist_ok=True)`.

### R4. `weights_only=False` in `CheckpointManager.load` **[safe, document]**

Required for legacy payloads carrying non-tensor objects; artifacts are locally
produced and trusted. Worth a comment marking it a trust boundary so nobody
copies the pattern for untrusted inputs.

## 4. Testing

### T1. The four training-loop bodies are the coverage frontier

56% overall; `cafo.py` (8%), `ff.py` (9%), `mf.py` (11%),
`experiment/factories.py` (0%). Only
`tests/integration/test_algorithm_adapters.py` exercises the trainer functions
today. The highest-leverage investment is a fast synthetic end-to-end test per
algorithm (tiny random `DataLoader`, 2 epochs, CPU) driving the real
`train_*_model` functions — it lifts coverage, protects the fairness-critical
loops, and is a prerequisite for safely doing D1, P2, and P3. Second: direct
tests for `experiment/factories.py`.

### T2. CI niceties (optional)

Cache pip wheels and upload the coverage report as a CI artifact so coverage
deltas are visible per PR.

## 5. Things that look like optimizations but must not be done **[do-not-do]**

- **Batching FF candidate forwards (P4)** without a bit-equivalence proof.
- **Changing MNIST's fixed 50k/10k split** to a seeded random split for
  consistency with other datasets — it is the published protocol.
- **Unifying "Effective Epochs" counting** into the runner — the semantic
  difference between algorithms (MF's sum-of-layer-epochs vs. BP's global
  epochs) is intentional and documented.
- **Rewriting the algorithm math modules** (`*_math.py`, well tested at
  89–92%) into "cleaner" vectorized forms.
- **Removing dual artifact names** (`summary.json`/`run_summary.json`) — they
  are documented restart/analysis compatibility surfaces.

## Suggested order of execution

1. P1, R2, R3, D3 — trivially safe hygiene/perf fixes.
2. T1 synthetic training-loop tests (unlocks everything below).
3. R1 checkpoint-saver migration behind the new tests.
4. D1 early-stopping characterization tests + unification (fairness-critical).
5. P2/P3/P4 performance items, each with measured before/after numbers and an
   explicit note on any reported-number impact.

---

*Findings verified against the working tree at commit `e061711`; line numbers
refer to that revision and may shift as fixes land.*


