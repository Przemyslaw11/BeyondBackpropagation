# E2E Validation & Fix Agent — Full Context Prompt

Copy everything below the line into the agent's task.

---

## Role and mission

You are working in `/Users/pspyra/Desktop/projects/MasterThesis/BeyondBackpropagation`,
branch `refactor/pspyra/refactoring-foundation`, HEAD `9dcff31`.

This repository backs a **published paper** (Forward-Forward, Mono-Forward,
CaFo vs BP baselines). A large behavior-preserving refactoring series (11
commits) has just been completed and verified by unit/integration tests on a
**CPU-only macOS host**. Your mission: **validate the refactored training and
evaluation loops end-to-end on real hardware (CUDA GPU + NVML), diagnose and
fix anything that regressed, and produce an evidence-based report.**

You are NOT redesigning anything. You are proving the refactor holds outside
the lab, and fixing defects if it does not.

## Non-negotiable ground rules

1. **Never change reported numbers silently.** Any change that alters logged
   metrics, stop timing, checkpoint formats, or measurement semantics must be
   either (a) provably equivalent, or (b) explicitly documented as a protocol
   deviation in `docs/optimization-review.md` (resolution log at the bottom).
2. **Do-not-touch list** (from the review, still binding):
   - Do not rewrite `*_math.py` modules or the algorithm math.
   - Do not change MNIST's fixed 50k/10k split.
   - Do not unify "Effective Epochs" counting semantics between algorithms.
   - Do not remove dual artifact names (`summary.json`/`run_summary.json`).
   - Do not migrate legacy checkpoint writes onto `CheckpointManager`
     (payload-format compatibility for restart paths, decision MIG-002).
   - Legacy checkpoint filenames (`ff_<exp>_best.pth`,
     `mf_matrix_M*_complete.pth`, `mf_layer_*_complete.pth`,
     `cafo_predictor_*_complete.pth`) are load-bearing.
3. Every fix is its own commit, message references the finding ID
   (`E2E-n:` prefix), and the full suite (`make check`) stays green.
4. New tests go under `tests/`; GPU-dependent tests must use the existing
   `@pytest.mark.gpu` marker.

## What was changed in the refactor (commits in order)

| Commit | Change | E2E risk |
| --- | --- | --- |
| `e98bc47` | P1: `math.isnan/isfinite` replace tensor-wrapped scalar checks (14 sites in ff/mf/bp/cafo); R2: NVML start-failure warning; R3: atomic dir creation; D3: dead locals removed | Trivial; smoke-level |
| `d315761` | T1: synthetic trainer-loop tests + factories tests | None (test-only) |
| `f2beaa3` | R1: legacy `save_checkpoint` is now atomic (temp file + `os.replace` + fsync) and **raises** on failure instead of swallowing | Checkpoint writes under real `checkpointing.checkpoint_dir`; disk-full now aborts runs (intended) |
| `5bfa343` | D1 step 1: characterization tests pinning legacy early-stopping rule | None (test-only) |
| `26baf0c` | D1 step 2: all four trainers (FF main loop, MF matrix-only, MF W-phase, CaFo predictor) use canonical `EarlyStopping(patience=max(P-1,0))`. Stop boundary `bad >= P` preserved exactly. ±inf now counts as a bad epoch everywhere (documented deviation) | Real early-stopping trigger timing |
| `4ef23e6` | P2: FF and both CaFo phases sample NVML only at `is_log_time`/`is_last_batch` instead of every batch (MF already did) | `Peak_*_Mem_Epoch_MiB` population and plausibility on real NVML |
| `878f35f` | P3: CaFo DFA-blocks and predictor phases defer `.item()` batch stats to epoch end, same addition order → bit-identical epoch metrics. FF/MF untouched (their per-batch NaN guards already sync every step) | CaFo epoch loss/accuracy values |
| `f3f2f8a` | P4: `evaluate_ff_model` evaluates all label candidates in ONE stacked forward pass (`repeat_interleave(images, C)` x `arange(C).repeat(B)` labels → goodness `.reshape(B, C)`), replacing `num_classes` sequential passes | **Highest risk**: bitwise row-invariance held on CPU, but cuBLAS may round differently across batch shapes → argmax ties could flip |
| `9dcff31` | R4 trust comment; D2 wrapper deletion (calls go directly to `ff_math.generate_hinton_inputs` / `linear_cooldown_lr`); D5 logging sentinel env-var → module flag + `force=`; T2 CI cache/coverage | Low |

## Known intentional deviations from byte-level legacy reproduction

These are accepted and test-pinned. Do NOT "fix" them back:

1. **±inf early-stopping metrics** count as bad epochs in all algorithms now
   (previously only NaN did; +inf could reset patience in max mode).
2. **FF evaluation consumes global torch RNG differently** (one
   `generate_hinton_inputs` call per batch instead of one per candidate).
   Consequence: any training run diverges from pre-refactor runs *after its
   first validation evaluation* because DataLoader shuffling draws from the
   same global RNG. Equivalence checks against the old code are therefore
   only bit-exact **up to the first validation point**; beyond that compare
   structurally (same metric keys, plausible ranges, equal stop epochs for
   identical metric streams).

## Risk register — what to verify and how

### R-A. FF stacked inference on CUDA (from P4) — highest priority
- Bitwise row-invariance was proven on CPU
  (`tests/integration/test_ff_eval_batching.py`, 10 tests, passing). On
  cuBLAS, matmul results for different M dimensions may differ in last-bit
  rounding; predictions near goodness ties could flip.
- Procedure: with a fixed seed, obtain an FF model (train briefly or load a
  checkpoint), then run `evaluate_ff_model` under both current code and the
  pre-refactor code (see "Golden baseline") on the SAME weights and loader.
  Compare predicted-label agreement rate and eval accuracy delta.
- Pass criterion: agreement ~100% and accuracy delta inside seed-noise
  (establish noise via 3 baseline seeds). Investigate only if deltas are
  systematic (>~0.5 pp) or agreement <99%.
- Confirm eval activation memory at CIFAR scale fits: the stacked pass
  multiplies peak activations by ~num_classes. A chunking upgrade path is
  commented in `evaluate_ff_model` — implementing row chunking IS an
  acceptable fix if OOM appears (keep equivalence tests green; chunk
  boundaries must never split a sample's candidates).

### R-B. Real NVML paths (from P2 + R2)
- Run one FF and one CaFo config with energy monitoring enabled on the GPU
  host. Verify: no exceptions from `get_gpu_memory_usage`;
  `FF_Hinton/Peak_GPU_Mem_Epoch_MiB` / CaFo equivalents positive and <=
  device memory; per-batch GPU-mem keys appear only at log boundaries;
  `NvmlResourceMonitor.stop()` reports `measured=True`.
- Verify epochs SHORTER than log_interval still get exactly one sample (last
  batch) and a plausible peak.

### R-C. Early stopping timing (from D1)
- For each algorithm run a short config with early stopping enabled and
  patience small enough to trigger. Confirm trigger epochs match pre-refactor
  code given the same validation stream. The harness in
  `tests/integration/test_early_stopping_equivalence.py` shows how to script
  evaluators via monkeypatching — reuse it for a GPU run if useful.

### R-D. CaFo epoch metrics (from P3)
- Train CaFo 2–3 epochs; confirm `Train_Loss_EpochAvg` / `Train_Acc_EpochAvg`
  are finite and consistent with per-batch postfix values; ideally assert
  bit-equality of epoch averages against local recomputation over captured
  batch losses.

### R-E. Checkpoints (from R1)
- Run FF/MF/CaFo with `checkpointing.checkpoint_dir` set. Verify expected
  filenames appear, `torch.load` accepts them, best payloads keep legacy
  shape (epoch file = full dict; best file = raw state_dict), no dotfile
  temp files linger. Simulate write failure ONLY in a unit-style check —
  the run must fail loudly rather than silently lose the best checkpoint.

### R-F. Logging redirection (from D5)
- In one Python process call `setup_logging(log_file=A)` then
  `setup_logging(log_file=B, force=True)`; B receives records, A stops.
  (No live callers today; prove the API works.)

## Golden baseline methodology

Create a read-only worktree of the pre-refactor code:

```bash
git worktree add /tmp/bbb-baseline e061711
```

- `e061711` = last commit before any refactor work (the review doc commit
  `169891e` touched docs only).
- Run the SAME config with the SAME seed in both trees. Expect bit-equality
  of logged metrics up to each algorithm's first validation evaluation;
  after that, structural equality only (deviation note 2 above).
- Configs live in `configs/{ff,mf,cafo,bp_baselines}/`; shrink `epochs`,
  `hidden_dims`, and dataset size for smoke passes before full-scale runs.
- Entry points: `make check`, `make test-fast`, `make test-cov`,
  `python -m beyond_backprop.cli.main --help`, `scripts/run_experiment.py`.

## Environment

- Linux + NVIDIA GPU + `pynvml==12.0.0` (extra `gpu`) required for R-B;
  without NVML the monitor degrades to `measured=False` plus a warning
  (that warning is new — R2 — and is expected in its absence).
- Python >= 3.10. Install: `pip install -e ".[dev,gpu]"`.
- Suite state at `9dcff31`: 138 tests green on CPU. `make check` =
  ruff format-check + lint + mypy + fast tests. Keep all four green.

## Fix policy

- Defect in refactor → minimal fix in refactored code + regression test,
  commit prefixed `E2E-n:`.
- Genuinely ambiguous behavior (legacy bug vs intended protocol) → do NOT
  pick silently. Pin both readings with a characterization test, leave
  production unchanged, flag it in the report for human protocol decision
  (precedent: the D1 off-by-one).
- GPU performance regressions → measure with `torch.cuda.synchronize()`-
  bounded timers, report numbers, no speculative optimization.

## Deliverables

1. Report covering every risk-register item (append under the resolution log
   in `docs/optimization-review.md` or as `docs/e2e-validation-report.md`):
   PASS/FAIL, hardware, driver/CUDA/torch versions, measurements, seeds.
2. Fixes as separate commits (`E2E-n:`), suite green after each.
3. Resolution-log rows updated for anything changed.
4. If all PASS: say so plainly with evidence, and change nothing else.


| Commit | Change | E2E risk |
| --- | --- | --- |
| `e98bc47` | P1: `math.isnan/isfinite` replace tensor-wrapped scalar checks (14 sites in ff/mf/bp/cafo); R2: NVML start-failure warning; R3: atomic dir creation; D3: dead locals removed | Trivial; smoke-level |
| `d315761` | T1: synthetic trainer-loop tests + factories tests | None (test-only) |
| `f2beaa3` | R1: legacy `save_checkpoint` is now atomic (temp file + `os.replace` + fsync) and **raises** on failure instead of swallowing | Checkpoint writes under real `checkpointing.checkpoint_dir`; disk-full now aborts runs (intended) |
| `5bfa343` | D1 step 1: characterization tests pinning legacy early-stopping rule | None (test-only) |
| `26baf0c` | D1 step 2: all four trainers (FF main loop, MF matrix-only, MF W-phase, CaFo predictor) use canonical `EarlyStopping(patience=max(P-1,0))`. Stop boundary `bad >= P` preserved exactly. ±inf now counts as a bad epoch everywhere (documented deviation) | Real early-stopping trigger timing |
| `4ef23e6` | P2: FF and both CaFo phases sample NVML only at `is_log_time`/`is_last_batch` instead of every batch (MF already did) | `Peak_*_Mem_Epoch_MiB` population and plausibility on real NVML |
| `878f35f` | P3: CaFo DFA-blocks and predictor phases defer `.item()` batch stats to epoch end, same addition order → bit-identical epoch metrics. FF/MF untouched (their per-batch NaN guards already sync every step) | CaFo epoch loss/accuracy values |
| `f3f2f8a` | P4: `evaluate_ff_model` evaluates all label candidates in ONE stacked forward pass (`repeat_interleave(images, C)` x `arange(C).repeat(B)` labels → goodness `.reshape(B, C)`), replacing `num_classes` sequential passes | **Highest risk**: bitwise row-invariance held on CPU, but cuBLAS may round differently across batch shapes → argmax ties could flip |
| `9dcff31` | R4 trust comment; D2 wrapper deletion (calls go directly to `ff_math.generate_hinton_inputs` / `linear_cooldown_lr`); D5 logging sentinel env-var → module flag + `force=`; T2 CI cache/coverage | Low |
