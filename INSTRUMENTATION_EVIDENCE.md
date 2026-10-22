# Instrumentation evidence — pre-W1 freeze (2026-10-22)

This note is evidence for the post-training diagnostic endpoints added in the
2026-10-22 freeze. It is **not** a W1 result and does not enter any decision
rule, Holm family, or threshold.

## 1. The probes do not change final weights

The protected property is exact equality of every state-dict tensor, not merely
an unchanged accuracy. The tests run the same seed twice, train one copy with
the probes and one without, and compare the final states with
`torch.equal` for every key.

Commands:

```bash
PYTHONPATH=. python3 -m pytest -q \
  tests/test_secondary_endpoints.py::test_secondary_endpoints_preserve_bp_training_and_weights \
  tests/test_secondary_endpoints.py::test_secondary_endpoints_preserve_mf_training_and_weights
```

Evidence:

- `test_secondary_endpoints_preserve_bp_training_and_weights` — BP, seed 123:
  clean-trained model and probe-exposed model have identical state dicts; the
  probe-exposed state is also unchanged before → after the endpoint call.
- `test_secondary_endpoints_preserve_mf_training_and_weights` — MF, seed 123:
  same two comparisons for the staged/local architecture.
- `test_secondary_endpoints_support_mf_and_heads` — the MF probe returns
  `a_1..a_3` and `M_0..M_3` without mutating state.
- `test_mf_stage_epochs_are_recorded` — `stage_epochs` is populated before the
  probes, so the stage-budget source is the run JSON.

## 2. The probes run after the NVML window closes

`src/training/engine.py` enters `with monitor:` only around the training call,
then evaluates the primary test pass and only afterwards calls
`compute_secondary_endpoints`; `_finalize_run` may call the idempotent
`monitor.stop()` again, but sampling has already stopped.

A runtime ordering test pins this:

```bash
PYTHONPATH=. python3 -m pytest -q \
  tests/test_secondary_endpoints.py::test_engine_runs_probes_after_monitor_exit
```

The fake-monitor event sequence asserted by the test is exactly:

```text
monitor_enter → train → monitor_exit → test → secondary → finalize
```

Every endpoint result also records
`secondary_eval_scope = "post_training; outside NVML energy window"`. The
`secondary_eval_seconds` field measures only the endpoint call; the wall
endpoint uses `training_duration_sec`, so the probes cannot alter the wall
comparison either.

## 3. Added seconds per run per arm (controlled synthetic check)

`scripts/jpdc/instrumentation_overhead.py` runs the exact endpoint code on a
fixed 5,000-row CIFAR-shaped probe set with the canonical 3×2000 MLP, twice per
arm, on CPU. The authoritative W1 numbers will be each run's own
`secondary_eval_seconds`; this table is the pre-freeze order-of-magnitude check.

```bash
PYTHONPATH=. python3 scripts/jpdc/instrumentation_overhead.py \
  --out-dir ../jpdc_data/g1/instrumentation --repeats 2
```

Measured 2026-10-22 (local CPU; means of 2):

| arm | mean seconds | min | max | state unchanged |
|---|---:|---:|---:|---|
| BP | 2.831 | 2.829 | 2.833 | yes |
| MF-Joint | 2.883 | 2.822 | 2.945 | yes |
| MF-Simul | 2.788 | 2.769 | 2.808 | yes |
| MF-Seq | 2.825 | 2.795 | 2.854 | yes |
| Seq-Global | 2.797 | 2.788 | 2.807 | yes |
| MF-Seq-nom0 | 2.810 | 2.805 | 2.814 | yes |

All six rows reported `secondary_eval_scope = post_training; outside NVML
energy window`, and `state_unchanged=True`. The synthetic check is CPU-only;
the real per-run seconds will be reported from each run JSON, including any
GPU-specific probe cost.

## 4. What is added to a run JSON

- `secondary_eval_scope`, `secondary_eval_probe_*` metadata
- `final_train_loss/accuracy`, `final_val_loss/accuracy`,
  `final_test_loss/accuracy`
- `linear_probe_test_accuracy` (`a_1..a_L`)
- `effective_rank` (`a_1..a_L`)
- `local_head_test_accuracy` (`M_i`, where present)
- `secondary_eval_seconds`
- `stage_epochs` and `stopping_epoch` in `_run_stats`

`stage_epochs` is written before the probes; the W1 per-stage pass source is the
run JSON, with G1 stdout retained only as the frozen fallback. The probes never
write to run JSON before training finishes.
