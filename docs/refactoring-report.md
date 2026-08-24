# Refactoring report

Captured on 2026-08-22 after the canonical refactoring sequence and extended on
2026-08-23 with the self-containment migration (Phases A–D). The frozen
baseline was commit `9b86bdc` on the refactoring branch, with 44 YAML
configurations and 66 CPU tests passing. The completed implementation preserves
the published values, `source_tex/`, `slurm_logs/`, and dataset protocols.

## Paper identification

`arXiv:2509.19063v1` is the authoritative identifier for this work. Existing
repository citations to `arXiv:2511.01061v1` (README badge and BibTeX entry)
are retained as cross-references and were deliberately not rewritten; see
decision SCI-002 in the refactoring decision register.

## Before and after architecture

The baseline workflow was centered on legacy scripts and modules under
`src/utils`, `src/data_utils`, `src/architectures`, `src/algorithms`, and
`src/training`. Configuration parsing, device policy, experiment lifecycle,
optional services, and file-system orchestration were interleaved with several
algorithm implementations.

The canonical workflow is now organized under `src/beyond_backprop/`:

```text
config -> runtime -> data -> architecture -> algorithm adapter -> runner
                                      \-> evaluator
runner -> monitoring / tracking / artifacts / checkpointing
CLI and legacy scripts -> canonical commands
```

Typed configuration, runtime metadata, data loading, architecture construction,
algorithm adapters, evaluation, tuning, monitoring, tracking, and artifacts
have explicit boundaries. Historical imports and scripts remain compatibility
entry points and translate into the canonical services.

## Preserved scientific behavior

- BP uses global cross-entropy with AdamW and retains its best-state and legacy
  checkpoint naming behavior.
- FF retains local goodness training, the downstream classifier stage, and its
  inference path.
- CaFo retains block and predictor stages, freezing, stopping, aggregation, and
  inference.
- MF retains M0/layer isolation, detached preceding activations, local
  validation, projection, frozen-layer behavior, and inference.
- Dataset identity, split rules, transforms, worker behavior, download policy,
  and published result values were not rewritten.

The canonical adapters expose the shared lifecycle without forcing these four
algorithms into one mathematical loop. Dependency-light math modules and
numerical regression tests make the extracted formulas independently testable.

## Intentional corrections

Strict validation now rejects incorrect scalar types, unknown keys, invalid
cross-field combinations, and incompatible monitoring/device settings before
data loading. Worker seeds and loader generators are derived deterministically.
Optional W&B, NVML, CodeCarbon, and Optuna integrations are lazy and degrade to
explicit no-op or unavailable results on CPU-only installations.

Canonical runs persist artifacts by default under the selected backend result
directory. Training-only monitoring excludes setup and evaluation, and profiling
labels forward and BP-update GFLOPs with estimated/measured provenance. SLURM
arrays validate the full manifest, use caller-provided resources, and propagate
the experiment exit status.

## Tests and verification

The test suite is organized by scope under `tests/{unit,integration,regression,
smoke,fixtures}`; pytest markers (`slow`, `gpu`, `smoke`) remain the selection
authority for the standard commands. It covers all four lifecycles,
configuration compatibility, all 44 repository YAML files, deterministic
seeding, checkpoint compatibility, best-state restoration, frozen-layer
gradients, detached activations, service failure cleanup, dry runs, legacy shim
imports/scripts, tuning, artifacts, profiling, and SLURM shell behavior. The
final CPU/non-slow/non-GPU run reports 84 passing tests.

The acceptance checks also cover formatting/linting, canonical-package type
checking, compilation of `src` and `scripts`, import-tree checks, and Git
whitespace validation. No test requires a dataset download, GPU, CUDA, W&B,
NVML, CodeCarbon, or Optuna. GPU-specific measurements remain optional and were
not claimed by the CPU acceptance run.

Coverage baseline (2026-08-25, `make test-cov`, 97 tests): **56%** of
`beyond_backprop` statements overall. Well-covered regions include the
algorithm math modules (89–92%), contracts/config/runtime (73–99%), and the
monitoring adapter interfaces (76–100%, with hardware-specific NVML paths
lower). The dominant gap is the full-scale training-loop bodies in
`algorithms/{cafo,ff,mf}.py` plus `experiment/factories.py` (~1,100
statements below 15%), which only execute under slow/full-dataset runs;
raising non-monitoring coverage toward 80% requires additional synthetic
full-lifecycle tests and is recorded as future work rather than silently
claimed. Future PRs must not decrease the overall percentage.

## Reproducibility

Each run records Python, PyTorch, TorchVision, CUDA/MPS availability and
identity when available, driver information when available, hostname, Git
state, command line, seed, and resolved configuration hash. Resolved YAML,
metadata, metrics, event history, summary, logs, checkpoints, and profiling
outputs are persisted together. Configuration precedence is typed defaults,
base YAML, experiment YAML, and explicit `--set` overrides.

## Measurement limitations

NVML energy and memory values are available only when compatible hardware and
the optional dependency are present. CPU and MPS runs therefore report an
explicit unavailable measurement rather than an inferred GPU value. Forward
GFLOPs and BP-update GFLOPs are estimates unless a measured profiler is used;
their provenance remains distinguishable in the artifacts. Wall-clock timing
covers the canonical training region only, so setup and evaluation costs must
not be compared using that field.

## Known remaining limitations

The self-containment migration is complete: architectures and trainer bodies
live only under `beyond_backprop`, the legacy experiment engine, BP baseline
trainers, Optuna objective modules, and all remaining `src/*` re-export shims
(including `scripts/_migrate_b2.py`) have been deleted. Historical `src.*`
import paths stop working by design (decision MIG-004); use
`beyond_backprop.*` and the canonical CLI commands instead. Synthetic CPU tests
do not substitute for full-scale GPU performance runs, and optional
integrations still require their own extras and credentials when enabled.

## Migration instructions

Use the canonical commands:

```bash
python -m beyond_backprop experiment run --config CONFIG
python -m beyond_backprop tune --config CONFIG --n-trials 3
python -m beyond_backprop batch --config-dir configs
python -m beyond_backprop validate-config --config CONFIG
python -m beyond_backprop inspect-config --config CONFIG
```

Add `--dry-run` to validate and inspect execution without data loading or
training. Use `--set general.backend=local` for local execution. Existing
`scripts/run_experiment.py`, `scripts/run_local_array.py`, and
`scripts/run_optuna_search.py` remain argument-translation shims. SLURM users
provide account, partition, and resource settings through their environment or
submission command; the repository no longer embeds a personal allocation.
