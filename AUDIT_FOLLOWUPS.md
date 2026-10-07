# Audit Follow-ups

This ledger records the release audit decisions and items that require owner confirmation. Canonical scientific values were not changed.

## Step 0 inventory

The baseline was taken on `feature/ppam-2026-camera-ready` before creating `release/canonical-v2`. The tracked tree contains 8 config groups, `src/`, `scripts/`, `plots/`, and `tests/`; there are no tracked notebooks or `slurm_logs/` files. The twenty largest tracked blobs are reproduced at the end of this file; the largest is `src/plotting/figures.py`.

| Item not described or requiring clarification | What it is | References | Methodology status |
|---|---|---|---|
| `tests/` | Unit, plotting, backend, gradient, and fairness invariants | `pytest`, CI/user validation | Current methodology |
| `scripts/slurm_scripts/` | Present as an empty directory in the working tree; no tracked files | None found | Historical/empty |
| `scripts/tuning_utils/` | Present as an empty directory; no tracked files | No current references | Historical/empty |
| `configs/tuning/` | Present as an empty directory; no tracked files | No current references | Historical/empty |
| `src/tuning/` | Present as an empty directory; no tracked files | No current references | Historical/empty |
| `plots/` | Plotting source for generated figures; the non-canonical teaser was removed | README structure and plotting modules | Current methodology |
| `configs/reproduction/` | Published MF reproduction checks | README reproduction section | Current methodology |
| `CHANGELOG.md` | Neutral note about the methodology correction | Repository metadata | Current canonical release |

Deleted historical paths include tuning configs, Optuna/slurm scripts, and tuning utilities. No deleted sensitive-name path (`*COPYRIGHT*`, `License-to-Publish*`, or `master_thesis/`) was found.

## Security scan

- `gitleaks detect --source . --log-opts="--all"` scanned 74 commits and found no leaks.
- `trufflehog git file://. --only-verified` found 0 verified and 0 unverified secrets.
- The requested `git log --all -p | grep -E "ghp_|AKIA|xox[bap]-|wandb|api[_-]?key" -i` scan found only W&B/config/code references and generic API-key vocabulary; no token-shaped secret was found. These hits are harmless/false positives.
- Working-tree scans found no `/home/`, `/net/`, `/mnt/`, `/Users/`, PLGrid-account-like token, SLURM account, or email identity.
- No tracked notebooks or `slurm_logs/` existed to strip or inspect.
- No real secret or potentially sensitive credential was found; no security blocker was raised.

## Uniform stopping-rule review

`configs/base.yaml` defines validation-loss stopping with patience 20 and a 100-epoch per-training-unit cap. The following deviations are intentional protocol controls or legacy disclosure fields and were not changed:

| Config/key | Value | Intended? | Follow-up |
|---|---:|---|---|
| `configs/diagnostics/*_equal_epochs.yaml`: fixed epoch budget | config-specific matched-pass budget | Yes, Protocol B control | Owner confirm the budget is the realized BP pass count for its paired study |
| `configs/diagnostics/cifar10_mlp_3x2000_legacy_es.yaml`: `algorithm_params.epochs_per_layer` | legacy diagnostic value | Yes, diagnostic only | Owner confirm this remains archived as a stopping-policy sensitivity check |
| `configs/diagnostics/cifar10_mlp_3x2000_legacy_es.yaml`: legacy patience/min-delta | legacy diagnostic values | Yes, diagnostic only | Owner confirm it is excluded from canonical headline comparisons |
| `configs/*`: `legacy_hyperparameters` disclosure blocks | historical values | Yes, disclosure metadata | Owner confirm these blocks are not read by the training engine |

The complete tracked override inventory is:

| Configs | Key/value(s) | Intended? |
|---|---|---|
| `configs/bp_baselines/cifar10_cnn_3block_bp.yaml` | `early_stopping_patience: 40` | Owner confirm tuned baseline budget |
| `configs/bp_baselines/cifar100_cnn_3block_bp.yaml` | `early_stopping_patience: 20` | Owner confirm tuned baseline budget |
| `configs/bp_baselines/fashion_mnist_cnn_3block_bp.yaml`, `mnist_cnn_3block_bp.yaml`, `fashion_mnist_mlp_2x1000_bp.yaml`, `mnist_mlp_2x1000_bp.yaml` | `early_stopping_patience: 10` | Owner confirm tuned baseline budget |
| `configs/bp_baselines/cifar100_mlp_3x2000_bp.yaml` | `early_stopping_patience: 5` | Owner confirm tuned baseline budget |
| `configs/bp_baselines/cifar10_mlp_3x2000_bp.yaml` | `early_stopping_patience: 20` | Matches shared patience; owner confirm |
| `configs/bp_baselines/mnist_mlp_3x1000_bp.yaml` | `early_stopping_patience: 30` | Owner confirm tuned baseline budget |
| `configs/bp_baselines/mnist_mlp_4x2000_bp.yaml` | `early_stopping_patience: 40` | Owner confirm tuned baseline budget |
| `configs/mf/cifar10_mlp_3x2000.yaml` | `epochs_per_layer: 15`, patience 4, min_delta 0.0001 | Owner confirm canonical Protocol A schedule |
| `configs/mf/cifar100_mlp_3x2000.yaml` | `epochs_per_layer: 8`, patience 4, min_delta 0.0001 | Owner confirm canonical Protocol A schedule |
| `configs/mf/mnist_mlp_2x1000.yaml`, `*_cache_device.yaml` | `epochs_per_layer: 14`, patience 5, min_delta 0.0001 | Owner confirm canonical Protocol A schedule |
| `configs/mf/mnist_mlp_2x1000_cache_host.yaml` | `epochs_per_layer: 14`, patience 5, min_delta 0.0001 | Owner confirm canonical Protocol A schedule |
| `configs/mf/fashion_mnist_mlp_2x1000*.yaml` | `epochs_per_layer: 12`, patience 10, min_delta 0.0001 | Owner confirm canonical Protocol A schedule |
| `configs/ff/*.yaml` | `early_stopping` blocks; algorithm patience 20, min_delta 0.01 | Owner confirm algorithm-unit interpretation |
| `configs/cafo/cafodfa_mnist_cnn_3block.yaml` | `block_training_epochs: 5`, `num_epochs_per_block: 12`, predictor patience 5, min_delta 0.0005 | Owner confirm tuned predictor budget |
| `configs/cafo/mnist_cnn_3block.yaml` | `num_epochs_per_block: 227`, predictor patience 8, min_delta 0.0005 | Owner confirm tuned predictor budget |
| `configs/cafo/cafodfa_fashion_mnist_cnn_3block.yaml` | `block_training_epochs: 15`, `num_epochs_per_block: 85`, predictor patience 6, min_delta 0.001 | Owner confirm tuned predictor budget |
| `configs/cafo/cafodfa_cifar10_cnn_3block.yaml` | `block_training_epochs: 250`, `num_epochs_per_block: 147`, predictor patience 8, min_delta 0.001 | Owner confirm tuned predictor budget |
| `configs/cafo/cifar10_cnn_3block.yaml` | `num_epochs_per_block: 733`, predictor patience 8, min_delta 0.001 | Owner confirm tuned predictor budget |
| `configs/cafo/cifar100_cnn_3block.yaml` | `num_epochs_per_block: 1220`, predictor patience 10, min_delta 0.001 | Owner confirm tuned predictor budget |
| `configs/cafo/cafodfa_cifar100_cnn_3block.yaml` | `block_training_epochs: 200`, `num_epochs_per_block: 1271`, predictor patience 10, min_delta 0.001 | Owner confirm tuned predictor budget |
| `configs/diagnostics/*_equal_epochs.yaml` | max/patience 9, 11, or 13; min_delta 0.0 | Intended Protocol B matched-pass control |
| `configs/diagnostics/cifar10_mlp_3x2000_legacy_es.yaml` | max 15, patience 4, min_delta 0.0001 | Intended legacy stopping diagnostic only |
| `configs/reproduction/cifar10_mlp_3x2000_mf_published.yaml` | max 15, patience 4, min_delta 0.0001 | Published reproduction protocol; owner confirm separation |
| `configs/reproduction/cifar10_mlp_3x2000_mf_published_no_m0.yaml` | max 15, patience 4, min_delta 0.0001 | Published reproduction protocol; owner confirm separation |
| `configs/reproduction/cifar10_mlp_3x2000_bp_published.yaml` | max 100, patience 20, min_delta 0.0 | Published BP reproduction; matches shared rule |

## Owner TODO: replace after arXiv v2

Protected links and entries were deliberately not changed:

- `README.md:12` arXiv badge URL
- `README.md:19` Paper link
- `README.md:31` arXiv preprint link
- `README.md:372` arXiv URL inside the BibTeX entry

Replace these locations after arXiv v2 is published. The `[Paper]` link, arXiv badge, and BibTeX entries were otherwise left untouched.

## Owner TODO: archive and release metadata

- Zenodo: deposit the raw per-run JSON records and NVML traces, then replace the README placeholder with the DOI. Current README wording intentionally leaves the archive separate; exact DOI is `TODO: Zenodo DOI`.
- Create the release tag after owner review: `git tag -a v2.0-ppam2026 -m "PPAM 2026 canonical release"`.
- Preserve an explicit old-provenance tag, for example: `git tag -a v1-superseded-do-not-cite <old-commit> -m "Superseded pre-uniform-stopping release"`.
- GPU-hour budget for the A100-40GB reproduction is `TODO: owner`.
- Provide `pip freeze` from the Athena environment so measurement dependencies can be pinned exactly.

## Owner TODO: GitHub metadata

Run only after reviewing the release:

```bash
gh repo edit Przemyslaw11/BeyondBackpropagation --description "Hardware-validated energy, power, memory, and accuracy benchmarks for forward-only deep learning algorithms."
gh api --method PUT repos/Przemyslaw11/BeyondBackpropagation/topics --input - <<'JSON'
{"names":["deep-learning","forward-forward","mono-forward","energy-efficiency","benchmarking"]}
JSON
```

The commands intentionally omit `CO2e`, `optuna`, and `hyperparameter-optimization` because the README does not report CO2e and this checkout does not retain the tuning implementation.

## Open questions and numbers requiring owner confirmation

- The README says “almost 1000 runs” while also naming 1027 NVML traces. The retained `src/plotting/tidy.py` provenance logic does not establish whether these are the same figure population; owner should confirm the denominator and wording.
- The removed tracked teaser displayed old matched-baseline deltas, not either canonical table; no figure reference remains.
- Confirm the exact Athena versions for `codecarbon`, `psutil`, `wandb`, `scikit-learn`, `numpy`, and SciPy from `pip freeze`.
- Confirm the intentional diagnostic overrides listed above and the intended seed manifests for the full-study commands.
- Confirm the approximate GPU-hour budget.
- Confirm whether historical tuning artifacts should be restored to a documented archive; they were not recreated or deleted from the current tree because the tracked checkout contains no implementation to validate.
- Historical reproduction comments containing superseded reported values were deleted from `configs/reproduction/`; the owner should decide whether those reproduction configs belong in a separately documented historical archive.
- The local MNIST quickstart reached training setup but could not download the dataset: the HTTPS certificate chain was rejected and the fallback URL returned 404. A pre-downloaded dataset or corrected CA trust is required to verify training end to end.

The remaining `superseded` identifiers in `src/plotting/tidy.py` and `tests/test_plotting.py` are active provenance fields and tests that exclude known non-canonical source directories. They are not README claims or current result tables.

## Measurement-region note

`src/training/engine.py` starts CodeCarbon and the NVML monitor before data/model setup, then enters the NVML monitor context only around the training function. The timed `training_duration_sec` begins immediately before the training function and ends after it returns. Profiling runs before that timer; evaluation runs after it. CodeCarbon spans the broader run lifecycle and is not identical to the training timer. Measurement logic was not changed.

## Follow-up resolution: stopping rules

The implementation resolves only the top-level `early_stopping` mapping. `src/utils/early_stopping.py:23-37` copies defaults and updates that mapping, and rejects unknown keys. BP consumes the resulting policy at `src/baselines/bp.py:190-204`; BP-DS/MF-Joint at `src/baselines/bp_ds.py:251-255`; FF at `src/algorithms/ff.py:152-164`; MF at `src/algorithms/mf.py:426-440`; and CaFo predictors at `src/algorithms/cafo.py:596-608`. Therefore the legacy per-algorithm names are not training inputs.

| Override | Classification | Evidence and effect |
|---|---|---|
| BP `legacy_hyperparameters.training.early_stopping_*` in `configs/bp_baselines/*.yaml:30-41` | INERT | The resolver accepts only `early_stopping.*` (`src/utils/early_stopping.py:23-37`); BP reads that resolved policy (`src/baselines/bp.py:190-204`). The invariant test reads the legacy block at `tests/test_fairness_invariants.py:274-283`, so it was retained. |
| MF `legacy_hyperparameters.algorithm_params.epochs_per_layer` and `mf_early_stopping_*` in `configs/mf/*.yaml:31-38` | INERT | MF sets `epochs_per_layer` from `es_policy["max_epochs"]` at `src/algorithms/mf.py:426-436`; the loop uses that value at `src/algorithms/mf.py:474-477`. The legacy block is also preserved by the invariant test. |
| FF `early_stopping.metric: val_accuracy`, `configs/ff/fashion_mnist_mlp_4x2000.yaml:41-44` and the three analogous FF configs | LIVE | FF calls `resolve_early_stopping` at `src/algorithms/ff.py:152-164` and uses the metric/mode in its stopping comparisons at `src/algorithms/ff.py:473-495`. This is a canonical live deviation and remains an owner blocker. |
| FF `legacy_hyperparameters.training.early_stopping_*`, `configs/ff/*.yaml:48-56` | INERT | FF reads the shared mapping, not this nested block (`src/algorithms/ff.py:152-164`). |
| CaFo `legacy_hyperparameters.algorithm_params.num_epochs_per_block` and `predictor_early_stopping_*`, `configs/cafo/*.yaml:36-46` or `45-52` | INERT | Predictor caps and stopping config come from `resolve_early_stopping` at `src/algorithms/cafo.py:596-608` and are passed at `src/algorithms/cafo.py:678-681`. |
| CaFo-DFA `algorithm_params.block_training_epochs`, `configs/cafo/cafodfa_mnist_cnn_3block.yaml:32`, `cafodfa_fashion_mnist_cnn_3block.yaml:32`, `cafodfa_cifar100_cnn_3block.yaml:32`, `cafodfa_cifar10_cnn_3block.yaml:32` | LIVE | `src/algorithms/cafo.py:57` reads the key and `src/algorithms/cafo.py:109` loops over that cap; the DFA phase is called at `src/algorithms/cafo.py:569`. Values are respectively 5, 15, 200, and 250, so these are canonical live deviations from the shared cap of 100 and are blockers. |
| Diagnostics `early_stopping.max_epochs/patience/min_delta`, `configs/diagnostics/*_equal_epochs.yaml` | LIVE and intentional | The top-level mapping is consumed by every algorithm through the resolver; these files are Protocol B matched-pass controls. |
| Legacy-ES diagnostic `early_stopping.*`, `configs/diagnostics/cifar10_mlp_3x2000_legacy_es.yaml:61-67` | LIVE and intentional | It changes the shared policy for a stopping-policy sensitivity run only; it is not canonical. |
| Reproduction `early_stopping.*`, `configs/reproduction/*:54-63` | LIVE and intentional | These restore published reproduction protocols outside canonical experiment groups. |

Effective merged policies:

- `configs/mf/cifar10_mlp_3x2000.yaml` and `configs/mf/cifar100_mlp_3x2000.yaml`: `val_loss`, `min`, patience 20, cap 100, `min_delta` 0.0. They equal the shared rule; their nested legacy fields are inert.
- `configs/bp_baselines/cifar10_mlp_3x2000_bp.yaml`: the same shared rule; its nested legacy fields are inert.
- `configs/ff/mnist_mlp_4x2000.yaml`: `val_accuracy`, `max`, patience 20, cap 100, `min_delta` 0.0. The metric differs from the shared `val_loss` rule and is a blocker.
- `configs/cafo/cifar10_cnn_3block.yaml`: the shared `val_loss`, `min`, patience 20, cap 100 predictor rule. Its nested `num_epochs_per_block` and predictor fields are inert. CaFo-DFA configs separately have the live block-training caps listed above.

No inert stopping keys were deleted from canonical configs because `tests/test_fairness_invariants.py:274-283` explicitly reads and requires each `legacy_hyperparameters` block. Removing those fields would break a reproducibility/invariant contract; the production training code does not consume them. No LIVE key was edited.

## Tuning provenance

The deleted tuning implementation was removed in `31a2e823ad644a8088671093ac92dc2d350e2898` (`Prepare reproduction package`). The old MF objective reads `mf_epochs_per_layer_range` and writes `algorithm_params.epochs_per_layer` (`src/tuning/optuna_objective_mf.py` in the parent of that commit, lines 34-39), while the search driver reads the configured pruner and passes it to Optuna (parent `scripts/run_optuna_search.py`, lines 115-167). It did not implement the current base-config `tuning.max_epochs: 20` contract: MF tuning varied its legacy epoch key over 5-30, and the driver defaulted to a Median pruner unless the config said `None`; the current deleted config used 50 trials and `None` but did not use `max_epochs: 20`.

Current production code does not call `resolve_tuning_max_epochs`; only its definition reads `tuning.max_epochs` (`src/utils/early_stopping.py:51-57`). The `tuning:` block is therefore dead in the current checkout. Owner choice remains: restore the deleted implementation under a documented `tuning/` package with its exact protocol, or remove the dead block and describe the historical tuning procedure in the README. It was not restored.

## Figure provenance resolution

The removed tracked teaser showed four MLP rows with positive accuracy gains and mixed time, energy, and memory deltas. Those values were from the old matched-baseline presentation, not the canonical Protocol B or Protocol A tables. The file and its references were removed.

## Measurement checks

- `torch.cuda.reset_peak_memory_stats(device)` is called at `src/training/engine.py:587`, after profiling and immediately before the training timer’s monitored training region.
- `peak_torch_alloc_mib` is read at `src/training/engine.py:600-606` after the training function returns. It measures the same reset-to-training region covered by the timer (`src/training/engine.py:583-611`) and the NVML monitor context (`src/training/engine.py:588-600`), while profiling and evaluation are outside that region.
- `general.backend: slurm` is a cluster execution profile selecting workers and pinned memory; it does not submit a job. The README configuration table now states this explicitly.

## Security follow-up

The requested exact fallback scan was also run: `git log --all -p | grep -E "ghp_|AKIA|xox[bap]-|wandb|api[_-]?key" -i`. Matches were W&B code/config references and generic API-key vocabulary; no token-shaped `ghp_`, `AKIA`, or `xox...` secret was found. `gitleaks` and `trufflehog` were unavailable in the environment. Owner commands:

```bash
gitleaks detect --source . --log-opts="--all"
trufflehog git file://. --only-verified
```

## Twenty largest tracked blobs at final audit

```text
52      src/plotting/figures.py
36      src/algorithms/cafo.py
32      src/algorithms/ff.py
28      src/training/engine.py
28      src/algorithms/mf.py
24      scripts/analyze_ablation_ladder.py
24      README.md
20      src/utils/monitoring.py
20      src/plotting/tidy.py
20      src/baselines/bp.py
20      src/architectures/ff_mlp.py
16      tests/test_fairness_invariants.py
16      tests/test_ablation_analysis.py
16      src/baselines/bp_ds.py
12      tests/test_plotting.py
12      tests/test_backend_policy.py
12      tests/test_ablation_ladder.py
12      src/data_utils/datasets.py
12      src/architectures/cafo_cnn.py
12      AUDIT_FOLLOWUPS.md
```
