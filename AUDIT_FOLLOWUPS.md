# Audit Follow-ups

This ledger records the release audit decisions and items that require owner confirmation. Canonical scientific values were not changed.

## Step 0 inventory

The baseline was taken on `feature/ppam-2026-camera-ready` before creating `release/canonical-v2`. The tracked tree contains 8 config groups, `src/`, `scripts/`, `plots/`, and `tests/`; there are no tracked notebooks or `slurm_logs/` files. The twenty largest tracked blobs were recorded in the audit session; the largest is `plots/teaser_mf_all_datasets.png`.

| Item not described or requiring clarification | What it is | References | Methodology status |
|---|---|---|---|
| `tests/` | Unit, plotting, backend, gradient, and fairness invariants | `pytest`, CI/user validation | Current methodology |
| `scripts/slurm_scripts/` | Present as an empty directory in the working tree; no tracked files | None found | Historical/empty |
| `scripts/tuning_utils/` | Present as an empty directory; no tracked files | No current references | Historical/empty |
| `configs/tuning/` | Present as an empty directory; no tracked files | No current references | Historical/empty |
| `src/tuning/` | Present as an empty directory; no tracked files | No current references | Historical/empty |
| `plots/` | One tracked teaser figure and plotting source is retained | README structure and plotting modules | Current checkout; protocol provenance needs owner confirmation |
| `configs/reproduction/` | Published MF reproduction checks | README reproduction section | Current methodology |
| `CHANGELOG.md` | Neutral note about the methodology correction | Repository metadata | Current canonical release |

Deleted historical paths include tuning configs, Optuna/slurm scripts, and tuning utilities. No deleted sensitive-name path (`*COPYRIGHT*`, `License-to-Publish*`, or `master_thesis/`) was found.

## Security scan

- `gitleaks` and `trufflehog` are not installed in the audit environment, so their requested commands could not run.
- Fallback `git log --all -p` keyword scan found no credential, token, password, or private-key material. Matches such as `key`, `secret`, and `password` were code/config vocabulary: harmless or false positives.
- Working-tree scans found no `/home/`, `/net/`, `/mnt/`, `/Users/`, PLGrid-account-like token, SLURM account, or email identity.
- No tracked notebooks or `slurm_logs/` existed to strip or inspect.
- No real secret or potentially sensitive credential was found; no blocker was raised.

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
- Confirm that `plots/teaser_mf_all_datasets.png` reflects the canonical protocol. It is retained because the plotting source and figure are tracked, but no raw data is present for regeneration.
- Confirm the exact Athena versions for `codecarbon`, `psutil`, `wandb`, `scikit-learn`, `numpy`, and SciPy from `pip freeze`.
- Confirm the intentional diagnostic overrides listed above and the intended seed manifests for the full-study commands.
- Confirm the approximate GPU-hour budget.
- Confirm whether historical tuning artifacts should be restored to a documented archive; they were not recreated or deleted from the current tree because the tracked checkout contains no implementation to validate.
- Historical reproduction comments still contain `61.13`, `268.45`, `62.34`, and `177.70` in `configs/reproduction/`. They are preserved because the release rule forbids changing reported scientific numbers; the owner should decide whether those reproduction configs belong in a separately documented historical archive.
- The local MNIST quickstart reached training setup but could not download the dataset: the HTTPS certificate chain was rejected and the fallback URL returned 404. A pre-downloaded dataset or corrected CA trust is required to verify training end to end.

The remaining `superseded` identifiers in `src/plotting/tidy.py` and `tests/test_plotting.py` are active provenance fields and tests that exclude known non-canonical source directories. They are not README claims or current result tables.

## Measurement-region note

`src/training/engine.py` starts CodeCarbon and the NVML monitor before data/model setup, then enters the NVML monitor context only around the training function. The timed `training_duration_sec` begins immediately before the training function and ends after it returns. Profiling runs before that timer; evaluation runs after it. CodeCarbon spans the broader run lifecycle and is not identical to the training timer. Measurement logic was not changed.
