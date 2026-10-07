<p align="center">
  <h1 align="center">Energy-Efficient Deep Learning<br>Without Backpropagation</h1>
  <p align="center"><b>A Rigorous Hardware-Validated Benchmarking Study of Forward-Only Algorithms</b></p>
  <p align="center">
    Forward-Forward, Cascaded Forward and Mono-Forward measured on an NVIDIA A100<br>
    against tuned backpropagation, with energy, power and memory read from the device.
  </p>
</p>

<p align="center">
  <a href="#citation"><img alt="PPAM 2026 · LNCS" src="https://img.shields.io/badge/PPAM%202026-LNCS%20(Springer)-1f4e79.svg?style=flat-square"></a>
  <a href="https://arxiv.org/abs/2511.01061"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2511.01061-b31b1b.svg?style=flat-square"></a>
  <a href="./LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square"></a>
  <a href="#installation"><img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue.svg?style=flat-square"></a>
  <a href="https://github.com/Przemyslaw11/BeyondBackpropagation/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/Przemyslaw11/BeyondBackpropagation?style=flat-square"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2511.01061">Paper</a> ·
  <a href="#key-results">Key Results</a> ·
  <a href="#reproducing-the-experiments">Reproduce</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#citation">Citation</a>
</p>

---

> **Paper:** *Energy-Efficient Deep Learning Without Backpropagation: A Rigorous Hardware-Validated
> Benchmarking Study of Forward-Only Algorithms*, accepted at PPAM 2026 (Springer LNCS).
> The paper source and PDF are maintained separately from this code repository.
> An earlier preprint is [arXiv:2511.01061](https://arxiv.org/abs/2511.01061).<br>
> **Authors:** Przemysław Spyra, Witold Dzwinel · AGH University of Krakow, Faculty of Computer Science

**Contents**

- [Overview](#overview)
- [Key Results](#key-results)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Checkpoints](#checkpoints)
- [Scope and Limitations](#scope-and-limitations)
- [Citation](#citation)
- [License and Acknowledgements](#license-and-acknowledgements)

## Overview

Backpropagation (BP) keeps every activation in memory until a global backward sweep consumes it.
Forward-only algorithms replace that global gradient with local learning signals and are usually
promoted as more efficient on the basis of FLOP counts. This repository tests the claim on
hardware: across almost 1000 runs on MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100, every run is
instrumented through NVML at 5 Hz for power, utilisation, clock and memory, and each algorithm is
compared with a BP baseline tuned under the same budget on the same architecture.

| Algorithm | Backward pass | Learning signal | Studied on |
|---|---|---|---|
| BP | yes | global end-to-end loss | every architecture (baseline) |
| Forward-Forward (FF) | no | layer-local goodness | MLPs |
| Cascaded Forward (CaFo) | no | per-block local predictors | 3-block CNNs |
| Mono-Forward (MF) | no | layer-local cross-entropy through learned projections | MLPs |

**In short.** Removing the backward pass alone saves nothing: FF costs 4.1× to 6.3× BP's energy and
CaFo 3.0× to 7.1×. Mono-Forward is different. Given the same number of data passes as BP, it uses
12.5% to 19.9% less energy and up to 27.5% less memory on the largest MLPs tested, and on CIFAR-100
its accuracy is statistically equivalent to BP's. Trained to its own convergence, MF exceeds BP's
test accuracy on CIFAR-10 and CIFAR-100 by 3.90 and 5.04 percentage points, at 3.2× to 4.0× BP's
energy because it needs 3.5× to 4.5× as many passes. A pre-specified ablation identifies gradient
detachment, not MF's auxiliary losses, as the dominant factor tested.

Under matched passes, the accuracy costs shown below are −1.11 pp on CIFAR-10, −0.22 pp on MNIST,
and −0.49 pp on Fashion-MNIST.

## Key Results

All numbers below are produced by the released analysis; none is transcribed by hand. The raw run
records are archived separately (`TODO: Zenodo DOI`).
Every run stops under one rule (validation loss, patience 20, cap 100 epochs per training unit),
and the paper reports two protocols because no single one answers both questions a practitioner
asks:

- **Protocol A, own convergence:** each algorithm trains until it stops improving.
- **Protocol B, matched data passes:** MF receives at least as many data passes as BP actually used,
  so the stopping rule cannot confound the cost comparison.

Memory is the per-process allocator peak, `torch.cuda.max_memory_allocated`, never the
device-wide NVML reading, which includes a ≈870 MiB CUDA context no algorithm controls.

### Mono-Forward against backpropagation

MF has n = 20 seeds per configuration; Δ is MF relative to BP, so negative cost is cheaper.

| Configuration | Protocol | Δ accuracy (pp) | Δ energy | Δ memory | Δ power | Δ time |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 MLP 3×2000 | B, matched passes | −1.11 | **−19.9%** | **−27.5%** | **−17.6%** | −2.8% |
| CIFAR-100 MLP 3×2000 | B, matched passes | +0.02 (equivalent) | **−12.5%** | **−24.6%** | **−12.8%** | +0.4% |
| MNIST MLP 2×1000 | B, matched passes | −0.22 | **−2.2%** | +0.3% | **−5.8%** | +3.8% |
| Fashion-MNIST MLP 2×1000 | B, matched passes | −0.49 | **−4.3%** | +0.3% | **−3.6%** | −0.8% |
| CIFAR-10 MLP 3×2000 | A, own convergence | **+3.90** | 3.20× | −27.5% | −19.1% | 3.96× |
| CIFAR-100 MLP 3×2000 | A, own convergence | **+5.04** | 3.77× | −24.6% | −14.3% | 4.39× |

**Read the two protocols together.** Per data pass, MF is the cheaper algorithm, and the saving is
a power saving at near-equal wall-clock time. Trained to convergence it is more expensive end to
end, because it needs 3.5× to 4.5× as many passes. The pass multiplier, not the cost of a pass, is
what stands between MF and an end-to-end energy advantage.

### Where the accuracy comes from: a pre-specified ablation ladder

The ladder moves from BP to MF one structural change at a time:
BP → BP with deep supervision (BP-DS) → MF-Joint (MF's readout, global gradients) → MF (local
gradients). The analysis script, its tests, Holm correction and the ±0.25 pp equivalence margin
were committed before any ladder result was inspected
([`scripts/analyze_ablation_ladder.py`](scripts/analyze_ablation_ladder.py)).

- Deep supervision alone makes BP **worse** on CIFAR-10 and CIFAR-100, by 1.65 and 1.06 pp.
- MF beats that deep supervision control by 5.56 and 6.10 pp (Holm-corrected p < 10⁻²⁰).
- The detach step, with the optimiser family held fixed, is the largest single gain
  (+4.10 and +3.18 pp), and it also carries essentially the whole energy increase.
- On the two 2×1000 MLPs, MF and the deep supervision control are equivalent within the margin.

### Forward-Forward and Cascaded Forward

- **FF** trails BP by 0.01 to 1.10 pp on its native MLPs while needing 4.1× to 7.4× BP's wall-clock
  time and 4.1× to 6.3× its energy, at 0.91× to 1.06× its allocator memory.
- **CaFo** costs 3.0× to 7.1× BP's energy. It lands within 0.4 pp of BP on the two MNIST-scale CNNs
  but loses 18.8 and 16.9 pp on CIFAR-10 and CIFAR-100; its DFA variant narrows the gap to 7.1 and
  8.5 pp at a further 1.5× to 2.6× the energy.

The complete per-configuration FF and CaFo table belongs to the paper archive and is not included in this code-only checkout.

## Reproducing the Experiments

The code repository retains the tidy measurements and experiment-side analysis needed to inspect the released results. The LaTeX source, generated paper tables, and archived raw run traces are maintained separately.

| Artefact | Location |
|---|---|
| Tidy table of 19 944 measurements | Generated locally from run records |
| Builder of the tidy table from raw run records | [`scripts/build_tidy_table.py`](scripts/build_tidy_table.py) |
| Pre-specified ladder analysis | [`scripts/analyze_ablation_ladder.py`](scripts/analyze_ablation_ladder.py) |
| Bit-exactness check of the MF activation cache | [`scripts/check_mf_cache_equivalence.py`](scripts/check_mf_cache_equivalence.py) |
The per-run JSON records and the 1027 NVML power traces are archived separately and available from
the authors on request (`TODO: Zenodo DOI`).

Experiment configurations for each arm of the study:

| Arm | Configs |
|---|---|
| BP baselines | [`configs/bp_baselines/`](configs/bp_baselines/) |
| Mono-Forward | [`configs/mf/`](configs/mf/) |
| Ablation ladder controls | [`configs/bp_ds/`](configs/bp_ds/), [`configs/mf_joint/`](configs/mf_joint/) |
| Matched data-pass budgets | [`configs/diagnostics/`](configs/diagnostics/) |
| Forward-Forward, CaFo | [`configs/ff/`](configs/ff/), [`configs/cafo/`](configs/cafo/) |

### Reproducing the full study

Run the study in this order on an A100-40GB, using the seed lists in each config; the MF
comparisons use n = 20 seeds per configuration. Run outputs are written below `results/`, with
the main arms in `results/phase4`, ladder controls in `results/runs`, matched-pass diagnostics in
`results/equal_epochs`, and reproduction checks in `results/reproduction`.

```bash
# BP baselines
python scripts/run_local_array.py --config-dir configs/bp_baselines/
# MF, FF and CaFo
python scripts/run_local_array.py --config-dir configs/mf/
python scripts/run_local_array.py --config-dir configs/ff/
python scripts/run_local_array.py --config-dir configs/cafo/
# Ablation ladder controls
python scripts/run_local_array.py --config-dir configs/bp_ds/
python scripts/run_local_array.py --config-dir configs/mf_joint/
# Matched data-pass diagnostics
python scripts/run_local_array.py --config-dir configs/diagnostics/
# Build the tidy measurements and analyze the ladder
python scripts/build_tidy_table.py --results-dir results/phase4 --output results/tidy.csv
python scripts/analyze_ablation_ladder.py --results-dir results/runs --json results/ablation_ladder.json
```

The exact Athena GPU-hour budget is `TODO: owner`. The `configs/reproduction/` files are the
published MF reproduction checks; this checkout does not retain the historical tuning driver or
its Optuna environment, so the tuned baseline values are consumed as config data.

## Repository Structure

```text
.
|-- configs/
|   |-- base.yaml                      # shared defaults: device, data root, logging, monitoring, tuning
|   |-- bp_baselines/                  # tuned BP baselines matching the FF, MF and CaFo architectures
|   |-- bp_ds/, mf_joint/              # ablation ladder controls
|   |-- diagnostics/                   # matched data-pass (equal-epoch) budgets
|   |-- ff/, cafo/, mf/                # final experiment configs per algorithm
|   |-- reproduction/                  # reproductions of the published MF setup
|-- plots/                             # retained plotting code and source figure
|-- scripts/
|   |-- run_experiment.py              # single train-and-test entry point
|   |-- run_local_array.py             # local sequential batch runner
|   |-- build_tidy_table.py            # run records -> generated tidy table
|   |-- analyze_ablation_ladder.py     # pre-specified ladder statistics
|   |-- check_mf_cache_equivalence.py  # MF activation-cache bit-exactness check
|   `-- __init__.py                    # package marker for analysis imports
`-- src/
    |-- algorithms/                    # FF, CaFo and MF training and evaluation loops
    |-- architectures/                 # FF_MLP, MF_MLP and CaFo_CNN modules
    |-- baselines/                     # standard BP training baseline
    |-- data_utils/                    # torchvision datasets, splits, transforms
    |-- plotting/                      # tidy-table construction and plotting helpers
    |-- training/                      # experiment orchestration engine
    `-- utils/                         # config parsing, logging, NVML monitoring, profiling
  |-- tests/                             # unit and invariant tests
  `-- CHANGELOG.md                       # neutral methodology correction note
```

`data/`, `results/`, `checkpoints/` and `wandb/` are created at run time and are not version
controlled.

## Installation

Environment used for every run in the paper:

| Component | Version |
|---|---|
| Cluster | Athena, ACK Cyfronet AGH (Rocky Linux 9.5) |
| CPU | AMD EPYC 7742 |
| GPU | NVIDIA A100-SXM4-40GB |
| NVIDIA driver | 595.71.05 |
| CUDA toolkit | 12.4.0 |
| Python | 3.10.4 |
| PyTorch | 2.4.0+cu121 |
| Torchvision | 0.19.0 |
| pynvml | 12.0.0 |

**Linux with an NVIDIA GPU (virtualenv)**

```bash
git clone https://github.com/Przemyslaw11/BeyondBackpropagation.git
cd BeyondBackpropagation
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The same works with conda (`conda create -n beyond-bp python=3.10`). Energy and power measurement
needs NVML, so hardware numbers comparable with the paper require an NVIDIA GPU.

**macOS on Apple Silicon (functional runs only)**

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

With `general.backend: local` and `general.device: auto`, MPS is used when available and the CPU
otherwise. NVML is absent on macOS, so no energy or power is recorded.

**SLURM (Athena)**

```bash
module purge
module load Python/3.10.4 CUDA/12.4.0
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

<details>
<summary>Common installation issues</summary>

- `torch.cuda.is_available()` is `False` on a login node: test inside a GPU allocation.
- PyTorch installed a CPU wheel: reinstall the CUDA wheel that matches your cluster.
- `pynvml` reports missing drivers: run on an NVIDIA GPU node with NVML available.
- Slow installs on the cluster: download wheels elsewhere and run
  `pip install --no-index --find-links=./wheels -r requirements.txt`.
- No network on compute nodes for Weights & Biases: `export WANDB_MODE=offline` and sync later.

</details>

## Data

Datasets are downloaded by torchvision on first use (`data.download: true`) into `data.root`
(default `./data`).

| Dataset | Config name | Split |
|---|---|---|
| [MNIST](https://yann.lecun.com/exdb/mnist/) | `MNIST` | fixed 50k train / 10k validation, official 10k test |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | `FashionMNIST` | `val_split: 0.1` of the training set, official test |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | `CIFAR10` | `val_split: 0.1` of the training set, official test |
| [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) | `CIFAR100` | `val_split: 0.1` of the training set, official test |

CIFAR training uses random horizontal flips and 4-pixel-padded random crops, applied identically
to every algorithm and its baseline; MNIST and Fashion-MNIST are not augmented. For the MLP
experiments, CIFAR inputs are flattened, following the Mono-Forward protocol. To use
pre-downloaded data, set `data.root: /path/to/datasets` and `data.download: false`.

## Quickstart

Train and test one Mono-Forward model and its BP baseline:

```bash
python scripts/run_experiment.py --config configs/mf/cifar10_mlp_3x2000.yaml
python scripts/run_experiment.py --config configs/bp_baselines/cifar10_mlp_3x2000_bp.yaml
```

Other entry points:

```bash
# local execution (MPS or CPU)
python scripts/run_experiment.py --config configs/mf/mnist_mlp_2x1000.yaml --backend local

# Forward-Forward and CaFo
python scripts/run_experiment.py --config configs/ff/fashion_mnist_mlp_4x2000.yaml
python scripts/run_experiment.py --config configs/cafo/cafodfa_cifar10_cnn_3block.yaml

# a directory of configs, locally
python scripts/run_local_array.py --config-dir configs/mf/
```

Each run logs, among others, these final metrics:

| Metric | Meaning |
|---|---|
| `final/training_duration_sec` | wall-clock training time |
| `final/total_gpu_energy_wh` | trapezoidal integral of the 5 Hz NVML power trace |
| `final/peak_torch_alloc_mib` | per-process allocator peak (the memory figure used in the paper) |
| `final/peak_gpu_mem_used_mib` | device-wide NVML memory, including the CUDA context |
| `final/epochs_completed` | epochs actually trained under early stopping |

## Configuration

Configs are plain YAML merged with [`configs/base.yaml`](configs/base.yaml) by
`src/utils/config_parser.py`.

| Field | Meaning |
|---|---|
| `experiment_name` | Run name used for logs, W&B, results and checkpoints |
| `general.backend` | `"slurm"` (default) or `"local"`; overridable with `--backend` |
| `algorithm.name` | `BP`, `FF`, `CaFo` or `MF` |
| `data.name`, `data.root`, `data.val_split` | Dataset, directory and validation fraction |
| `data_loader.batch_size` | Training batch size (128 for every MLP in the paper) |
| `model.name` | `FF_MLP`, `MF_MLP` or `CaFo_CNN` |
| `model.params.hidden_dims` | MLP hidden widths |
| `model.params.block_channels` | CaFo CNN block channels |
| `optimizer.lr`, `optimizer.weight_decay` | BP optimiser settings |
| `algorithm_params.lr`, `algorithm_params.epochs_per_layer` | MF layer-local optimiser and per-stage budget |
| `algorithm_params.ff_learning_rate`, `algorithm_params.downstream_learning_rate` | FF goodness and classifier learning rates |
| `algorithm_params.predictor_lr`, `algorithm_params.num_epochs_per_block` | CaFo predictor learning rate and budget |
| `monitoring.energy_enabled` | NVML energy and power sampling |
| `profiling.enabled` | Forward-pass GFLOP profiling |

New datasets go in `src/data_utils/datasets.py` and `src/data_utils/preprocessing.py`; new models
in `src/architectures/` and `src/training/engine.py`.

## Checkpoints

Checkpoints are not distributed. A config writes them when `checkpointing.checkpoint_dir` is set,
for example `checkpoints/mf_cifar10_mlp_3x2000/`, which then holds one file per trained MF stage
(`mf_matrix_M0_complete.pth`, `mf_layer_1_complete.pth`, …). `checkpoints/` is not version
controlled.

## Scope and Limitations

- **Architectures.** MF is evaluated on MLPs only; the headline savings are measured on the
  3×2000 MLPs, and at 1.8 M weights the memory advantage has not yet appeared. No convolutional MF
  was run.
- **Accuracy context.** Flattening CIFAR for the MLP protocol caps accuracy far below the 85.6%
  our own 3-block CNN reaches under BP, so the comparisons are internally valid and say nothing
  about a competitive vision pipeline.
- **Hardware.** One accelerator and one driver. Board power at these low utilisations is dominated
  by static and clock-domain components of a large A100 die, so a smaller or better-saturated
  device would likely narrow the power gap; the memory saving and the pass multiplier should carry
  across.
- **Convergence.** MF's end-to-end cost is set by its 3.5× to 4.5× pass multiplier; per-stage
  schedules and stopping criteria are the natural next target.

## Citation

If you use this code or data, please cite the PPAM 2026 paper:

```bibtex
@inproceedings{spyra2026energy,
  title     = {Energy-Efficient Deep Learning Without Backpropagation: A Rigorous
               Hardware-Validated Benchmarking Study of Forward-Only Algorithms},
  author    = {Spyra, Przemys{\l}aw and Dzwinel, Witold},
  booktitle = {Parallel Processing and Applied Mathematics (PPAM 2026)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```

The earlier preprint:

```bibtex
@misc{spyra2025energyefficientdeeplearningbackpropagation,
  title         = {Energy-Efficient Deep Learning Without Backpropagation: A Rigorous Evaluation of Forward-Only Algorithms},
  author        = {Spyra, Przemys{\l}aw and Dzwinel, Witold},
  year          = {2025},
  eprint        = {2511.01061},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2511.01061}
}
```

## License and Acknowledgements

Released under the [MIT License](LICENSE).

This work was supported by the National Science Centre (NCN), Poland, under grant OPUS-29,
DEC-2025/57/B/ST6/04377, and by the Ministry of Science and Higher Education of Poland. We thank the
PLGrid Infrastructure and ACK Cyfronet AGH for computational resources on the Athena cluster under
grant PLG/2025/018341, and AGH University of Krakow for institutional support.

The algorithms studied are due to Hinton (Forward-Forward), Zhao et al. (Cascaded Forward) and
Gong, Li and Abdulla (Mono-Forward).
