# Beyond Backpropagation: Exploring Innovative Algorithms for Energy-Efficient Deep Neural Network Training

This repository contains the code for a Master's thesis investigating the performance and energy efficiency of alternative, backpropagation-free deep learning training algorithms compared to standard backpropagation.

## Project Goal

The primary objective is to rigorously compare the training performance (accuracy, convergence) and energy efficiency (energy consumption, time, FLOPs, memory) of three alternative algorithms:

1.  **Forward-Forward (FF)** \cite{hinton2022forward}
2.  **Cascaded Forward (CaFo)** \cite{zhao2023cafo}
3.  **Mono-Forward (MF)** \cite{gong2025mono}

These are compared against standard **Backpropagation (BP)** baselines using identical network architectures to isolate the effect of the training algorithm itself. Experiments are conducted on Fashion-MNIST, CIFAR-10, and CIFAR-100 datasets.

## Repository Structure

```
.
├── .gitignore
├── LICENSE
├── README.md                 # This file
├── configs/                  # Experiment configuration files (YAML)
│   ├── base.yaml             # Base configuration defaults
│   ├── bp_baselines/         # Configs for BP baselines (tuned)
│   ├── cafo/                 # Configs for CaFo experiments
│   ├── ff/                   # Configs for FF experiments
│   └── mf/                   # Configs for MF experiments
├── data/                     # (Gitignored) Datasets downloaded here
├── notebooks/                # Jupyter notebooks for analysis, visualization
├── requirements.txt          # Python package dependencies
├── results/                  # (Gitignored) Parent dir for generated outputs
│   ├── logs/                 # Python application logs
│   └── optuna/               # Optuna study databases
├── scripts/                  # Main Python execution scripts
│   ├── run_experiment.py     # Run a single experiment from a config file
│   └── run_optuna_search.py  # Run Optuna hyperparameter search for BP baselines
├── src/                      # Source code
│   ├── algorithms/           # Implementations of FF, CaFo, MF
│   ├── architectures/        # PyTorch nn.Module definitions
│   ├── baselines/            # BP baseline training logic
│   ├── data_utils/           # Data loading and preprocessing
│   ├── training/             # Core training/evaluation loop engine
│   ├── tuning/               # Optuna objective function
│   └── utils/                # Helper utilities (config, logging, metrics, monitoring, profiling)
├── scripts/                  # Main Python execution scripts
│   ├── run_experiment.py     # Run a single experiment from a config file
│   ├── run_optuna_search.py  # Run Optuna hyperparameter search for BP baselines
│   └── slurm_scripts/        # SLURM submission scripts for Athena cluster
│       ├── run_array.slurm       # Example for running multiple configs via job array
│       ├── run_optuna.slurm      # Submit an Optuna search job
│       └── run_single_experiment.slurm # Submit a single experiment job
├── slurm_logs/               # (Gitignored) SLURM stdout/stderr files
└── venv/                     # (Gitignored) Python virtual environment
```

## Setup

### Local Setup (for development/testing)

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd BeyondBackpropagationThesis
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download datasets:** Datasets will be automatically downloaded to the `data/` directory upon the first run if not present.

### Athena Cluster Setup (Cyfronet)

Experiments are designed to run on the `plgrid-gpu-a100` partition of the Athena cluster.

1.  **Log in to Athena:**
    ```bash
    ssh <your_login>@pro.cyfronet.pl
    ssh <your_login>@athena.cyfronet.pl
    ```
2.  **Navigate to your `$SCRATCH` directory and clone the repository:**
    ```bash
    cd $SCRATCH
    git clone <repository_url>
    cd BeyondBackpropagationThesis
    ```
3.  **Set up the Python environment (within an interactive job is recommended for the first time):**
    *   Request an interactive job (adjust resources if needed):
        ```bash
        srun -A <grantname>-gpu-a100 -p plgrid-gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 --pty /bin/bash -l
        ```
    *   Load necessary modules:
        ```bash
        module load plgrid/tools/python/3.10
        module load plgrid/tools/cuda/11.8
        # Add any other required modules
        ```
    *   Create and activate the virtual environment:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    *   Exit the interactive job once setup is complete.

## Running Experiments

All commands below assume you are in the project's root directory (`BeyondBackpropagationThesis/`).

### Running a Single Experiment

*   **Locally (if GPU available):**
    ```bash
    source venv/bin/activate
    python scripts/run_experiment.py --config-path configs/<algo_or_baseline>/<config_name>.yaml
    ```
*   **On Athena (using Slurm):**
    1.  Modify `scripts/slurm_scripts/run_single_experiment.slurm` if necessary (e.g., adjust resources, grant name).
    2.  Submit the job using `sbatch`, passing the config file path relative to the project root as an argument:
        ```bash
        sbatch scripts/slurm_scripts/run_single_experiment.slurm configs/<algo_or_baseline>/<config_name>.yaml
        ```
        *Example:*
        ```bash
        sbatch scripts/slurm_scripts/run_single_experiment.slurm configs/ff/fashion_mnist_mlp_4x2000.yaml
        ```
    3.  Slurm output logs will be saved in `slurm_logs/`. Experiment results (metrics, logs) will be saved in `results/`.

### Running Hyperparameter Optimization (Optuna for BP Baselines)

*   **Locally (if GPU available):**
    ```bash
    source venv/bin/activate
    python scripts/run_optuna_search.py --config-path configs/bp_baselines/<config_name>.yaml --n-trials <num_trials>
    ```
*   **On Athena (using Slurm):**
    1.  Modify `scripts/slurm_scripts/run_optuna.slurm` if necessary (e.g., adjust resources, grant name, number of trials).
    2.  Submit the job using `sbatch`, passing the config file path and number of trials:
        ```bash
        sbatch scripts/slurm_scripts/run_optuna.slurm configs/bp_baselines/<config_name>.yaml <num_trials>
        ```
        *Example:*
        ```bash
        sbatch scripts/slurm_scripts/run_optuna.slurm configs/bp_baselines/cifar10_cnn_3block_bp.yaml 100
        ```
    3.  The Optuna study database (`.db` file) will be saved in `results/optuna/`. Slurm logs are in `slurm_logs/`.

### Running Multiple Experiments (Slurm Job Array - Example)

The `scripts/slurm_scripts/run_array.slurm` script provides an example of how to run multiple configurations using a Slurm job array. You will need to adapt it to list the specific config files you want to run.

1.  Modify `scripts/slurm_scripts/run_array.slurm` to define the `CONFIG_FILES` array with the paths to your desired configuration files. Adjust resources and grant name as needed.
2.  Submit the job array:
    ```bash
    sbatch scripts/slurm_scripts/run_array.slurm
    ```

## Monitoring and Logging

*   **Weights & Biases (WandB):** Experiments automatically log metrics, hyperparameters, and system stats (GPU usage, etc.) to WandB if configured. Set up your WandB account and log in (`wandb login`) or configure environment variables (`WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`).
*   **Local Logs:** Python logs are saved to `results/logs/`.
*   **Slurm Logs:** Standard output and error from Slurm jobs are saved to `slurm_logs/`.
*   **Energy Monitoring:** GPU energy consumption is monitored using `pynvml` and logged.
*   **Profiling:** FLOPs and parameter counts can be estimated using `torchprof` (integrated into the training engine).

## Citation

If you use this code or methodology, please cite the relevant papers for the algorithms:

*   **FF:** Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations. *arXiv preprint arXiv:2212.13345*.
*   **CaFo:** Zhao, Q., et al. (2023). Cascaded Forward-Forward Algorithm for Green AI. *arXiv preprint arXiv:2311.15170*.
*   **MF:** Gong, P., et al. (2025). Mono-Forward: An Energy-Efficient Training Algorithm Free from Backpropagation. *To be published*. (Adjust citation details as needed).
