# Beyond Backpropagation: Exploring Innovative Algorithms for Energy-Efficient Deep Neural Network Training

This repository contains the code for a Master's thesis investigating the performance and energy efficiency of alternative, backpropagation-free deep learning training algorithms compared to standard backpropagation.

## Project Goal

The primary objective is to rigorously compare the training performance (accuracy, convergence) and energy efficiency (energy consumption, time, FLOPs, memory) of three alternative algorithms:

1.  **Forward-Forward (FF)** \cite{hinton2022forward}
2.  **Cascaded Forward (CaFo)** \cite{zhao2023cafo}
3.  **Mono-Forward (MF)** \cite{gong2025mono}

These are compared against standard **Backpropagation (BP)** baselines using identical network architectures to isolate the effect of the training algorithm itself. Experiments are conducted on Fashion-MNIST, CIFAR-10, and CIFAR-100 datasets using a consistent, modern software environment.

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
├── requirements.txt          # Python package dependencies (targets latest compatible versions)
├── results/                  # (Gitignored) Parent dir for generated outputs
│   ├── logs/                 # Python application logs
│   └── optuna/               # Optuna study databases
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
    git clone https://github.com/Przemyslaw11/BeyondBackpropagation.git
    cd BeyondBackpropagation
    ```
2.  **Create and activate a Python virtual environment:** (Use a Python version compatible with `requirements.txt`, e.g., 3.10+)
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

Experiments are designed to run on the `plgrid-gpu-a100` partition of the Athena cluster using a consistent environment.

1.  **Log in to Athena:**
    ```bash
    ssh plgspyra@pro.cyfronet.pl
    ssh plgspyra@athena.cyfronet.pl
    ```
2.  **Navigate to your `$SCRATCH` directory and clone the repository:**
    ```bash
    cd $SCRATCH
    git clone https://github.com/Przemyslaw11/BeyondBackpropagation.git
    cd BeyondBackpropagation
    ```
3.  **Set up the Python environment (Initial Setup):**
    *   **Install Packages:** It's recommended to install packages from the **login node** due to potentially slow internet on compute nodes. If downloads are slow even on the login node, download wheels locally first (see Troubleshooting below).
        *   Load modules on the login node:
            ```bash
            module purge
            module load Python/3.10.4
            module load CUDA/12.4.0 # Use latest compatible CUDA
            module list
            ```
        *   Create and activate the virtual environment:
            ```bash
            python3 -m venv venv
            source venv/bin/activate
            ```
        *   Install dependencies (will install into `venv/`):
            ```bash
            # Ensure you are in $SCRATCH/BeyondBackpropagation
            pip install -r requirements.txt
            # If pip is slow, use the offline method (download wheels locally, transfer, then use pip install --no-index --find-links=...)
            ```
    *   **Key Runtime Environment Versions:**
        *   Python: `3.10.4` (via `module load Python/3.10.4`)
        *   CUDA Toolkit: `12.4.0` (via `module load CUDA/12.4.0`)
        *   PyTorch: `2.4.0`
        *   Torchvision: `0.19.0`
        *   Optuna: `4.2.1`
        *   Weights & Biases: `0.19.8`
        *   pynvml: `12.0.0`
        *(Confirm versions using `python -m pip list` inside the activated `venv`)*

## Running Experiments

All commands below assume you are in the project's root directory (`$SCRATCH/BeyondBackpropagation/`). **Remember to edit the Slurm scripts in `scripts/slurm_scripts/` to set your grant account correctly: `#SBATCH -A plgoncotherapy-gpu-a100`**.

### Running a Single Experiment

*   **On Athena (using Slurm):**
    1.  Verify/Edit `scripts/slurm_scripts/run_single_experiment.slurm`, ensuring the `-A plgoncotherapy-gpu-a100` line is correct and the loaded modules match the setup (Python 3.10.4, CUDA 12.4.0).
    2.  Submit the job using `sbatch`, passing the config file path relative to the project root as an argument:
        ```bash
        sbatch scripts/slurm_scripts/run_single_experiment.slurm configs/<algo_or_baseline>/<config_name>.yaml
        ```
        *Example (MF):*
        ```bash
        sbatch scripts/slurm_scripts/run_single_experiment.slurm configs/mf/cifar10_mlp_3x2000.yaml
        ```
    3.  Slurm output logs will be saved in `slurm_logs/`. Experiment results (metrics, logs) will be saved in `results/`.

### Running Hyperparameter Optimization (Optuna for BP Baselines)

*   **On Athena (using Slurm):**
    1.  Verify/Edit `scripts/slurm_scripts/run_optuna.slurm`, ensuring the `-A plgoncotherapy-gpu-a100` line is correct and modules match.
    2.  Submit the job using `sbatch`, passing the baseline config file path and optionally the number of trials:
        ```bash
        sbatch scripts/slurm_scripts/run_optuna.slurm configs/bp_baselines/<config_name>.yaml <num_trials>
        ```
        *Example:*
        ```bash
        sbatch scripts/slurm_scripts/run_optuna.slurm configs/bp_baselines/cifar10_cnn_3block_bp.yaml 100
        ```
    3.  The Optuna study database (`.db`) will be saved in `results/optuna/`. Slurm logs are in `slurm_logs/`.
    4.  **IMPORTANT:** After Optuna finishes, manually update the `optimizer` section (`lr`, `weight_decay`) in the corresponding baseline `.yaml` file with the best parameters found before running the final baseline comparison experiment.

### Running Multiple Experiments (Slurm Job Array - Example)

The `scripts/slurm_scripts/run_array.slurm` script provides an example.

1.  Modify `scripts/slurm_scripts/run_array.slurm`:
    *   Set the correct account: `-A plgoncotherapy-gpu-a100`.
    *   Update the `CONFIG_FILES` array with desired config paths.
    *   Adjust the `--array=1-N` range (where N is the number of configs).
    *   Ensure loaded modules match the setup.
2.  Submit the job array:
    ```bash
    sbatch scripts/slurm_scripts/run_array.slurm
    ```

## Monitoring and Logging

*   **Weights & Biases (WandB):** Experiments log to WandB if configured (`use_wandb: true`). Set up your account and API key (e.g., `export WANDB_API_KEY=your_key` before `sbatch` or inside script). Project: `BeyondBackpropagation`.
*   **Local Logs:** Python logs are saved to `results/<experiment_name>/`.
*   **Slurm Logs:** Standard output/error from Slurm jobs are saved to `slurm_logs/`. Check these first for submission/environment errors.
*   **Energy Monitoring:** GPU energy consumption is monitored using `pynvml` and logged.
*   **Profiling:** FLOPs and parameter counts can be estimated using `torchprof`.

## Citation

If you use this code or methodology, please cite the relevant papers for the algorithms:

*   **FF:** Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations. *arXiv preprint arXiv:2212.13345*.
*   **CaFo:** Zhao, Q., et al. (2023). Cascaded Forward-Forward Algorithm for Green AI. *arXiv preprint arXiv:2311.15170*. Code: `https://github.com/Graph-ZKY/CaFo`
*   **MF:** Gong, J., et al. (2025). Mono-Forward: Backpropagation-Free Algorithm for Efficient Neural Network Training Harnessing Local Errors. *arXiv preprint arXiv:2501.09238*.

## Troubleshooting

*   **Slow `pip install` on Athena:** Compute nodes (and sometimes login nodes) have slow external internet. Download wheels locally first using `python -m pip download -r requirements.txt -d ./wheels/`, transfer the `wheels` directory to Athena's `$SCRATCH`, then install offline on the login node using `pip install --no-index --find-links=./wheels -r requirements.txt` inside the activated venv (after loading modules).
*   **`CUDA Available: False`:** This is normal on the *login node*. Verify CUDA availability within an interactive GPU job (`srun ...`) or check the output of your actual experiment Slurm jobs.
*   **Module/Environment Conflicts:** Always `module purge` before loading your specific required modules (`Python/3.10.4`, `CUDA/12.4.0`). Ensure you activate the correct virtual environment (`source venv/bin/activate`).
