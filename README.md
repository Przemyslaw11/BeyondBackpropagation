# Beyond Backpropagation: Exploring Innovative Algorithms for Energy-Efficient Deep Neural Network Training

This repository contains the code for a Master's thesis investigating the performance and energy efficiency of alternative, backpropagation-free deep learning training algorithms compared to standard backpropagation.

## Project Goal

The primary objective is to rigorously compare the training performance (accuracy, convergence) and energy efficiency (energy consumption, time, FLOPs, memory) of three alternative algorithms:

1.  **Forward-Forward (FF)** \cite{hinton2022forward}
2.  **Cascaded Forward (CaFo)** \cite{zhao2023cafo}
3.  **Mono-Forward (MF)** \cite{gong2025mono}

These are compared against standard **Backpropagation (BP)** baselines using identical network architectures to isolate the effect of the training algorithm itself. Experiments are conducted on Fashion-MNIST, CIFAR-10, and CIFAR-100 datasets.

## Note on Environment and Reproducibility

For fair comparison, all experiments presented in this thesis (FF, CaFo, MF, BP) are conducted within a single, consistent, modern environment:

*   **Python:** 3.10.4
*   **PyTorch:** 2.4.0 (or latest stable compatible at time of setup)
*   **CUDA:** 12.4.0 (via Athena module)
*   **Hardware:** NVIDIA A100 GPU (via Athena `plgrid-gpu-a100` partition)

This ensures that observed differences in performance and efficiency can be primarily attributed to the algorithmic mechanisms rather than variations in software versions or underlying libraries. While this approach prioritizes fair comparison within this study, results may differ numerically from those reported in the original publications, which might have used different, often unspecified, execution environments.

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
├── requirements.txt          # Python package dependencies (updated for latest compatible versions)
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
├── slurm/                    # SLURM submission scripts for Athena cluster
│   ├── run_array.slurm       # Example for running multiple configs via job array
│   ├── run_optuna.slurm      # Submit an Optuna search job
│   └── run_single_experiment.slurm # Submit a single experiment job
├── slurm_logs/               # (Gitignored) SLURM stdout/stderr files
└── venv/                     # (Gitignored) Python virtual environment
```

## Setup

### Local Setup (for development/testing only)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Przemyslaw11/BeyondBackpropagation.git
    cd BeyondBackpropagation
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    # Use a Python version compatible with the target environment (e.g., 3.10)
    python3.10 -m venv venv_local
    source venv_local/bin/activate
    ```
3.  **Install dependencies (using updated requirements):**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download datasets:** Datasets will be automatically downloaded to `data/` upon the first run if not present. Note: GPU acceleration may not be available locally.

### Athena Cluster Setup (Cyfronet - Recommended for Experiments)

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
3.  **Set up the Python environment (within an interactive job is recommended for the first time):**
    *   Request an interactive job:
        ```bash
        srun -A plgoncotherapy-gpu-a100 -p plgrid-gpu-a100 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 --pty /bin/bash -l
        ```
    *   Load necessary modules (latest compatible versions):
        ```bash
        module purge
        module load Python/3.10.4
        module load CUDA/12.4.0   # Use latest compatible CUDA for PT 2.4+
        module list
        ```
    *   Create and activate the virtual environment:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   Install dependencies (using the updated `requirements.txt`):
        ```bash
        pip install -r requirements.txt
        ```
    *   Verify installation (check PyTorch version and CUDA detection):
        ```bash
        python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'PyTorch CUDA Version: {torch.version.cuda}'); import optuna; print(f'Optuna: {optuna.__version__}')"
        # Expected: PyTorch 2.4.0 (or similar), CUDA Available: True, CUDA Version: 12.1 (or 12.4)
        ```
    *   Exit the interactive job once setup is complete (`exit`).

## Running Experiments on Athena

**IMPORTANT:**

*   All `sbatch` commands **must be executed from the project root directory** (`$SCRATCH/BeyondBackpropagation/`).
*   **Before submitting any jobs:** Edit the Slurm script files located in the `slurm/` directory (`run_single_experiment.slurm`, `run_optuna.slurm`, `run_array.slurm`). Find the line starting with `#SBATCH -A` and **replace** the placeholder `<your_grant_name>-gpu-a100` with your specific allocation name: `plgoncotherapy-gpu-a100`.

### Running a Single Experiment

1.  Modify `slurm/run_single_experiment.slurm` (ensure correct `-A plgoncotherapy-gpu-a100` line and loaded modules: `Python/3.10.4`, `CUDA/12.4.0`).
2.  Submit the job using `sbatch`, passing the config file path relative to the project root as an argument:
    ```bash
    # Example for FF
    sbatch slurm/run_single_experiment.slurm configs/ff/fashion_mnist_mlp_4x2000.yaml

    # Example for BP Baseline (run AFTER Optuna tuning and updating the config)
    sbatch slurm/run_single_experiment.slurm configs/bp_baselines/fashion_mnist_mlp_4x2000_bp.yaml
    ```
3.  Slurm output logs will be saved in `slurm_logs/`. Experiment results will be saved in `results/`.

### Running Hyperparameter Optimization (Optuna for BP Baselines)

1.  Modify `slurm/run_optuna.slurm` (ensure correct `-A plgoncotherapy-gpu-a100` line and loaded modules: `Python/3.10.4`, `CUDA/12.4.0`). Adjust time limit if needed.
2.  Submit the job using `sbatch`, passing the baseline config file path and optionally the number of trials:
    ```bash
    # Example with 100 trials
    sbatch slurm/run_optuna.slurm configs/bp_baselines/cifar10_cnn_3block_bp.yaml 100
    ```
3.  The Optuna study database (`.db`) and best parameter summary (`.yaml`) will be saved in `results/optuna/`. Slurm logs are in `slurm_logs/`.
4.  **CRITICAL:** After the job finishes, check the output/summary for the best `lr` and `weight_decay`. **Manually update the `optimizer` section in the corresponding BP baseline YAML config file** before running the final baseline experiment for comparison.

### Running Multiple Experiments (Slurm Job Array - Example)

1.  Modify `slurm/run_array.slurm`:
    *   Ensure correct `-A plgoncotherapy-gpu-a100` line and loaded modules (`Python/3.10.4`, `CUDA/12.4.0`).
    *   Define the `CONFIG_FILES` bash array with the paths to your desired configuration files.
    *   Adjust the `#SBATCH --array=1-N` range (where `N` is the number of configs).
2.  Submit the job array:
    ```bash
    sbatch slurm/run_array.slurm
    ```

## Monitoring and Logging

*   **Weights & Biases (WandB):** Experiments automatically log metrics, hyperparameters, and system stats (GPU usage, etc.) to WandB if configured (`use_wandb: true` in config). Set up your WandB account and ensure authentication (e.g., `wandb login` in an interactive job or set `WANDB_API_KEY` environment variable). Project name defaults to `BeyondBackpropagation`.
*   **Python Logs:** Detailed logs saved to `results/<experiment_name>/<experiment_name>_run.log`.
*   **Slurm Logs:** Standard output/error from Slurm jobs saved to `slurm_logs/`. Check these first for submission/environment errors.
*   **Energy/Resource Monitoring:** GPU energy, power, and memory usage are monitored using `pynvml` and logged via Python logger and WandB.
*   **Profiling:** FLOPs estimates using `torchprof` are logged.

## Citation
*   **FF:** Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations. *arXiv preprint arXiv:2212.13345*.
*   **CaFo:** Zhao, G., et al. (2023). The Cascaded Forward Algorithm for Neural Network Training. *arXiv preprint arXiv:2303.09728*.
*   **MF:** Gong, J., Li, B., & Abdulla, W. (2025). Mono-Forward: Backpropagation-Free Algorithm for Efficient Neural Network Training Harnessing Local Errors. *arXiv preprint arXiv:2501.09238*.
