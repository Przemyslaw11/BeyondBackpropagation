# Beyond Backpropagation: Exploring Innovative Algorithms for Energy-Efficient Deep Neural Network Training
 
 This repository contains the code for a Master's thesis investigating the performance and energy efficiency of alternative, backpropagation-free deep learning training algorithms compared to standard backpropagation.
 
 ## Project Goal
 
 The primary objective is to rigorously compare the training performance (accuracy, convergence) and energy efficiency (energy consumption, time, FLOPs, memory) of three alternative algorithms:
 
 1.  **Forward-Forward (FF)** \cite{hinton2022forward}
 2.  **Cascaded Forward (CaFo)** \cite{zhao2023cafo}
 3.  **Mono-Forward (MF)** \cite{gong2025mono}
 
 These are compared against standard **Backpropagation (BP)** baselines using identical network architectures to isolate the effect of the training algorithm itself. Experiments are conducted on MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 datasets using a consistent, modern software environment.
 
 ## Repository Structure
```
 .
├── LICENSE
├── README.md
├── configs
│   ├── base.yaml
│   ├── bp_baselines
│   │   ├── cifar100_cnn_3block_bp.yaml
│   │   ├── cifar100_mlp_3x2000_bp.yaml
│   │   ├── cifar10_cnn_3block_bp.yaml
│   │   ├── cifar10_mlp_3x2000_bp.yaml
│   │   ├── fashion_mnist_cnn_3block_bp.yaml
│   │   ├── fashion_mnist_mlp_2x1000_bp.yaml
│   │   ├── fashion_mnist_mlp_4x2000_bp.yaml
│   │   ├── mnist_cnn_3block_bp.yaml
│   │   ├── mnist_mlp_2x1000_bp.yaml
│   │   ├── mnist_mlp_3x1000_bp.yaml
│   │   └── mnist_mlp_4x2000_bp.yaml
│   ├── cafo
│   │   ├── cafodfa_cifar100_cnn_3block.yaml
│   │   ├── cafodfa_cifar10_cnn_3block.yaml
│   │   ├── cafodfa_fashion_mnist_cnn_3block.yaml
│   │   ├── cafodfa_mnist_cnn_3block.yaml
│   │   ├── cifar100_cnn_3block.yaml
│   │   ├── cifar10_cnn_3block.yaml
│   │   ├── fashion_mnist_cnn_3block.yaml
│   │   └── mnist_cnn_3block.yaml
│   ├── ff
│   │   ├── fashion_mnist_mlp_4x2000.yaml
│   │   ├── mnist_mlp_3x1000_ADAMW.yaml
│   │   ├── mnist_mlp_3x1000_SGD.yaml
│   │   └── mnist_mlp_4x2000.yaml
│   ├── mf
│   │   ├── cifar100_mlp_3x2000.yaml
│   │   ├── cifar10_mlp_3x2000.yaml
│   │   ├── fashion_mnist_mlp_2x1000.yaml
│   │   └── mnist_mlp_2x1000.yaml
│   ├── test
│   │   ├── test_bp_fmnist_mlp_2x1000.yaml
│   │   ├── test_cafo_fmnist_cnn_3block.yaml
│   │   ├── test_ff_fmnist_mlp_4x2000.yaml
│   │   └── test_mf_fmnist_mlp_2x1000.yaml
│   └── tuning
│       ├── cafo_cafodfa_cifar100_cnn_3block_tune.yaml
│       ├── cafo_cafodfa_cifar10_cnn_3block_tune.yaml
│       ├── cafo_cafodfa_fashion_mnist_cnn_3block_tune.yaml
│       ├── cafo_cafodfa_mnist_cnn_3block_tune.yaml
│       ├── cafo_cifar100_cnn_3block_tune.yaml
│       ├── cafo_cifar10_cnn_3block_tune.yaml
│       ├── cafo_fashion_mnist_cnn_3block_tune.yaml
│       ├── cafo_mnist_cnn_3block_tune.yaml
│       ├── ff_fashion_mnist_mlp_4x2000_tune.yaml
│       ├── ff_mnist_mlp_3x1000_ADAMW_tune.yaml
│       ├── ff_mnist_mlp_3x1000_SGD_tune.yaml
│       ├── ff_mnist_mlp_4x2000_tune.yaml
│       ├── mf_cifar100_mlp_3x2000_mf_tune.yaml
│       ├── mf_cifar10_mlp_3x2000_mf_tune.yaml
│       ├── mf_fashion_mnist_mlp_2x1000_mf_tune.yaml
│       └── mf_mnist_mlp_2x1000_mf_tune.yaml
├── data
├── requirements.txt
├── results
├── scripts
│   ├── run_alt_optuna_search.py
│   ├── run_experiment.py
│   ├── run_optuna_search.py
│   ├── slurm_scripts
│   │   ├── run_alt_optuna.slurm
│   │   ├── run_array.slurm
│   │   ├── run_optuna.slurm
│   │   ├── run_single_experiment.slurm
│   │   └── run_test_experiment.slurm
│   └── tuning_utils
│       ├── update_bp_configs.py
│       ├── update_cafo_configs.py
│       ├── update_ff_configs.py
│       └── update_mf_configs.py
├── slurm_logs
├── src
│   ├── __init__.py
│   ├── algorithms
│   │   ├── __init__.py
│   │   ├── cafo.py
│   │   ├── ff.py
│   │   └── mf.py
│   ├── architectures
│   │   ├── __init__.py
│   │   ├── cafo_cnn.py
│   │   ├── ff_mlp.py
│   │   └── mf_mlp.py
│   ├── baselines
│   │   ├── __init__.py
│   │   └── bp.py
│   ├── data_utils
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   └── preprocessing.py
│   ├── training
│   │   ├── __init__.py
│   │   └── engine.py
│   ├── tuning
│   │   ├── __init__.py
│   │   ├── alt_optuna_objective.py
│   │   ├── optuna_objective.py
│   │   ├── optuna_objective_cafo.py
│   │   ├── optuna_objective_ff.py
│   │   └── optuna_objective_mf.py
│   └── utils
│       ├── __init__.py
│       ├── codecarbon_utils.py
│       ├── config_parser.py
│       ├── helpers.py
│       ├── logging_utils.py
│       ├── metrics.py
│       ├── monitoring.py
│       └── profiling.py
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
    ssh plgUSERNAME@pro.cyfronet.pl # Replace plgUSERNAME
    ssh plgUSERNAME@athena.cyfronet.pl # Replace plgUSERNAME
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
            module load CUDA/12.4.0 # Ensure compatibility with PyTorch build (+cu121)
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
        *   CUDA Toolkit (Module): `12.4.0` (via `module load CUDA/12.4.0`)
        *   PyTorch: `2.4.0 (+cu121)` (Check with `pip show torch`)
        *   Torchvision: `0.19.0`
        *   Optuna: `4.2.1`
        *   Weights & Biases: `0.19.8`
        *   pynvml: `12.0.0`
        *   CodeCarbon: `3.0.1`

## Running Experiments

All commands below assume you are in the project's root directory (`$SCRATCH/<your_login>/BeyondBackpropagation/`). **Remember to edit the Slurm scripts in `scripts/slurm_scripts/` to set your grant account correctly: `#SBATCH -A <your_grant_name>-gpu-a100`**.

### Running a Single Experiment

*   **On Athena (using Slurm):**
    1.  Verify/Edit `scripts/slurm_scripts/run_single_experiment.slurm`, ensuring the `-A <your_grant_name>-gpu-a100` line is correct and the loaded modules match the setup (Python 3.10.4, CUDA 12.4.0).
    2.  **Offline Wandb:** The Slurm scripts are configured to run Weights & Biases in offline mode (`export WANDB_MODE=offline`). Logs will be stored locally in the `wandb/` directory within the project root.
    3.  Submit the job using `sbatch`, passing the *final* config file path (e.g., from `configs/ff/`, `configs/cafo/`, `configs/mf/`, `configs/bp_baselines/`) relative to the project root.
        ```bash
        # Example: Running final FF MNIST SGD experiment
        sbatch --job-name="FF_MNIST_SGD_final" scripts/slurm_scripts/run_single_experiment.slurm configs/ff/mnist_mlp_3x1000_SGD_ref.yaml
        # Example: Running final CaFo-DFA CIFAR-100 experiment
        sbatch --job-name="CaFoDFA_C100_final" scripts/slurm_scripts/run_single_experiment.slurm configs/cafo/cafodfa_cifar100_cnn_3block.yaml
        # Example: Running a BP baseline
        sbatch --job-name="BP_CNN_C10_final" scripts/slurm_scripts/run_single_experiment.slurm configs/bp_baselines/cifar10_cnn_3block_bp.yaml
        ```
    4.  Slurm output logs go to `slurm_logs/`. Experiment results go to `results/`. Wandb offline logs are in `wandb/`. CodeCarbon offline logs are in `results/carbon/`.
    5.  **Syncing Offline Wandb:** After completion, sync using `wandb sync --sync-all` from the login node.

### Running Hyperparameter Optimization (Optuna)

*   **On Athena (using Slurm):**
    1.  Verify/Edit `scripts/slurm_scripts/run_optuna.slurm`, ensuring the `-A <your_grant_name>-gpu-a100` line is correct and modules match. Wandb is typically disabled for Optuna trials (`use_wandb: false` in the `tuning:` section of the config). CodeCarbon is also disabled by default in base Optuna configs.
    2.  Submit the job using `sbatch`, passing the **tuning configuration file path** (from `configs/tuning/` for FF/CaFo/MF, or `configs/bp_baselines/` for BP) and optionally the number of trials.
        ```bash
        # Example: Tuning CaFo-DFA for CIFAR-100 CNN with 50 trials
        sbatch --job-name="Optuna_CaFoDFA_C100" scripts/slurm_scripts/run_optuna.slurm configs/tuning/cafo_cafodfa_cifar100_cnn_3block_tune.yaml 50

        # Example: Tuning FF (AdamW) for MNIST 4x2000 MLP using config's trial count
        sbatch --job-name="Optuna_FF_MNIST_4x2000" scripts/slurm_scripts/run_optuna.slurm configs/tuning/ff_mnist_mlp_4x2000_tune.yaml

        # Example: Tuning BP for the MF CIFAR-10 MLP baseline using config's trial count
        sbatch --job-name="Optuna_BP_MF_C10" scripts/slurm_scripts/run_optuna.slurm configs/bp_baselines/cifar10_mlp_3x2000_bp.yaml
        ```
    3.  The Optuna study database (`.db`) will be saved in `results/optuna/`. Slurm logs are in `slurm_logs/`. The script outputs best parameters to the Slurm `.out` file and saves them to `_best_params.yaml`.
    4.  **IMPORTANT:** The `run_optuna.slurm` script attempts to **automatically update** the corresponding *final* `.yaml` config file (e.g., `configs/ff/mnist_mlp_4x2000.yaml` if tuning was run with `configs/tuning/ff_mnist_mlp_4x2000_tune.yaml`) using the appropriate utility script from `scripts/tuning_utils/` after the search completes.
        *   **Check the Slurm `.out` log** to confirm if the automatic update succeeded or failed.
        *   If it failed, or for manual updates, use the output YAML snippet from the log or the `_best_params.yaml` file to update the `optimizer` (for BP) or `algorithm_params` (for FF/CaFo/MF) section in the *final* algorithm config file (e.g., in `configs/ff/`, `configs/cafo/`, `configs/mf/`).
        *   **A backup** of the original config file (`.bak_<timestamp>`) is created by the update script before automatic update.

### Running Multiple Experiments (Slurm Job Array - Example)

The `scripts/slurm_scripts/run_array.slurm` script provides an example for submitting multiple *final* experiments defined in an array within the script.

1.  Modify `scripts/slurm_scripts/run_array.slurm`:
    *   Set the correct account: `-A <your_grant_name>-gpu-a100`.
    *   Update the `CONFIG_FILES` array with desired *final* config paths (from `configs/ff/`, `configs/cafo/`, `configs/mf/`, `configs/bp_baselines/`).
    *   Adjust the `--array=1-N` range (where N is the *exact* number of configs listed).
    *   Ensure loaded modules match the setup. (Wandb and CodeCarbon will run offline per task).
2.  Submit the job array:
    ```bash
    sbatch scripts/slurm_scripts/run_array.slurm
    ```
3.  After completion, sync all offline runs using `wandb sync --sync-all`.

## Monitoring and Logging

*   **Weights & Biases (WandB):** Experiments log to WandB. When running on Athena compute nodes, logs are saved offline to the `wandb/` directory. Sync them later using `wandb sync --sync-all` from the login node. Project: `BeyondBackpropagation`, Entity: `przspyra11` (can be changed in `configs/base.yaml`).
*   **Local Logs:** Python logs are saved to `results/<experiment_name>/<experiment_name>_run.log`. Optuna logs are saved to `results/optuna/<study_name>.log`.
*   **Slurm Logs:** Standard output/error from Slurm jobs are saved to `slurm_logs/`. Check these first for submission/environment errors.
*   **Energy Monitoring (NVML):** GPU energy consumption is monitored using `pynvml` during training and logged to WandB (`total_gpu_energy_joules`, `total_gpu_energy_wh`) in the final summary. Requires NVML to be functional.
*   **Memory Monitoring (NVML):** Peak GPU memory usage during the training loop is sampled using `pynvml` and logged to WandB (`final/peak_gpu_mem_used_mib`). Requires NVML.
*   **Carbon Tracking (CodeCarbon):**
    *   Tracks estimated carbon footprint (CO2 emissions equivalent).
    *   Enabled/disabled via the `carbon_tracker:` section in `configs/base.yaml`.
    *   Runs in **offline mode** on Athena, saving detailed results to a CSV file in `results/carbon/<experiment_name>_carbon.csv`.
    *   Uses the `country_iso_code` specified in the config (e.g., "POL" for Poland) to estimate grid intensity.
    *   The final estimated emissions (`final/codecarbon_emissions_gCO2e`) are logged to the console output and WandB summary at the end of the run.
*   **Profiling (torch.profiler):** FLOPs and parameter counts are estimated using `torch.profiler` (for FLOPs) and standard PyTorch parameter counting at the start of the run and logged to WandB.

## Citation

If you use this code or methodology, please cite the relevant papers for the algorithms:

*   **FF:** Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations. *arXiv preprint arXiv:2212.13345*.
*   **CaFo:** Zhao, Q., et al. (2023). Cascaded Forward-Forward Algorithm for Green AI. *arXiv preprint arXiv:2311.15170*. Code: `https://github.com/Graph-ZKY/CaFo`
*   **MF:** Gong, J., et al. (2025). Mono-Forward: Backpropagation-Free Algorithm for Efficient Neural Network Training Harnessing Local Errors. *arXiv preprint arXiv:2501.09238*.

## Troubleshooting

*   **Slow `pip install` on Athena:** Compute nodes (and sometimes login nodes) have slow external internet. Download wheels locally first using `pip download -r requirements.txt -d ./wheels/`, transfer the `wheels` directory to Athena's `$SCRATCH`, then install offline on the login node using `pip install --no-index --find-links=./wheels -r requirements.txt` inside the activated venv (after loading modules).
*   **`CUDA available: False`:** This is normal on the *login node*. Verify CUDA availability within an interactive GPU job (`srun --gres=gpu:1 --pty /bin/bash -l`) or check the output of your actual experiment Slurm jobs.
*   **Module/Environment Conflicts:** Always `module purge` before loading your specific required modules (`Python/3.10.4`, `CUDA/12.4.0`). Ensure you activate the correct virtual environment (`source venv/bin/activate`). Check Slurm log files (`slurm_logs/`) for errors related to module loading or environment activation.
*   **Wandb Errors:** If you see network errors related to Wandb in `.err` files, ensure your Slurm script sets `export WANDB_MODE=offline`. Remember to sync runs later using `wandb sync --sync-all`. If syncing fails, check your `WANDB_API_KEY` and internet connection.
*   **NVML Errors:** Errors like `NVML_ERROR_LIBRARY_NOT_FOUND` or `NVML_ERROR_DRIVER_NOT_LOADED` usually indicate issues with the NVIDIA driver or the node environment. Report persistent NVML errors on compute nodes to Cyfronet support. Warnings about missing version info are generally harmless.
*   **CodeCarbon Errors:**
    *   If you see "CodeCarbon library not found", ensure it's installed (`pip install codecarbon` or check `requirements.txt`).
    *   Errors related to country detection might occur if `country_iso_code` is not set and auto-detection fails. Specify it in `configs/base.yaml`.
    *   Errors writing the CSV file could be due to permissions in the `results/carbon/` directory.
*   **Optuna Config Update Errors:** Check the Slurm log for the `run_optuna.slurm` job. Common issues include incorrect database path (`--db-path`), target config path (`--config-path`), or study name (`--study-name`), or file permission errors. Ensure the `.db` file exists and the update script (now located in `scripts/tuning_utils/`) has write permission for the target `.yaml` file in `configs/<algo>/` or `configs/bp_baselines/`.
