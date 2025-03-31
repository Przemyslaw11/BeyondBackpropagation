#!/bin/bash
#SBATCH --job-name=BBP_Optuna # Job name - will be prefixed with config later
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16   # Optuna trials might benefit from multiple cores if parallelized internally, but start with proportional
#SBATCH --mem=128000M
#SBATCH --time=12:00:00      # Optuna might take longer, allocate more time (e.g., 12 hours)
#SBATCH -A <your_grant_name>-gpu-a100 # ACCOUNT NAME - REPLACE <your_grant_name>
#SBATCH -p plgrid-gpu-a100        # PARTITION name for A100 nodes
#SBATCH --output=slurm_logs/%x-%j.out # Standard output log file
#SBATCH --error=slurm_logs/%x-%j.err  # Standard error log file

# --- User Configuration ---
# REQUIRED: Set the path to the BASELINE configuration file for Optuna
CONFIG_FILE="configs/bp_baselines/fashion_mnist_mlp_4x2000_bp.yaml" # <--- CHANGE THIS LINE

# --- Environment Setup ---
echo "#####################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Executing command in: $(pwd)"
echo "Using config file for Optuna: $CONFIG_FILE"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

EXP_NAME=$(basename "$CONFIG_FILE" .yaml)
echo "Setting log identifier to: BBP_Optuna_$EXP_NAME"

cd $SLURM_SUBMIT_DIR || exit 1

echo "Loading modules..."
module purge
module load plgrid/tools/python/3.10 # Use the SAME version
module load plgrid/tools/cuda/11.8   # Use the SAME version

echo "Modules loaded:"
module list

echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Output directory for Optuna results, relative to project root
OPTUNA_OUTPUT_DIR="results/optuna/${EXP_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OPTUNA_OUTPUT_DIR" # Create directory for DB and logs
echo "Optuna output will be saved in: $OPTUNA_OUTPUT_DIR"

# --- Run Optuna Search ---
echo "Running the Optuna search script..."
srun python scripts/run_optuna_search.py \
    --config "$CONFIG_FILE" \
    --output-dir "$OPTUNA_OUTPUT_DIR" \
    --study-name "optuna_${EXP_NAME}"
    # --n-trials 50 # Optionally override n_trials from config here

EXIT_CODE=$?

# --- Cleanup ---
echo "Deactivating virtual environment..."
deactivate

echo "#####################################################################"
# Print reminder to check results and update config
if [ $EXIT_CODE -eq 0 ]; then
    echo "Optuna search finished successfully."
    echo "REMINDER: Check the logs and YAML file in $OPTUNA_OUTPUT_DIR"
    echo "          Manually update the 'optimizer' section in $CONFIG_FILE"
    echo "          with the best hyperparameters before running the final baseline."
else
    echo "Optuna search finished with errors (Exit Code: $EXIT_CODE)."
fi
echo "#####################################################################"
date
echo "#####################################################################"

exit $EXIT_CODE