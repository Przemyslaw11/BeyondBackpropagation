#!/bin/bash
#SBATCH --job-name=BBP_Optuna # Job name - will be prefixed with config later
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16   # Optuna trials might benefit from multiple cores if parallelized internally, but start with proportional
#SBATCH --mem=128000M
#SBATCH --time=12:00:00      # Optuna might take longer, allocate more time (e.g., 12 hours)
#SBATCH -A plgoncotherapy-gpu-a100
#SBATCH -p plgrid-gpu-a100        # PARTITION name for A100 nodes
#SBATCH --output=slurm_logs/%x-%j.out # Standard output log file
#SBATCH --error=slurm_logs/%x-%j.err  # Standard error log file

# --- Configuration via Command Line Arguments ---
# Arg 1: Path to the BASELINE configuration file for Optuna
# Arg 2: Number of Optuna trials to run (overrides config if provided)
# Example: sbatch scripts/slurm_scripts/run_optuna.slurm configs/bp_baselines/fashion_mnist_mlp_4x2000_bp.yaml 100
CONFIG_FILE="$1"
NUM_TRIALS="$2" # Optional, script will use config value if not provided

# Check if CONFIG_FILE argument is provided
if [ -z "$CONFIG_FILE" ]; then
  echo "Error: No configuration file path provided as the first argument."
  echo "Usage: sbatch $0 <path_to_config.yaml> [num_trials]"
  exit 1
fi

# Check if CONFIG_FILE exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at: $CONFIG_FILE"
    exit 1
fi

# --- Environment Setup ---
echo "#####################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Executing command in: $(pwd)"
echo "Using config file for Optuna: $CONFIG_FILE"
if [ -n "$NUM_TRIALS" ]; then
    echo "Number of trials (from command line): $NUM_TRIALS"
else
    echo "Number of trials: (from config file)"
fi
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

EXP_NAME=$(basename "$CONFIG_FILE" .yaml)
echo "Derived Experiment Name: $EXP_NAME"
# Note: To set the SLURM job name dynamically, use the --job-name flag during sbatch submission, e.g.:
# sbatch --job-name="Optuna_${EXP_NAME}" scripts/slurm_scripts/run_optuna.slurm "$CONFIG_FILE" "$NUM_TRIALS"

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

# Construct arguments for the python script
OPTUNA_ARGS="--config $CONFIG_FILE --output-dir $OPTUNA_OUTPUT_DIR --study-name optuna_${EXP_NAME}"
if [ -n "$NUM_TRIALS" ]; then
    OPTUNA_ARGS="$OPTUNA_ARGS --n-trials $NUM_TRIALS"
fi

# --- Run Optuna Search ---
echo "Running the Optuna search script..."
echo "Command: srun python scripts/run_optuna_search.py $OPTUNA_ARGS"
srun python scripts/run_optuna_search.py $OPTUNA_ARGS

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
