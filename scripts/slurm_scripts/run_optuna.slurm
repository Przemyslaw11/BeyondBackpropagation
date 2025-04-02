#!/bin/bash
#SBATCH --job-name=BBP_Optuna # Job name - will be prefixed with config later
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16   # Optuna trials might benefit from multiple cores if parallelized internally, but start with proportional
#SBATCH --mem=128000M
#SBATCH --time=12:00:00      # Optuna might take longer, allocate more time (e.g., 12 hours)
#SBATCH -A plgoncotherapy-gpu-a100 # !!! YOUR GRANT ACCOUNT !!!
#SBATCH -p plgrid-gpu-a100        # PARTITION name for A100 nodes
#SBATCH --output=slurm_logs/%x-%j.out # Standard output log file
#SBATCH --error=slurm_logs/%x-%j.err  # Standard error log file

# --- Configuration via Command Line Arguments ---
# Arg 1: Path to the BASELINE configuration file for Optuna
# Arg 2: Number of Optuna trials to run (overrides config if provided)
CONFIG_FILE="$1"
NUM_TRIALS="$2" # Optional

if [ -z "$CONFIG_FILE" ]; then echo "Error: No config file provided (Arg 1)." >&2; exit 1; fi
if [ ! -f "$CONFIG_FILE" ]; then echo "Error: Config file not found: $CONFIG_FILE" >&2; exit 1; fi

# --- Environment Setup ---
echo "#####################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Executing command in: $(pwd)"
echo "Using config file for Optuna: $CONFIG_FILE"
if [ -n "$NUM_TRIALS" ]; then echo "Num trials (CMD): $NUM_TRIALS"; else echo "Num trials: (from config)"; fi
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

EXP_NAME=$(basename "$CONFIG_FILE" .yaml)
echo "Derived Experiment Name: $EXP_NAME"
# Dynamic job name via sbatch:
# sbatch --job-name="Optuna_${EXP_NAME}" scripts/slurm_scripts/run_optuna.slurm "$CONFIG_FILE" "$NUM_TRIALS"

cd $SLURM_SUBMIT_DIR || exit 1

echo "Loading modules..."
module purge
module load Python/3.10.4   # Use the SAME version
module load CUDA/12.4.0     # Use the SAME version

echo "Modules loaded:"
module list

echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

OPTUNA_OUTPUT_DIR="results/optuna/${EXP_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OPTUNA_OUTPUT_DIR"
echo "Optuna output directory: $OPTUNA_OUTPUT_DIR"

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
if [ $EXIT_CODE -eq 0 ]; then
    echo "Optuna search finished successfully."
    echo "REMINDER: Check logs/YAML in $OPTUNA_OUTPUT_DIR"
    echo "          Update 'optimizer' section in $CONFIG_FILE"
else
    echo "Optuna search finished with errors (Exit Code: $EXIT_CODE)."
fi
echo "#####################################################################"
date
echo "#####################################################################"

exit $EXIT_CODE