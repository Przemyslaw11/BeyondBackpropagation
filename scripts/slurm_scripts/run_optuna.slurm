#!/bin/bash
#SBATCH --job-name=BBP_Optuna # Job name - will be prefixed with config later
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16   # Optuna trials might benefit from multiple cores if parallelized internally, but start with proportional
#SBATCH --mem=128000M
#SBATCH --time=12:00:00      # Optuna might take longer, allocate more time (e.g., 12 hours)
#SBATCH -A plgoncotherapy-gpu-a100 # !!! YOUR GRANT ACCOUNT !!! - IMPORTANT: SET THIS!
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

CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)
echo "Derived Config Basename: $CONFIG_BASENAME"

# Use a consistent study name based on the config basename
# The Python script run_optuna_search.py will use this if --study-name is passed.
# If you prefer the timestamped version, remove --study-name from OPTUNA_ARGS
# and adjust the DB_PATH construction below accordingly.
STUDY_NAME="optuna_${CONFIG_BASENAME}"
echo "Using Study Name: $STUDY_NAME"

# Define the output directory relative to the submission directory
OPTUNA_OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/optuna"
mkdir -p "$OPTUNA_OUTPUT_DIR"
echo "Optuna output directory: $OPTUNA_OUTPUT_DIR"

# Define the expected path to the database file
# This MUST match how run_optuna_search.py names the DB when --study-name is used
DB_PATH="${OPTUNA_OUTPUT_DIR}/${STUDY_NAME}.db"

# Change to the submission directory (project root)
cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Loading modules..."
module purge
module load Python/3.10.4   # Use the SAME version
module load CUDA/12.4.0     # Use the SAME version

echo "Modules loaded:"
module list

echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Construct arguments for the optuna search script
OPTUNA_ARGS="--config $CONFIG_FILE --output-dir $OPTUNA_OUTPUT_DIR --study-name $STUDY_NAME"
if [ -n "$NUM_TRIALS" ]; then
    OPTUNA_ARGS="$OPTUNA_ARGS --n-trials $NUM_TRIALS"
fi

# --- Run Optuna Search ---
echo "Running the Optuna search script..."
echo "Command: srun python scripts/run_optuna_search.py $OPTUNA_ARGS"
srun python scripts/run_optuna_search.py $OPTUNA_ARGS

OPTUNA_EXIT_CODE=$?

# --- AUTOMATICALLY UPDATE CONFIG IF OPTUNA SUCCEEDED ---
if [ $OPTUNA_EXIT_CODE -eq 0 ]; then
    echo "---------------------------------------------------------------------"
    echo "Optuna search successful (Exit Code: 0)."
    echo "Attempting to automatically update baseline config file..."
    echo "Config file to update: $CONFIG_FILE"
    echo "Using Optuna DB: $DB_PATH"
    echo "Using Study Name: $STUDY_NAME"

    if [ -f "$DB_PATH" ]; then
        # Run the update script
        UPDATE_SCRIPT_ARGS="--db-path $DB_PATH --config-path $CONFIG_FILE --study-name $STUDY_NAME"
        echo "Command: srun python scripts/update_bp_configs.py $UPDATE_SCRIPT_ARGS"
        srun python scripts/update_bp_configs.py $UPDATE_SCRIPT_ARGS
        UPDATE_EXIT_CODE=$?

        if [ $UPDATE_EXIT_CODE -eq 0 ]; then
            echo "SUCCESS: Config file '$CONFIG_FILE' updated automatically."
            echo "         YAML backup should exist in the same directory."
        else
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "!! ERROR: Config file update FAILED (Exit Code: $UPDATE_EXIT_CODE). !!"
            echo "!!        Check the logs for 'update_bp_configs.py'.             !!"
            echo "!!        The config file '$CONFIG_FILE' was NOT updated.        !!"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # Decide if this should make the whole Slurm job fail
            # exit 1 # Uncomment if update failure should cause job failure
        fi
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!! ERROR: Optuna DB file not found at expected path: $DB_PATH     !!"
        echo "!!        Cannot automatically update config file '$CONFIG_FILE'.  !!"
        echo "!!        Check 'run_optuna_search.py' output/logs.              !!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # Decide if this should make the whole Slurm job fail
        # exit 1 # Uncomment if DB not found should cause job failure
    fi
    echo "---------------------------------------------------------------------"

else
    echo "---------------------------------------------------------------------"
    echo "Optuna search failed (Exit Code: $OPTUNA_EXIT_CODE)."
    echo "Skipping automatic config file update for: $CONFIG_FILE"
    echo "---------------------------------------------------------------------"
fi

# --- Cleanup ---
echo "Deactivating virtual environment..."
deactivate

echo "#####################################################################"
if [ $OPTUNA_EXIT_CODE -eq 0 ]; then
    echo "Optuna Job finished. Config update attempted (see logs above)."
else
    echo "Optuna Job finished with errors (Exit Code: $OPTUNA_EXIT_CODE)."
fi
echo "#####################################################################"
date
echo "#####################################################################"

# Final exit code reflects Optuna search status primarily
exit $OPTUNA_EXIT_CODE