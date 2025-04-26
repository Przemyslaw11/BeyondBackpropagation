#!/bin/bash
#SBATCH --job-name=BBP_Optuna # Job name - will be prefixed with config later
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=12:00:00      # Optuna might take longer, allocate more time (e.g., 12 hours)
#SBATCH -A plgoncotherapy-gpu-a100 # !!! YOUR GRANT ACCOUNT !!! - IMPORTANT: SET THIS!
#SBATCH -p plgrid-gpu-a100        # PARTITION name for A100 nodes
#SBATCH --output=slurm_logs/%x-%j.out # Standard output log file
#SBATCH --error=slurm_logs/%x-%j.err  # Standard error log file

# --- Configuration via Command Line Arguments ---
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

# --- <<< MODIFICATION: Determine Algorithm from Config >>> ---
ALGORITHM_NAME="UNKNOWN"
echo "Attempting to determine algorithm using yq..."
if command -v yq &> /dev/null; then
    # Using yq (more robust YAML parsing) - assumes v4+ syntax
    # Added error handling for yq command itself
    DETECTED_NAME=$(yq eval '.algorithm.name' "$CONFIG_FILE" 2>/dev/null | tr '[:lower:]' '[:upper:]')
    YQ_EXIT_CODE=$?
    if [ $YQ_EXIT_CODE -eq 0 ] && [ -n "$DETECTED_NAME" ]; then
        ALGORITHM_NAME="$DETECTED_NAME"
        echo "Detected algorithm using yq: $ALGORITHM_NAME"
    else
        echo "yq command failed (Exit Code: $YQ_EXIT_CODE) or returned empty. Falling back to awk."
        # Proceed to awk fallback
    fi
else
    echo "yq command not found. Falling back to awk."
    # Proceed to awk fallback
fi

# Fallback using awk if yq failed or not found
if [ "$ALGORITHM_NAME" = "UNKNOWN" ]; then
    echo "Attempting to determine algorithm using awk..."
    # Use awk to find the 'name' line under the 'algorithm:' block
    # It looks for the algorithm block, then the indented name line, prints the value, and exits.
    # Handles potential quotes around the value.
    DETECTED_NAME=$(awk '
        /^\s*algorithm:/ { in_algo_block=1; next }
        in_algo_block && /^\s*name:/ {
            gsub(/"/, "", $2); # Remove potential quotes
            print toupper($2);
            exit; # Found it, stop processing
        }
        # Reset flag if indentation decreases or another top-level key is found
        in_algo_block && !/^\s+/ { in_algo_block=0 }
    ' "$CONFIG_FILE")

    if [ -n "$DETECTED_NAME" ]; then
        if [ "$DETECTED_NAME" = "MF" ] || [ "$DETECTED_NAME" = "BP" ]; then
            ALGORITHM_NAME="$DETECTED_NAME"
            echo "Detected algorithm using awk: $ALGORITHM_NAME"
        else
            echo "awk detected name '$DETECTED_NAME', but it's not 'MF' or 'BP'."
        fi
    else
        echo "awk could not find algorithm name under 'algorithm:' block."
    fi
fi

echo "Final Detected Algorithm: $ALGORITHM_NAME"
if [ "$ALGORITHM_NAME" = "UNKNOWN" ]; then
    echo "Error: Could not determine algorithm (BP or MF) from config file: $CONFIG_FILE using yq or awk." >&2
    # Optionally dump the relevant config part for debugging:
    echo "--- Relevant config section ---"
    grep -A 2 "algorithm:" "$CONFIG_FILE" || echo "Could not grep algorithm section."
    echo "-----------------------------"
    exit 1
fi
# --- <<< END MODIFICATION >>> ---

CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)
echo "Derived Config Basename: $CONFIG_BASENAME"

# Include algorithm in study name
STUDY_NAME="optuna_${CONFIG_BASENAME}_${ALGORITHM_NAME}"
echo "Using Study Name: $STUDY_NAME"

OPTUNA_OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/optuna"
mkdir -p "$OPTUNA_OUTPUT_DIR"
echo "Optuna output directory: $OPTUNA_OUTPUT_DIR"

DB_PATH="${OPTUNA_OUTPUT_DIR}/${STUDY_NAME}.db"

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Loading modules..."
module purge
module load Python/3.10.4
module load CUDA/12.4.0

echo "Modules loaded:"; module list
echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

OPTUNA_ARGS="--config $CONFIG_FILE --output-dir $OPTUNA_OUTPUT_DIR --study-name $STUDY_NAME"
if [ -n "$NUM_TRIALS" ]; then
    OPTUNA_ARGS="$OPTUNA_ARGS --n-trials $NUM_TRIALS"
fi

echo "Running the Optuna search script..."
echo "Command: srun python scripts/run_optuna_search.py $OPTUNA_ARGS"
srun python scripts/run_optuna_search.py $OPTUNA_ARGS
OPTUNA_EXIT_CODE=$?

# --- <<< MODIFICATION: Select Update Script Based on Algorithm >>> ---
if [ $OPTUNA_EXIT_CODE -eq 0 ]; then
    echo "---------------------------------------------------------------------"
    echo "Optuna search successful (Exit Code: 0)."
    echo "Attempting to automatically update config file..."
    echo "Config file to update: $CONFIG_FILE"
    echo "Using Optuna DB: $DB_PATH"
    echo "Using Study Name: $STUDY_NAME"

    if [ -f "$DB_PATH" ]; then
        UPDATE_SCRIPT_ARGS="--db-path $DB_PATH --config-path $CONFIG_FILE --study-name $STUDY_NAME"

        # Choose the correct update script
        if [ "$ALGORITHM_NAME" = "BP" ]; then
            UPDATE_SCRIPT="scripts/update_bp_configs.py"
        elif [ "$ALGORITHM_NAME" = "MF" ]; then
            UPDATE_SCRIPT="scripts/update_mf_configs.py"
        else
            echo "ERROR: Cannot determine update script for algorithm $ALGORITHM_NAME."
            UPDATE_SCRIPT="" # Prevent execution
        fi

        if [ -n "$UPDATE_SCRIPT" ]; then
             echo "Using Update Script: $UPDATE_SCRIPT"
             echo "Command: srun python $UPDATE_SCRIPT $UPDATE_SCRIPT_ARGS"
             srun python $UPDATE_SCRIPT $UPDATE_SCRIPT_ARGS
             UPDATE_EXIT_CODE=$?

             if [ $UPDATE_EXIT_CODE -eq 0 ]; then
                 echo "SUCCESS: Config file '$CONFIG_FILE' updated automatically by $UPDATE_SCRIPT."
             else
                 echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                 echo "!! ERROR: Config file update FAILED using $UPDATE_SCRIPT (Exit Code: $UPDATE_EXIT_CODE). !!"
                 echo "!!        Check the logs for '$UPDATE_SCRIPT'.             !!"
                 echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
             fi
        fi
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!! ERROR: Optuna DB file not found at expected path: $DB_PATH     !!"
        echo "!!        Cannot automatically update config file '$CONFIG_FILE'.  !!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    fi
    echo "---------------------------------------------------------------------"
else
    echo "---------------------------------------------------------------------"
    echo "Optuna search failed (Exit Code: $OPTUNA_EXIT_CODE)."
    echo "Skipping automatic config file update for: $CONFIG_FILE"
    echo "---------------------------------------------------------------------"
fi
# --- <<< END MODIFICATION >>> ---

echo "Deactivating virtual environment..."
deactivate
echo "#####################################################################"
echo "Optuna Job finished (Exit Code: $OPTUNA_EXIT_CODE)."
date
echo "#####################################################################"
exit $OPTUNA_EXIT_CODE