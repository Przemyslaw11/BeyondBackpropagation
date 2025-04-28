#!/bin/bash
#SBATCH --job-name=BBP_Optuna                    # Job name - will be prefixed with config later
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=48:00:00      # Optuna might take longer, allocate more time (e.g., 12 hours)
#SBATCH -A plgoncotherapy-gpu-a100 # !!! YOUR GRANT ACCOUNT !!! - IMPORTANT: SET THIS!
#SBATCH -p plgrid-gpu-a100        # PARTITION name for A100 nodes
#SBATCH --output=slurm_logs/%x-%j.out # Standard output log file
#SBATCH --error=slurm_logs/%x-%j.err  # Standard error log file

# --- Configuration via Command Line Arguments ---
CONFIG_FILE="$1"
NUM_TRIALS="$2"  # Optional

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file provided (Arg 1)." >&2
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE" >&2
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
    echo "Num trials (CMD): $NUM_TRIALS"
else
    echo "Num trials: (from config)"
fi
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

# --- Detect Algorithm from Config ---
ALGORITHM_NAME="UNKNOWN"
echo "Attempting to determine algorithm using yq..."
if command -v yq &> /dev/null; then
    # Using yq (more robust YAML parsing) - assumes v4+ syntax
    DETECTED_NAME=$(yq eval '.algorithm.name' "$CONFIG_FILE" 2>/dev/null | tr '[:lower:]' '[:upper:]')
    YQ_EXIT_CODE=$?
    if [ $YQ_EXIT_CODE -eq 0 ] && [ -n "$DETECTED_NAME" ]; then
        ALGORITHM_NAME="$DETECTED_NAME"
        echo "Detected algorithm using yq: $ALGORITHM_NAME"
    else
        echo "yq failed (Code: $YQ_EXIT_CODE) or returned empty. Falling back to awk."
    fi
else
    echo "yq command not found. Falling back to awk."
fi

# --- Fallback: Use awk if yq failed ---
# Improved awk logic to handle different indentations and quotes
if [ "$ALGORITHM_NAME" = "UNKNOWN" ]; then
    echo "Attempting to determine algorithm using awk..."
    DETECTED_NAME=$(awk '
        /^[[:space:]]*algorithm:[[:space:]]*$/ { in_algo_block=1; next }
        in_algo_block && /^[[:space:]]*name:[[:space:]]*/ {
            # Extract value after "name:", removing leading/trailing whitespace and quotes
            val = $0;
            sub(/^[[:space:]]*name:[[:space:]]*/, "", val); # Remove key and leading space
            sub(/^[[:space:]]*["'"'"']/, "", val);          # Remove leading quote
            sub(/["'"'"'][[:space:]]*$/, "", val);          # Remove trailing quote and space
            sub(/[[:space:]]*$/, "", val);                 # Remove trailing space if no quote
            print toupper(val);                            # Convert to uppercase
            exit; # Found it, stop processing
        }
        # Reset flag if indentation decreases or another top-level key is found
        in_algo_block && !/^[[:space:]]+/ { in_algo_block=0 }
        # Handle case where name is on the same line as algorithm: { name: XX }
        /^[[:space:]]*algorithm:[[:space:]]*{.*name:[[:space:]]*/ {
             val = $0;
             sub(/.*name:[[:space:]]*/, "", val);
             sub(/^[[:space:]]*["'"'"']/, "", val);
             sub(/["'"'"'][[:space:]}].*/, "", val); # Remove trailing quote/brace/space
             sub(/[[:space:]}].*/, "", val);       # Remove trailing brace/space if no quote
             print toupper(val);
             exit;
        }
    ' "$CONFIG_FILE")


    if [ -n "$DETECTED_NAME" ]; then
        # Validate against known algorithms
        if [[ "$DETECTED_NAME" == "BP" || "$DETECTED_NAME" == "MF" || "$DETECTED_NAME" == "CAFO" || "$DETECTED_NAME" == "FF" ]]; then
            ALGORITHM_NAME="$DETECTED_NAME"
            echo "Detected algorithm using awk: $ALGORITHM_NAME"
        else
            echo "awk detected name '$DETECTED_NAME', but it's not supported (BP/MF/CAFO/FF)."
        fi
    else
        echo "awk could not find algorithm name under 'algorithm:' block."
    fi
fi

echo "Final Detected Algorithm: $ALGORITHM_NAME"
if [ "$ALGORITHM_NAME" = "UNKNOWN" ]; then
    echo "Error: Could not determine algorithm (BP/MF/CAFO/FF) from config file: $CONFIG_FILE using yq or awk." >&2
    echo "--- Relevant config section ---"
    grep -A 2 -B 1 "algorithm:" "$CONFIG_FILE" || echo "Could not grep algorithm section."
    echo "-----------------------------"
    exit 1
fi

# --- Prepare Optuna Arguments ---
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)
echo "Derived Config Basename: $CONFIG_BASENAME"
STUDY_NAME="optuna_${CONFIG_BASENAME}_${ALGORITHM_NAME}"
echo "Using Study Name: $STUDY_NAME"
OPTUNA_OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results/optuna"
mkdir -p "$OPTUNA_OUTPUT_DIR"
echo "Optuna output directory: $OPTUNA_OUTPUT_DIR"
DB_PATH="${OPTUNA_OUTPUT_DIR}/${STUDY_NAME}.db"

# --- Environment and Modules ---
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Loading modules..."
module purge
module load Python/3.10.4
module load CUDA/12.4.0
echo "Modules loaded:"
module list || true # Use || true to prevent failure if list fails
echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# --- Run Optuna Search ---
OPTUNA_ARGS="--config $CONFIG_FILE --output-dir $OPTUNA_OUTPUT_DIR --study-name $STUDY_NAME"
if [ -n "$NUM_TRIALS" ]; then
    OPTUNA_ARGS="$OPTUNA_ARGS --n-trials $NUM_TRIALS"
fi

echo "Running the Optuna search script..."
echo "Command: srun python scripts/run_optuna_search.py $OPTUNA_ARGS"
srun python scripts/run_optuna_search.py $OPTUNA_ARGS
OPTUNA_EXIT_CODE=$?

# --- Update Config Based on Results ---
if [ $OPTUNA_EXIT_CODE -eq 0 ]; then
    echo "---------------------------------------------------------------------"
    echo "Optuna search successful (Exit Code: 0). Attempting config update..."
    echo "Config file: $CONFIG_FILE"
    echo "Optuna DB: $DB_PATH"
    echo "Study Name: $STUDY_NAME"

    if [ -f "$DB_PATH" ]; then
        UPDATE_SCRIPT_ARGS="--db-path $DB_PATH --config-path $CONFIG_FILE --study-name $STUDY_NAME"
        UPDATE_SCRIPT="" # Initialize empty
        # Select the correct update script based on the detected algorithm
        if [ "$ALGORITHM_NAME" = "BP" ]; then
            UPDATE_SCRIPT="scripts/update_bp_configs.py"
        elif [ "$ALGORITHM_NAME" = "MF" ]; then
            UPDATE_SCRIPT="scripts/update_mf_configs.py"
        elif [ "$ALGORITHM_NAME" = "CAFO" ]; then
            UPDATE_SCRIPT="scripts/update_cafo_configs.py"
        elif [ "$ALGORITHM_NAME" = "FF" ]; then
            UPDATE_SCRIPT="scripts/update_ff_configs.py" # <<< ADDED FF
        else
            echo "ERROR: Cannot determine update script for algorithm $ALGORITHM_NAME."
        fi

        # Check if the selected script exists and execute it
        if [ -n "$UPDATE_SCRIPT" ] && [ -f "$UPDATE_SCRIPT" ]; then
            echo "Using Update Script: $UPDATE_SCRIPT"
            echo "Command: srun python $UPDATE_SCRIPT $UPDATE_SCRIPT_ARGS"
            srun python $UPDATE_SCRIPT $UPDATE_SCRIPT_ARGS
            UPDATE_EXIT_CODE=$?
            if [ $UPDATE_EXIT_CODE -eq 0 ]; then
                echo "SUCCESS: Config '$CONFIG_FILE' updated by $UPDATE_SCRIPT."
            else
                echo "!!!!!!!!!!!!!!!! ERROR: Update FAILED using $UPDATE_SCRIPT (Exit Code: $UPDATE_EXIT_CODE). Check logs. !!!!!!!!!!!!!!!!"
            fi
        elif [ -n "$UPDATE_SCRIPT" ]; then
            echo "!!!!!!!!!!!!!!!! ERROR: Update script '$UPDATE_SCRIPT' not found. Config not updated. !!!!!!!!!!!!!!!!"
        fi
    else
        echo "!!!!!!!!!!!!!!!! ERROR: Optuna DB file not found: $DB_PATH. Cannot update config. !!!!!!!!!!!!!!!!"
    fi
    echo "---------------------------------------------------------------------"
else
    echo "---------------------------------------------------------------------"
    echo "Optuna search failed (Exit Code: $OPTUNA_EXIT_CODE). Skipping config update."
    echo "---------------------------------------------------------------------"
fi

# --- Cleanup ---
echo "Deactivating virtual environment..."
deactivate
echo "#####################################################################"
echo "Optuna Job finished (Exit Code: $OPTUNA_EXIT_CODE)."
date
echo "#####################################################################"

exit $OPTUNA_EXIT_CODE