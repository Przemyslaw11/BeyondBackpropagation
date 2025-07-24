#!/bin/bash
#SBATCH --job-name=BBP_Optuna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=08:00:00
#SBATCH -A plgoncotherapy-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

CONFIG_FILE="$1"
NUM_TRIALS="$2"

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

# --- Algorithm Detection ---
ALGORITHM_NAME="UNKNOWN"
echo "Attempting to determine algorithm using yq..."
if command -v yq &> /dev/null; then
    DETECTED_NAME=$(yq eval '.algorithm.name' "$CONFIG_FILE" 2>/dev/null | tr '[:lower:]' '[:upper:]')
    YQ_EXIT_CODE=$?
    if [ $YQ_EXIT_CODE -eq 0 ] && [ -n "$DETECTED_NAME" ] && [[ "$DETECTED_NAME" == "BP" || "$DETECTED_NAME" == "MF" || "$DETECTED_NAME" == "CAFO" || "$DETECTED_NAME" == "FF" ]]; then
        ALGORITHM_NAME="$DETECTED_NAME"
        echo "Detected algorithm using yq: $ALGORITHM_NAME"
    else
        echo "yq failed (Code: $YQ_EXIT_CODE), returned empty, or returned unsupported name '$DETECTED_NAME'. Falling back to awk."
    fi
else
    echo "yq command not found. Falling back to awk."
fi

if [ "$ALGORITHM_NAME" = "UNKNOWN" ]; then
    echo "Attempting to determine algorithm using awk..."
    DETECTED_NAME=$(awk '
        BEGIN { found_algo_key=0; algo_name="UNKNOWN"; }

        # Match the start of the algorithm block
        /^[[:space:]]*algorithm:[[:space:]]*$/ {
            found_algo_key=1;
            next; # Move to the next line immediately
        }

        # If we are inside the algorithm block, look for the name line
        found_algo_key == 1 && /^[[:space:]]+name:[[:space:]]*/ {
            val = $0;
            # Extract value: remove "name:", leading/trailing spaces, quotes, comments
            sub(/^[[:space:]]*name:[[:space:]]*/, "", val);
            sub(/^[[:space:]]*/, "", val);
            sub(/^"/, "", val); sub(/"$/, "", val);
            sub(/^'"'"'/, "", val); sub(/'"'"'$/, "", val);
            sub(/[[:space:]]*#.*$/, "", val);
            sub(/[[:space:]]*$/, "", val);

            algo_name_extracted = toupper(val);

            # Validate immediately
            if (algo_name_extracted == "BP" || algo_name_extracted == "MF" || algo_name_extracted == "CAFO" || algo_name_extracted == "FF") {
                algo_name = algo_name_extracted;
                # print "AWK_DEBUG: Found valid name: " algo_name # Optional debug
                exit; # Exit awk successfully after finding the valid name
            } else {
                # print "AWK_DEBUG: Found potentially invalid name: " algo_name_extracted # Optional debug
                found_algo_key = 0; # Reset flag if invalid name found under algorithm
            }
        }

        # Reset flag if we hit another top-level key OR leave the indentation level
        found_algo_key == 1 && (!/^[[:space:]]+/ || /^[a-zA-Z0-9_]+:/) {
            # print "AWK_DEBUG: Resetting flag on line: " $0 # Optional debug
            found_algo_key = 0;
        }

        END { print algo_name } # Print the final result (will be UNKNOWN if not found/validated)

    ' "$CONFIG_FILE")

    if [[ "$DETECTED_NAME" == "BP" || "$DETECTED_NAME" == "MF" || "$DETECTED_NAME" == "CAFO" || "$DETECTED_NAME" == "FF" ]]; then
        ALGORITHM_NAME="$DETECTED_NAME"
        echo "Detected algorithm using awk: $ALGORITHM_NAME"
    elif [ "$DETECTED_NAME" != "UNKNOWN" ]; then
        echo "awk detected name '$DETECTED_NAME', but it's not supported (BP/MF/CAFO/FF) or extraction failed."
    else
        echo "awk could not find a valid algorithm name under 'algorithm:' block."
    fi
fi

echo "Final Detected Algorithm: $ALGORITHM_NAME"
if [ "$ALGORITHM_NAME" = "UNKNOWN" ]; then
    echo "Error: Could not determine algorithm (BP/MF/CAFO/FF) from config file: $CONFIG_FILE using yq or awk." >&2
    echo "--- Relevant config section ---"
    grep -A 3 -B 1 "algorithm:" "$CONFIG_FILE" || echo "Could not grep algorithm section."
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

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

echo "Loading modules..."
module purge
module load Python/3.10.4
module load CUDA/12.4.0
echo "Modules loaded:"
module list || true
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
        # --- Determine the TARGET FINAL config file ---
        TARGET_CONFIG_FILE=""
        if [[ "$CONFIG_FILE" == *"configs/tuning/"* ]]; then
            SUBDIR_NAME=$(echo "$ALGORITHM_NAME" | tr '[:upper:]' '[:lower:]')
            BASE_TUNE_NAME=$(basename "$CONFIG_FILE" "_tune.yaml")
            BASE_FINAL_NAME=$(echo "$BASE_TUNE_NAME" | sed -e "s/_${SUBDIR_NAME}$//" -e "s/^${SUBDIR_NAME}_//")
            TARGET_CONFIG_FILE="configs/${SUBDIR_NAME}/${BASE_FINAL_NAME}.yaml"
            echo "Derived target FINAL config path (for FF/MF/CaFo): $TARGET_CONFIG_FILE"
        elif [[ "$CONFIG_FILE" == *"configs/bp_baselines/"* ]]; then
             TARGET_CONFIG_FILE="$CONFIG_FILE"
             echo "Target config path (for BP): $TARGET_CONFIG_FILE (updating tuning file)"
        else
            echo "WARNING: Could not determine target FINAL config path structure from tuning path '$CONFIG_FILE'. Update script will target the tuning file."
            TARGET_CONFIG_FILE="$CONFIG_FILE"
        fi

        if [ ! -f "$TARGET_CONFIG_FILE" ]; then
             echo "!!!!!!!!!!!!!!!! ERROR: Target config file '$TARGET_CONFIG_FILE' not found. Cannot update. !!!!!!!!!!!!!!!!"
        else
            # --- Select and Run Update Script ---
            UPDATE_SCRIPT_ARGS="--db-path $DB_PATH --config-path $TARGET_CONFIG_FILE --study-name $STUDY_NAME"
            UPDATE_SCRIPT=""
            if [ "$ALGORITHM_NAME" = "BP" ]; then
                UPDATE_SCRIPT="scripts/tuning_utils/update_bp_configs.py"
            elif [ "$ALGORITHM_NAME" = "MF" ]; then
                UPDATE_SCRIPT="scripts/tuning_utils/update_mf_configs.py"
            elif [ "$ALGORITHM_NAME" = "CAFO" ]; then
                UPDATE_SCRIPT="scripts/tuning_utils/update_cafo_configs.py"
            elif [ "$ALGORITHM_NAME" = "FF" ]; then
                UPDATE_SCRIPT="scripts/tuning_utils/update_ff_configs.py"
            else
                echo "ERROR: Cannot determine update script for algorithm $ALGORITHM_NAME."
            fi

            if [ -n "$UPDATE_SCRIPT" ] && [ -f "$UPDATE_SCRIPT" ]; then
                echo "Using Update Script: $UPDATE_SCRIPT"
                echo "Targeting config: $TARGET_CONFIG_FILE"
                echo "Command: srun python $UPDATE_SCRIPT $UPDATE_SCRIPT_ARGS"
                srun python $UPDATE_SCRIPT $UPDATE_SCRIPT_ARGS
                UPDATE_EXIT_CODE=$?
                if [ $UPDATE_EXIT_CODE -eq 0 ]; then
                    echo "SUCCESS: Config '$TARGET_CONFIG_FILE' updated by $UPDATE_SCRIPT."
                else
                    echo "!!!!!!!!!!!!!!!! ERROR: Update FAILED using $UPDATE_SCRIPT (Exit Code: $UPDATE_EXIT_CODE). Check logs. !!!!!!!!!!!!!!!!"
                fi
            elif [ -n "$UPDATE_SCRIPT" ]; then
                echo "!!!!!!!!!!!!!!!! ERROR: Update script '$UPDATE_SCRIPT' not found. Config not updated. !!!!!!!!!!!!!!!!"
            fi
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