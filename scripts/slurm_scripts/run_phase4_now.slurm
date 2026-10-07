#!/bin/bash
# Phase 4 finals runner: walks an explicit (config, seed) list on plgrid-now.
#
# Phase 3 hard-coded its pair list into the runner. Phase 4 has four campaigns with
# different seed counts (n=7 for FF, n=5 for CaFo and the CNN baselines, n=3 for the
# reproduction check), so the list lives in a spec file named by PHASE4_SPEC instead
# and one script serves all of them.
#
# Spec format, one campaign entry per line, blank lines and # comments ignored:
#   <config path> <comma-separated seeds>
#
# plgrid-now caps a job at one hour, so a chunk stops starting runs once the budget
# is spent and the driver submits the next one. Runs that already have a summary
# JSON are skipped, which is what makes the chunking safe to repeat.
#
#SBATCH --job-name=P4_now
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH -A plglscclass26-gpu-a100
#SBATCH -p plgrid-now
#SBATCH --output=logs/phase4/%x-%j.out
#SBATCH --error=logs/phase4/%x-%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1

SPEC=${PHASE4_SPEC:?PHASE4_SPEC must name a spec file}
[ -f "$SPEC" ] || { echo "ERROR: spec file '$SPEC' not found"; exit 1; }

PAIR_CONFIG=()
PAIR_SEED=()
while read -r SPEC_CONFIG SPEC_SEEDS; do
    case "$SPEC_CONFIG" in ''|\#*) continue ;; esac
    IFS=',' read -r -a SEEDS <<< "$SPEC_SEEDS"
    for S in "${SEEDS[@]}"; do
        PAIR_CONFIG+=("$SPEC_CONFIG")
        PAIR_SEED+=("$S")
    done
done < "$SPEC"

NUM_JOBS=${#PAIR_CONFIG[@]}
[ "$NUM_JOBS" -gt 0 ] || { echo "ERROR: spec '$SPEC' expanded to no runs"; exit 1; }

START_INDEX=${PHASE4_START:-1}
END_INDEX=${PHASE4_END:-$NUM_JOBS}
# Stop starting runs at 45 min so a long trailing run still fits the 1 h limit.
BUDGET_SEC=${PHASE4_BUDGET:-2700}
# The budget alone only protects runs shorter than the 15 min it leaves spare, so
# the loop also refuses to start a run longer than the wall clock actually left.
WALL_SEC=${PHASE4_WALL:-3600}
TEARDOWN_SEC=120
if [ "$END_INDEX" -gt "$NUM_JOBS" ]; then END_INDEX=$NUM_JOBS; fi

module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export WANDB_MODE=online
# Unset deliberately: unlike the ladder, Phase 4 reproduces and extends the published
# protocol, in which BP and FF restore their best-validation weights before testing.
unset BBP_DISABLE_CHECKPOINTS

START_TS=$(date +%s)
COMPLETED=0
SKIPPED=0
FAILED=0
LONGEST_RUN_SEC=0

echo "#####################################################################"
echo "Phase 4 finals on $(hostname): ${SPEC}, indices ${START_INDEX}-${END_INDEX} of ${NUM_JOBS}, budget ${BUDGET_SEC}s"
date
echo "#####################################################################"

for FLAT in $(seq "$START_INDEX" "$END_INDEX"); do
    ELAPSED=$(( $(date +%s) - START_TS ))
    if [ "$ELAPSED" -ge "$BUDGET_SEC" ]; then
        echo "INFO: budget reached after ${ELAPSED}s, stopping before index ${FLAT}."
        break
    fi

    WALL_LEFT=$(( WALL_SEC - ELAPSED - TEARDOWN_SEC ))
    if [ "$LONGEST_RUN_SEC" -gt 0 ] && [ "$WALL_LEFT" -lt "$LONGEST_RUN_SEC" ]; then
        echo "INFO: ${WALL_LEFT}s of wall clock left but the longest run here took ${LONGEST_RUN_SEC}s, stopping before index ${FLAT}."
        break
    fi

    CONFIG_FILE=${PAIR_CONFIG[$((FLAT - 1))]}
    SEED=${PAIR_SEED[$((FLAT - 1))]}

    EXP_NAME=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['experiment_name'])" "$CONFIG_FILE")
    RESULT_ROOT=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1])).get('run_summary_dir','results/runs'))" "$CONFIG_FILE")
    if [ "$RESULT_ROOT" = "results/runs" ]; then
        echo "ERROR: ${CONFIG_FILE} writes to results/runs, which holds the ladder. Set run_summary_dir."
        FAILED=$((FAILED + 1))
        continue
    fi
    RESULT="${RESULT_ROOT}/${EXP_NAME}/${EXP_NAME}_seed${SEED}.json"
    if [ -f "$RESULT" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "--- [${FLAT}/${END_INDEX}] ${EXP_NAME} seed ${SEED} (elapsed ${ELAPSED}s)"
    RUN_TS=$(date +%s)
    EXPERIMENT_SEED=$SEED python scripts/run_experiment.py --config "$CONFIG_FILE"
    RUN_RC=$?
    RUN_SEC=$(( $(date +%s) - RUN_TS ))
    if [ "$RUN_SEC" -gt "$LONGEST_RUN_SEC" ]; then LONGEST_RUN_SEC=$RUN_SEC; fi
    if [ "$RUN_RC" -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
    else
        FAILED=$((FAILED + 1))
        echo "ERROR: ${EXP_NAME} seed ${SEED} exited non-zero."
    fi
done

TOTAL_ELAPSED=$(( $(date +%s) - START_TS ))
echo "#####################################################################"
echo "PILOT_SUMMARY completed=${COMPLETED} skipped=${SKIPPED} failed=${FAILED} elapsed=${TOTAL_ELAPSED}s"
date
echo "#####################################################################"
