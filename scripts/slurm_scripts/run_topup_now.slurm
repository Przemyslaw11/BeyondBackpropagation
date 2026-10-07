#!/bin/bash
# Phase 3, task 11: extra seeds for the three contrasts the n=20 ladder left undecided.
#
# WHY A SEPARATE RUNNER
# run_ladder_now.slurm indexes as IDX=(FLAT-1)/NUM_SEEDS over a fixed 20-seed list, so
# widening that list would renumber every existing index. This script instead carries an
# explicit (config, seed) pair list, which lets each contrast take the seed count its own
# variance demands rather than forcing one n across the whole ladder.
#
# WHICH CONTRASTS AND WHY THESE COUNTS
# From the realised n=20 standard deviations, the seeds needed for TOST at the
# pre-registered 0.25 pp margin, alpha 0.05, power 0.80:
#   fashion bp->bp_ds        observed +0.120 pp, sd 0.5777 -> n=122
#   fashion bp->mf_recompute observed +0.148 pp, sd 0.2820 -> n=48
#   mnist   bp->bp_ds        observed +0.137 pp, sd 0.3495 -> n=60
# mnist bp->mf_cache_* needs n=603 and is deliberately omitted: ~20 GPU-h to certify a
# 0.217 pp gap that sits too close to the margin to be worth it. That one gets reported
# as undecidable.
#
# The analysis pairs on the seed intersection, so a rung carrying more seeds than its
# partner costs nothing. These write to results/runs because they ARE ladder runs, merely
# additional seeds; new seeds create new keys and cannot overwrite the existing 400.
#
#SBATCH --job-name=T_now
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH -A plglscclass26-gpu-a100
#SBATCH -p plgrid-now
#SBATCH --output=logs/phase3/%x-%j.out
#SBATCH --error=logs/phase3/%x-%j.err

# "config first_seed last_seed"; the ladder already holds seeds 42-61.
TOPUP_SPEC=(
    "configs/bp_baselines/mnist_mlp_2x1000_bp.yaml 62 101"
    "configs/bp_ds/mnist_mlp_2x1000.yaml 62 101"
    "configs/bp_baselines/fashion_mnist_mlp_2x1000_bp.yaml 62 163"
    "configs/bp_ds/fashion_mnist_mlp_2x1000.yaml 62 163"
    "configs/mf/fashion_mnist_mlp_2x1000.yaml 62 89"
)

PAIR_CONFIG=()
PAIR_SEED=()
for ENTRY in "${TOPUP_SPEC[@]}"; do
    read -r SPEC_CONFIG SPEC_FIRST SPEC_LAST <<< "$ENTRY"
    for S in $(seq "$SPEC_FIRST" "$SPEC_LAST"); do
        PAIR_CONFIG+=("$SPEC_CONFIG")
        PAIR_SEED+=("$S")
    done
done

NUM_JOBS=${#PAIR_CONFIG[@]}

START_INDEX=${TOPUP_NOW_START:-1}
END_INDEX=${TOPUP_NOW_END:-$NUM_JOBS}
# Stop starting runs at 45 min so a long trailing run still fits the 1 h limit.
BUDGET_SEC=${TOPUP_NOW_BUDGET:-2700}

if [ "$END_INDEX" -gt "$NUM_JOBS" ]; then END_INDEX=$NUM_JOBS; fi

cd "$SLURM_SUBMIT_DIR" || exit 1

module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export WANDB_MODE=online
# Uniform protocol: no rung restores best-validation weights before testing.
export BBP_DISABLE_CHECKPOINTS=1

START_TS=$(date +%s)
COMPLETED=0
SKIPPED=0
FAILED=0

echo "#####################################################################"
echo "Top-up pilot on $(hostname), indices ${START_INDEX}-${END_INDEX} of ${NUM_JOBS}, budget ${BUDGET_SEC}s"
date
echo "#####################################################################"

for FLAT in $(seq "$START_INDEX" "$END_INDEX"); do
    ELAPSED=$(( $(date +%s) - START_TS ))
    if [ "$ELAPSED" -ge "$BUDGET_SEC" ]; then
        echo "INFO: budget reached after ${ELAPSED}s, stopping before index ${FLAT}."
        break
    fi

    CONFIG_FILE=${PAIR_CONFIG[$((FLAT - 1))]}
    SEED=${PAIR_SEED[$((FLAT - 1))]}

    EXP_NAME=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['experiment_name'])" "$CONFIG_FILE")
    RESULT_ROOT=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1])).get('run_summary_dir','results/runs'))" "$CONFIG_FILE")
    RESULT="${RESULT_ROOT}/${EXP_NAME}/${EXP_NAME}_seed${SEED}.json"
    if [ -f "$RESULT" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "--- [${FLAT}/${END_INDEX}] ${EXP_NAME} seed ${SEED} (elapsed ${ELAPSED}s)"
    EXPERIMENT_SEED=$SEED python scripts/run_experiment.py --config "$CONFIG_FILE"
    if [ $? -eq 0 ]; then
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
