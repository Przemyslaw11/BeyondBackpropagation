#!/bin/bash
# Phase 3, task 8: ladder runs as a pilot job on the plgrid-now partition.
#
# WHY A PILOT JOB RATHER THAN AN ARRAY
# plgrid-now starts jobs in seconds where plgrid-gpu-a100 made us wait 51 hours,
# because our grant holds RawShares=1 and FairShare is pinned at 0. But its QOS
# allows MaxSubmitJobsPU=1: one job in the queue, ever. An array is impossible
# and so is a dependency chain. So one job claims a whole hour and works through
# as many ladder runs as fit, paying the module load and venv activation once.
#
# RESUMABILITY
# A run is skipped when its summary JSON already exists, which is what lets the
# driver resubmit blindly until the ladder is complete. The same property makes
# a job that dies mid-run harmless: only the interrupted run is lost.
#
#SBATCH --job-name=L_now
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH -A plglscclass26-gpu-a100
#SBATCH -p plgrid-now
#SBATCH --output=slurm_logs/phase3/%x-%j.out
#SBATCH --error=slurm_logs/phase3/%x-%j.err

CONFIG_FILES=(
    # --- MNIST 2x1000: all six rungs ---
    "configs/bp_baselines/mnist_mlp_2x1000_bp.yaml"
    "configs/bp_ds/mnist_mlp_2x1000.yaml"
    "configs/mf_joint/mnist_mlp_2x1000.yaml"
    "configs/mf/mnist_mlp_2x1000.yaml"
    "configs/mf/mnist_mlp_2x1000_cache_device.yaml"
    "configs/mf/mnist_mlp_2x1000_cache_host.yaml"

    # --- Fashion-MNIST 2x1000: all six rungs ---
    "configs/bp_baselines/fashion_mnist_mlp_2x1000_bp.yaml"
    "configs/bp_ds/fashion_mnist_mlp_2x1000.yaml"
    "configs/mf_joint/fashion_mnist_mlp_2x1000.yaml"
    "configs/mf/fashion_mnist_mlp_2x1000.yaml"
    "configs/mf/fashion_mnist_mlp_2x1000_cache_device.yaml"
    "configs/mf/fashion_mnist_mlp_2x1000_cache_host.yaml"

    # --- CIFAR-10 3x2000: rungs 1-4 only, augmentation rules out the cache ---
    "configs/bp_baselines/cifar10_mlp_3x2000_bp.yaml"
    "configs/bp_ds/cifar10_mlp_3x2000.yaml"
    "configs/mf_joint/cifar10_mlp_3x2000.yaml"
    "configs/mf/cifar10_mlp_3x2000.yaml"

    # --- CIFAR-100 3x2000: rungs 1-4 only, same reason ---
    "configs/bp_baselines/cifar100_mlp_3x2000_bp.yaml"
    "configs/bp_ds/cifar100_mlp_3x2000.yaml"
    "configs/mf_joint/cifar100_mlp_3x2000.yaml"
    "configs/mf/cifar100_mlp_3x2000.yaml"

    # --- Task 10 diagnostic, index 20, flat 401-420. Not a rung. ---
    "configs/diagnostics/cifar10_mlp_3x2000_legacy_es.yaml"

    # --- Equal-total-epoch condition, indices 21-24, flat 421-500. Not a rung. ---
    # Rung 4 under a total rather than a per-stage epoch cap. These write to
    # results/equal_epochs, so they cannot collide with rung 4 in results/runs.
    "configs/diagnostics/mnist_mlp_2x1000_equal_epochs.yaml"
    "configs/diagnostics/fashion_mnist_mlp_2x1000_equal_epochs.yaml"
    "configs/diagnostics/cifar10_mlp_3x2000_equal_epochs.yaml"
    "configs/diagnostics/cifar100_mlp_3x2000_equal_epochs.yaml"
)

SEEDS=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61)

NUM_SEEDS=${#SEEDS[@]}
NUM_JOBS=$((${#CONFIG_FILES[@]} * NUM_SEEDS))

# Defaults cover the twelve MNIST/Fashion-MNIST rung-configs, the only ones that
# carry all six rungs and so the only ones that answer reviewer R3 in full.
START_INDEX=${LADDER_NOW_START:-1}
END_INDEX=${LADDER_NOW_END:-240}
# Stop starting runs at 45 min so a long trailing run still fits the 1 h limit.
BUDGET_SEC=${LADDER_NOW_BUDGET:-2700}

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
echo "Ladder pilot on $(hostname), indices ${START_INDEX}-${END_INDEX}, budget ${BUDGET_SEC}s"
date
echo "#####################################################################"

for FLAT in $(seq "$START_INDEX" "$END_INDEX"); do
    ELAPSED=$(( $(date +%s) - START_TS ))
    if [ "$ELAPSED" -ge "$BUDGET_SEC" ]; then
        echo "INFO: budget reached after ${ELAPSED}s, stopping before index ${FLAT}."
        break
    fi

    IDX=$(( (FLAT - 1) / NUM_SEEDS ))
    SEED_IDX=$(( (FLAT - 1) % NUM_SEEDS ))
    CONFIG_FILE=${CONFIG_FILES[$IDX]}
    SEED=${SEEDS[$SEED_IDX]}

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
        echo "WARNING: index ${FLAT} (${EXP_NAME} seed ${SEED}) failed; continuing."
    fi
done

WANDB_LOCAL_DIR="${WANDB_DIR:-$SLURM_SUBMIT_DIR}/wandb"
if compgen -G "$WANDB_LOCAL_DIR/offline-run-*" > /dev/null; then
    wandb sync "$WANDB_LOCAL_DIR"/offline-run-* || echo "WARNING: wandb sync failed"
fi

deactivate

echo "#####################################################################"
echo "PILOT_SUMMARY completed=${COMPLETED} skipped=${SKIPPED} failed=${FAILED} elapsed=$(( $(date +%s) - START_TS ))s"
date
echo "#####################################################################"
exit 0
