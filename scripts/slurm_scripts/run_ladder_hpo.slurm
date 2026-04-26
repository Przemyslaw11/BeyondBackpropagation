#!/bin/bash
# Phase 3, task 7: tune the four independently-tunable rungs of the ablation
# ladder under one shared protocol.
#
# Four rungs x four MLP configurations = 16 studies, 50 trials each.
#
# Rungs 5 and 6 do not appear here. They are rung 4 with a different activation
# cache and MUST reuse rung 4's hyperparameters, otherwise the cache comparison
# would measure the search as well as the cache.
#
# BP is re-tuned here at 50 trials. The published BP baselines were tuned with
# 5-10 trials while the forward-only methods got 50, and the ladder must not
# inherit that budget asymmetry.
#
# Submitted as 16 INDEPENDENT jobs, not as an array. A 16-task array is scheduled
# as one unit against the partition's limits, so the whole ladder waits on the
# worst-placed task. Independent jobs backfill into gaps and start as slots free.
# Select the study with LADDER_HPO_INDEX (1-16); SLURM_ARRAY_TASK_ID still works.
#
#SBATCH --job-name=BBP_LadderHPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=08:00:00
#SBATCH -A plglscclass26-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/phase3/%x-%j.out
#SBATCH --error=slurm_logs/phase3/%x-%j.err

TUNE_CONFIGS=(
    # --- Rung 1: BP ---
    "configs/tuning/bp_mnist_mlp_2x1000_tune.yaml"
    "configs/tuning/bp_fashion_mnist_mlp_2x1000_tune.yaml"
    "configs/tuning/bp_cifar10_mlp_3x2000_tune.yaml"
    "configs/tuning/bp_cifar100_mlp_3x2000_tune.yaml"

    # --- Rung 2: BP-DS ---
    "configs/tuning/bpds_mnist_mlp_2x1000_tune.yaml"
    "configs/tuning/bpds_fashion_mnist_mlp_2x1000_tune.yaml"
    "configs/tuning/bpds_cifar10_mlp_3x2000_tune.yaml"
    "configs/tuning/bpds_cifar100_mlp_3x2000_tune.yaml"

    # --- Rung 3: MF-Joint ---
    "configs/tuning/mfjoint_mnist_mlp_2x1000_tune.yaml"
    "configs/tuning/mfjoint_fashion_mnist_mlp_2x1000_tune.yaml"
    "configs/tuning/mfjoint_cifar10_mlp_3x2000_tune.yaml"
    "configs/tuning/mfjoint_cifar100_mlp_3x2000_tune.yaml"

    # --- Rung 4: MF ---
    "configs/tuning/mf_mnist_mlp_2x1000_mf_tune.yaml"
    "configs/tuning/mf_fashion_mnist_mlp_2x1000_mf_tune.yaml"
    "configs/tuning/mf_cifar10_mlp_3x2000_mf_tune.yaml"
    "configs/tuning/mf_cifar100_mlp_3x2000_mf_tune.yaml"
)

NUM_CONFIGS=${#TUNE_CONFIGS[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:-${LADDER_HPO_INDEX}}"
if [ -z "$TASK_ID" ]; then
    echo "Error: set LADDER_HPO_INDEX (1-${NUM_CONFIGS}) or submit as an array." >&2
    exit 1
fi

INDEX=$((TASK_ID - 1))
if [ $INDEX -lt 0 ] || [ $INDEX -ge $NUM_CONFIGS ]; then
    echo "Error: Invalid study index ($TASK_ID) for ${NUM_CONFIGS} studies." >&2
    exit 1
fi

CONFIG_FILE=${TUNE_CONFIGS[$INDEX]}
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '${CONFIG_FILE}' not found." >&2
    exit 1
fi

echo "#####################################################################"
echo "Ladder HPO study $TASK_ID / ${NUM_CONFIGS}"
echo "Config: $CONFIG_FILE"
echo "Host: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

cd "$SLURM_SUBMIT_DIR" || exit 1

module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Python puts the script's own directory on sys.path, not the working directory,
# so without this 'import src' fails.
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export WANDB_MODE=online

srun python scripts/run_optuna_search.py --config "$CONFIG_FILE"
EXIT_CODE=$?

WANDB_LOCAL_DIR="${WANDB_DIR:-$SLURM_SUBMIT_DIR}/wandb"
if compgen -G "$WANDB_LOCAL_DIR/offline-run-*" > /dev/null; then
    wandb sync "$WANDB_LOCAL_DIR"/offline-run-* || echo "WARNING: wandb sync failed"
fi

deactivate

echo "#####################################################################"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Ladder HPO study $TASK_ID finished successfully."
else
    echo "Ladder HPO study $TASK_ID failed (Exit Code: $EXIT_CODE)."
fi
date
echo "#####################################################################"
exit $EXIT_CODE
