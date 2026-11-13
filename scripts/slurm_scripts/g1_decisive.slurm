#!/bin/bash
# G1 decisive runs (PREREG.md §4, §7): n = 20, seeds 42-61, CIFAR-10 + CIFAR-100,
# all four arms rerun contemporaneously on one commit (drift control).
#
# 8 configs x 20 seeds = 160 runs. MF-Simul uses the ladder-tuned hyperparameters
# frozen into configs/mf_simul/ (see PREREG.md §4).
#
# Submit only after PREREG.md §10 records the freeze (PREREG + analysis-script
# hashes) and no further changes to the four arm configs are made.
#
#SBATCH --job-name=BBP_G1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=04:00:00
#SBATCH -A plglscclass26-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=1-160

CONFIG_FILES=(
    "configs/bp_baselines/cifar10_mlp_3x2000_bp.yaml"
    "configs/mf_joint/cifar10_mlp_3x2000.yaml"
    "configs/mf_simul/cifar10_mlp_3x2000.yaml"
    "configs/mf/cifar10_mlp_3x2000.yaml"
    "configs/bp_baselines/cifar100_mlp_3x2000_bp.yaml"
    "configs/mf_joint/cifar100_mlp_3x2000.yaml"
    "configs/mf_simul/cifar100_mlp_3x2000.yaml"
    "configs/mf/cifar100_mlp_3x2000.yaml"
)

SEEDS=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61)

NUM_CONFIGS=${#CONFIG_FILES[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_JOBS=$((NUM_CONFIGS * NUM_SEEDS))
if [ "${SLURM_ARRAY_TASK_MAX}" -ne "${NUM_JOBS}" ]; then
    echo "!! WARNING: --array range (1-${SLURM_ARRAY_TASK_MAX}) does not match ${NUM_JOBS} runs. !!"
fi

INDEX=$((SLURM_ARRAY_TASK_ID - 1))
if [ $INDEX -lt 0 ] || [ $INDEX -ge $NUM_JOBS ]; then
    echo "Error: Invalid SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID)." >&2
    exit 1
fi

CONFIG_INDEX=$((INDEX % NUM_CONFIGS))
SEED_INDEX=$((INDEX / NUM_CONFIGS))
CONFIG_FILE=${CONFIG_FILES[$CONFIG_INDEX]}
export EXPERIMENT_SEED=${SEEDS[$SEED_INDEX]}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '${CONFIG_FILE}' not found for task ID $SLURM_ARRAY_TASK_ID." >&2
    exit 1
fi

echo "#####################################################################"
echo "G1 run $SLURM_ARRAY_TASK_ID / ${NUM_JOBS}"
echo "Config (${CONFIG_INDEX}): $CONFIG_FILE"
echo "Seed: $EXPERIMENT_SEED"
echo "Host: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

cd "$SLURM_SUBMIT_DIR" || exit 1

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Canonical protocol (PREREG.md §3): final weights, single test evaluation,
# no checkpoint restore. HOME is quota-limited (10 GiB); BP checkpoints fill it.
export BBP_DISABLE_CHECKPOINTS=1

export WANDB_MODE=offline
# Keep wandb artifacts off $HOME: offline runs grow ~100 MB per task.
export WANDB_DIR="${WANDB_DIR:-${SCRATCH:-$SLURM_SUBMIT_DIR}/bbp_wandb}"

srun python scripts/run_experiment.py --config "$CONFIG_FILE"
EXIT_CODE=$?

WANDB_LOCAL_DIR="${WANDB_DIR:-$SLURM_SUBMIT_DIR}/wandb"
if compgen -G "$WANDB_LOCAL_DIR/offline-run-*" > /dev/null; then
    wandb sync "$WANDB_LOCAL_DIR"/offline-run-* || echo "WARNING: wandb sync failed"
fi

deactivate

echo "#####################################################################"
if [ $EXIT_CODE -eq 0 ]; then
    echo "G1 run $SLURM_ARRAY_TASK_ID finished successfully."
else
    echo "G1 run $SLURM_ARRAY_TASK_ID failed (Exit Code: $EXIT_CODE)."
fi
date
echo "#####################################################################"
exit $EXIT_CODE
