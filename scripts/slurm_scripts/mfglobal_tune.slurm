#!/bin/bash
# W1 (PREREG-phase2.md §5, option b): tune Seq-Global per dataset with the same
# 50-trial ladder protocol as MF-Simul, then freeze the result in configs/mf_global/*.
#
# Two studies: CIFAR-10 and CIFAR-100. Studies are resumable (load_if_exists),
# so a requeue backfills instead of restarting.
#
#SBATCH --job-name=BBP_MFGlobalTune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=12:00:00
#SBATCH -A plglscclass26-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=1-2

TUNE_CONFIGS=(
    "configs/tuning/mfglobal_cifar10_mlp_3x2000_tune.yaml"
    "configs/tuning/mfglobal_cifar100_mlp_3x2000_tune.yaml"
)

STUDY_NAMES=(
    "ladder_mfglobal_cifar10_mlp_3x2000_tune"
    "ladder_mfglobal_cifar100_mlp_3x2000_tune"
)

NUM_CONFIGS=${#TUNE_CONFIGS[@]}
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
if [ $INDEX -lt 0 ] || [ $INDEX -ge $NUM_CONFIGS ]; then
    echo "Error: Invalid SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) for ${NUM_CONFIGS} studies." >&2
    exit 1
fi

CONFIG_FILE=${TUNE_CONFIGS[$INDEX]}
STUDY_NAME=${STUDY_NAMES[$INDEX]}
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '${CONFIG_FILE}' not found." >&2
    exit 1
fi

echo "#####################################################################"
echo "Seq-Global tuning task $SLURM_ARRAY_TASK_ID / ${NUM_CONFIGS}"
echo "Config: $CONFIG_FILE"
echo "Study: $STUDY_NAME"
echo "Host: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

cd "$SLURM_SUBMIT_DIR" || exit 1

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export WANDB_DIR="${SCRATCH:-$HOME}/bbp_wandb"
mkdir -p "$WANDB_DIR"
export BBP_DISABLE_CHECKPOINTS=1

module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

srun python scripts/run_optuna_search.py \
    --config "$CONFIG_FILE" \
    --output-dir results/optuna \
    --study-name "$STUDY_NAME"
EXIT_CODE=$?

deactivate

echo "#####################################################################"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Seq-Global tuning task $SLURM_ARRAY_TASK_ID finished successfully."
    echo "Best params: results/optuna/${STUDY_NAME}_best_params.yaml"
else
    echo "Seq-Global tuning task $SLURM_ARRAY_TASK_ID failed (Exit Code: $EXIT_CODE)."
fi
date
echo "#####################################################################"
exit $EXIT_CODE
