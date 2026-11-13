#!/bin/bash
# 5-hour resumable Seq-Global tuning segment. One array task per dataset.
# Submit a chain with submit_mfglobal_segments.sh; each segment uses the same
# Optuna study and --target-trials 50, so it adds only the missing trials.
#
#SBATCH --job-name=BBP_MFGlobalSeg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=05:00:00
#SBATCH -A plgoncotherapy2-gpu-a100
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
SEGMENT_INDEX="${SEGMENT_INDEX:-0}"

INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CONFIG_FILE=${TUNE_CONFIGS[$INDEX]}
STUDY_NAME=${STUDY_NAMES[$INDEX]}

cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
export WANDB_DIR="${SCRATCH:-$HOME}/bbp_wandb"
mkdir -p "$WANDB_DIR"
export BBP_DISABLE_CHECKPOINTS=1

module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

echo "#####################################################################"
echo "Seq-Global tuning segment ${SEGMENT_INDEX}, task ${SLURM_ARRAY_TASK_ID}/2"
echo "Config: $CONFIG_FILE"
echo "Study:  $STUDY_NAME"
echo "Host: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

srun python scripts/run_optuna_search.py \
    --config "$CONFIG_FILE" \
    --output-dir results/optuna \
    --study-name "$STUDY_NAME" \
    --target-trials 50
EXIT_CODE=$?
deactivate

echo "#####################################################################"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Segment ${SEGMENT_INDEX} task ${SLURM_ARRAY_TASK_ID} finished."
else
    echo "Segment ${SEGMENT_INDEX} task ${SLURM_ARRAY_TASK_ID} failed ($EXIT_CODE)."
fi
date
exit $EXIT_CODE
