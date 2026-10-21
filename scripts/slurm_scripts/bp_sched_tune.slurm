#!/bin/bash
# W1 control (a) tuning: 50-trial lr/wd study per dataset for BP-sched.
# The objective honours training.scheduler, so the cosine schedule is identical
# between tuning and the confirmatory 100-epoch arm. Studies are resumable via
# --target-trials 50.
#
#SBATCH --job-name=BBP_BPSchedTune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=08:00:00
#SBATCH -A plglscclass26-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=1-2

TUNE_CONFIGS=(
    "configs/tuning/bp_sched_cifar10_mlp_3x2000_tune.yaml"
    "configs/tuning/bp_sched_cifar100_mlp_3x2000_tune.yaml"
)
STUDY_NAMES=(
    "ladder_bp_sched_cifar10_mlp_3x2000_tune"
    "ladder_bp_sched_cifar100_mlp_3x2000_tune"
)

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
echo "BP-sched tuning task ${SLURM_ARRAY_TASK_ID}/2"
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
    echo "BP-sched tuning task ${SLURM_ARRAY_TASK_ID} finished."
else
    echo "BP-sched tuning task ${SLURM_ARRAY_TASK_ID} failed ($EXIT_CODE)."
fi
date
exit $EXIT_CODE
