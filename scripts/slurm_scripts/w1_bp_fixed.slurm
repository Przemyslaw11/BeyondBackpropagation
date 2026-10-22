#!/bin/bash
# W1 control (H-W1d, amended 2026-10-22): BP-fixed.
#
# BP-long was retired because BP never reaches the 100-epoch cap, so "300
# epochs" duplicated core BP. BP-fixed instead spends exactly the measured
# MF-Seq pass budget: round(mean MF-Seq G1 passes) = 211 (CIFAR-10) and 191
# (CIFAR-100). It uses the BP-sched hyperparameters and cosine schedule with no
# early stopping, and scores the final weights. It is a same-pass accuracy
# control, not a new primary arm.
#
# The lr/wd live in configs/bp_sched/* and are gated on the BP-sched HPO branch
# (deadline 2026-12-02 23:59 CET). This launcher refuses to run while that
# config still contains PLACEHOLDER, so BP-fixed and BP-sched always share the
# same frozen hyperparameters. 2 datasets x 20 seeds = 40 tasks.
#
#SBATCH --job-name=BBP_W1Fixed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=01:00:00
#SBATCH -A plgoncotherapy2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=1-40

CONFIG_FILES=(
    "configs/bp_sched/cifar10_mlp_3x2000.yaml"
    "configs/bp_sched/cifar100_mlp_3x2000.yaml"
)
EXPERIMENT_NAMES=(
    "bp_fixed_cifar10_mlp_3x2000"
    "bp_fixed_cifar100_mlp_3x2000"
)
FIXED_EPOCHS=(211 191)
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
CONFIG_INDEX=$((INDEX / NUM_SEEDS))
SEED_INDEX=$((INDEX % NUM_SEEDS))
CONFIG_FILE=${CONFIG_FILES[$CONFIG_INDEX]}
EXP_NAME=${EXPERIMENT_NAMES[$CONFIG_INDEX]}
FIXED_EPOCH=${FIXED_EPOCHS[$CONFIG_INDEX]}
export EXPERIMENT_SEED=${SEEDS[$SEED_INDEX]}

cd "$SLURM_SUBMIT_DIR" || exit 1

if grep -q "PLACEHOLDER" "$CONFIG_FILE"; then
    echo "!! $CONFIG_FILE still contains PLACEHOLDER; freeze the BP-sched branch first. !!"
    exit 1
fi

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

USED_MB=$(quota -s | awk '/net\/people/ {getline; gsub(/M/,"",$1); print $1}')
if [ -n "$USED_MB" ] && [ "$USED_MB" -ge 9500 ]; then
    echo "HOME quota at ${USED_MB}M >= 9500M; refusing to run."
    exit 1
fi

export BBP_DISABLE_CHECKPOINTS=1
export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH/bbp_wandb"

RUN_LOG="results/${EXP_NAME}/${EXP_NAME}_seed${EXPERIMENT_SEED}_run.log"
TMP_CONFIG="/tmp/w1fixed_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
python3 - "$CONFIG_FILE" "$TMP_CONFIG" "$RUN_LOG" "$EXP_NAME" "$FIXED_EPOCH" <<'EOF'
import sys, yaml
src, dst, log_file, exp_name, epochs = sys.argv[1:6]
config = yaml.safe_load(open(src))
config["experiment_name"] = exp_name
config.setdefault("training", {})["scheduler"] = "CosineAnnealingLR"
config["training"].setdefault("scheduler_params", {})
config.setdefault("early_stopping", {})["enabled"] = False
config["early_stopping"]["max_epochs"] = int(epochs)
config.setdefault("logging", {})["log_file"] = log_file
config["checkpointing"] = {"checkpoint_dir": None}
yaml.safe_dump(config, open(dst, "w"), sort_keys=False)
EOF

echo "#####################################################################"
echo "W1 BP-fixed run $SLURM_ARRAY_TASK_ID / ${NUM_JOBS}"
echo "Base config (${CONFIG_INDEX}): $CONFIG_FILE"
echo "Fixed epochs: $FIXED_EPOCH (no early stopping)"
echo "Seed: $EXPERIMENT_SEED"
echo "Run log: $RUN_LOG"
echo "Host: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

srun python scripts/run_experiment.py --config "$TMP_CONFIG"
EXIT_CODE=$?
rm -f "$TMP_CONFIG"

if [ $EXIT_CODE -eq 0 ]; then
    echo "W1 BP-fixed run $SLURM_ARRAY_TASK_ID finished successfully."
else
    echo "W1 BP-fixed run $SLURM_ARRAY_TASK_ID failed (Exit Code: $EXIT_CODE)."
fi
date
exit $EXIT_CODE
