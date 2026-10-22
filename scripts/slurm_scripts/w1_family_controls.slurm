#!/bin/bash
# W1 frozen controls (PREREG-phase2.md §5/§9): BP-bestval only,
# 1 control x 2 datasets x 20 seeds = 40 tasks. These parameters are frozen
# independently of Seq-Global and BP-sched tuning.
#
#   BP-bestval       in-memory best-val-accuracy restore, no disk checkpoints.
#
# H-W1d is now BP-fixed, which shares the BP-sched hyperparameters and is
# launched by w1_bp_fixed.slurm only after that HPO branch is frozen. BP-long
# was retired on 2026-10-22 because BP never reached its 100-epoch cap.
# Protocol B (MF-Seq/nom0) is in w1_controls_protocol_b.slurm because its budget
# is frozen from the actual core-BP mean after core BP finishes, not now.
#
#SBATCH --job-name=BBP_W1Ctl
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
#SBATCH --array=1-40%20

CONFIG_FILES=(
    "configs/bp_bestval/cifar10_mlp_3x2000.yaml"
    "configs/bp_bestval/cifar100_mlp_3x2000.yaml"
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
# Seed-major order (2026-10-22 amendment): both datasets of one seed are
# adjacent in index order, so paired runs start together under the array
# throttle. Mapping only; no config, seed, arm or task-count change.
SEED_INDEX=$((INDEX / NUM_CONFIGS))
CONFIG_INDEX=$((INDEX % NUM_CONFIGS))
CONFIG_FILE=${CONFIG_FILES[$CONFIG_INDEX]}
export EXPERIMENT_SEED=${SEEDS[$SEED_INDEX]}

cd "$SLURM_SUBMIT_DIR" || exit 1

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

EXP_NAME=$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['experiment_name'])" "$CONFIG_FILE")
RUN_LOG="results/${EXP_NAME}/${EXP_NAME}_seed${EXPERIMENT_SEED}_run.log"
TMP_CONFIG="/tmp/w1fctl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
python3 - "$CONFIG_FILE" "$TMP_CONFIG" "$RUN_LOG" <<'EOF'
import sys, yaml
src, dst, log_file = sys.argv[1], sys.argv[2], sys.argv[3]
config = yaml.safe_load(open(src))
config.setdefault("logging", {})["log_file"] = log_file
yaml.safe_dump(config, open(dst, "w"), sort_keys=False)
EOF

echo "#####################################################################"
echo "W1 frozen-control run $SLURM_ARRAY_TASK_ID / ${NUM_JOBS}"
echo "Config (${CONFIG_INDEX}): $CONFIG_FILE"
echo "Seed: $EXPERIMENT_SEED"
echo "Run log: $RUN_LOG"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}   Start: $(date -Is)"
echo "Host: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

srun python scripts/run_experiment.py --config "$TMP_CONFIG"
EXIT_CODE=$?
rm -f "$TMP_CONFIG"

if [ $EXIT_CODE -eq 0 ]; then
    echo "W1 frozen-control run $SLURM_ARRAY_TASK_ID finished successfully."
else
    echo "W1 frozen-control run $SLURM_ARRAY_TASK_ID failed (Exit Code: $EXIT_CODE)."
fi
date
exit $EXIT_CODE
