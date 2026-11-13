#!/bin/bash
# H-W1cap Step 1 (PREREG-phase2.md §3): exploratory pilot, n = 5, seeds 42-46,
# CIFAR-100, MF-Seq with max_epochs 300 instead of 100. Nothing here enters a
# decisive table; the trigger is: mean accuracy gain >= +0.5 pp vs the G1 MF-Seq
# C100 baseline OR mean W1_M1 epochs > 130 -> run the full n=20 cap-300 arm.
#
# Hygiene mirrors g1_repair.slurm: per-seed run log via generated config,
# forced scratch wandb, HOME-quota precheck, no checkpoints.
#
#SBATCH --job-name=BBP_W1CapPilot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=08:00:00
#SBATCH -A plgoncotherapy2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=1-5

CONFIG_FILE="configs/mf_cap300/cifar100_mlp_3x2000.yaml"
SEEDS=(42 43 44 45 46)

INDEX=$((SLURM_ARRAY_TASK_ID - 1))
if [ $INDEX -lt 0 ] || [ $INDEX -ge ${#SEEDS[@]} ]; then
    echo "Error: Invalid SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID)." >&2
    exit 1
fi
export EXPERIMENT_SEED=${SEEDS[$INDEX]}

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

EXP_NAME=$(basename "$CONFIG_FILE" .yaml)
RUN_LOG="results/${EXP_NAME}/${EXP_NAME}_seed${EXPERIMENT_SEED}_run.log"
TMP_CONFIG="/tmp/w1cap_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
python3 - "$CONFIG_FILE" "$TMP_CONFIG" "$RUN_LOG" <<'EOF'
import sys, yaml
src, dst, log_file = sys.argv[1], sys.argv[2], sys.argv[3]
config = yaml.safe_load(open(src))
config.setdefault("logging", {})["log_file"] = log_file
yaml.safe_dump(config, open(dst, "w"), sort_keys=False)
EOF

echo "#####################################################################"
echo "H-W1cap pilot $SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}"
echo "Config: $CONFIG_FILE   Seed: $EXPERIMENT_SEED"
echo "Run log: $RUN_LOG"
echo "Host: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

srun python scripts/run_experiment.py --config "$TMP_CONFIG"
EXIT_CODE=$?
rm -f "$TMP_CONFIG"

if [ $EXIT_CODE -eq 0 ]; then
    echo "H-W1cap pilot $SLURM_ARRAY_TASK_ID finished successfully."
else
    echo "H-W1cap pilot $SLURM_ARRAY_TASK_ID failed (Exit Code: $EXIT_CODE)."
fi
date
exit $EXIT_CODE
