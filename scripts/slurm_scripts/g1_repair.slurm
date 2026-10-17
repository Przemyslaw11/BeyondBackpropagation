#!/bin/bash
# G1 artifact repair (12 runs): rerun the decisive (arm, seed) pairs whose NVML
# CSV was lost in the 2026-10-17 disk-quota window. Their run JSONs survived but
# the energy trace did not, so `analyze_g1`'s old glob fallback silently used
# September PPAM-era traces for those seeds. Same commit, same configs, same
# protocol; only the artifacts are regenerated.
#
# Extra hardening over g1_decisive.slurm:
# - WANDB_DIR is forced onto scratch (not ${WANDB_DIR:-...}: an inherited value
#   silently defeated the previous default).
# - the run log goes to a per-seed file via a generated config copy, because the
#   shared append-only log lost lines/headers under the Oct-17 conditions
#   (verified in slurm_logs: e.g. MF-Simul seed51/52 rerun blocks missing).
# - HOME quota checked before training starts.
#
#SBATCH --job-name=BBP_G1Fix
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
#SBATCH --array=1-12

CONFIG_FILES=(
    "configs/mf_joint/cifar10_mlp_3x2000.yaml"   # 1
    "configs/mf_joint/cifar10_mlp_3x2000.yaml"   # 2
    "configs/mf_joint/cifar10_mlp_3x2000.yaml"   # 3
    "configs/mf_joint/cifar10_mlp_3x2000.yaml"   # 4
    "configs/mf_joint/cifar100_mlp_3x2000.yaml"  # 5
    "configs/mf_joint/cifar100_mlp_3x2000.yaml"  # 6
    "configs/mf_joint/cifar100_mlp_3x2000.yaml"  # 7
    "configs/mf_joint/cifar100_mlp_3x2000.yaml"  # 8
    "configs/mf/cifar10_mlp_3x2000.yaml"         # 9
    "configs/mf/cifar100_mlp_3x2000.yaml"        # 10
    "configs/mf/cifar100_mlp_3x2000.yaml"        # 11
    "configs/mf/cifar100_mlp_3x2000.yaml"        # 12
)
SEEDS=(50 51 52 55 50 51 52 54 47 46 47 48)

INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CONFIG_FILE=${CONFIG_FILES[$INDEX]}
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

# Per-seed run log: generate a config copy with logging.log_file set.
EXP_NAME=$(basename "$CONFIG_FILE" .yaml)
RUN_LOG="results/${EXP_NAME}/${EXP_NAME}_seed${EXPERIMENT_SEED}_run.log"
TMP_CONFIG="/tmp/g1fix_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
python3 - "$CONFIG_FILE" "$TMP_CONFIG" "$RUN_LOG" <<'EOF'
import sys, yaml
src, dst, log_file = sys.argv[1], sys.argv[2], sys.argv[3]
config = yaml.safe_load(open(src))
config.setdefault("logging", {})["log_file"] = log_file
yaml.safe_dump(config, open(dst, "w"), sort_keys=False)
EOF

echo "#####################################################################"
echo "G1 repair $SLURM_ARRAY_TASK_ID / 12"
echo "Config: $CONFIG_FILE   Seed: $EXPERIMENT_SEED"
echo "Run log: $RUN_LOG"
echo "Host: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

srun python scripts/run_experiment.py --config "$TMP_CONFIG"
EXIT_CODE=$?
rm -f "$TMP_CONFIG"

if [ $EXIT_CODE -eq 0 ]; then
    echo "G1 repair $SLURM_ARRAY_TASK_ID finished successfully."
else
    echo "G1 repair $SLURM_ARRAY_TASK_ID failed (Exit Code: $EXIT_CODE)."
fi
date
exit $EXIT_CODE
