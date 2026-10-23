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
#SBATCH -A plgoncotherapy2-gpu-a100
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
# Optuna SQLite store off HOME (quota + NFS locking). One writer per study:
# tasks 1 and 2 own different databases, and continuations are dependency-chained.
OPTUNA_DIR="${SCRATCH:-$HOME}/bbp_optuna"
mkdir -p "$OPTUNA_DIR"

# HOME quota guard: a full HOME kills the run and wastes a GPU allocation.
USED_MB=$(quota -s | awk '/net\/people/ {getline; gsub(/M/,"",$1); print $1}')
if [ -z "${USED_MB:-}" ] || [ "${USED_MB}" -ge 9500 ]; then
    echo "HOME quota at ${USED_MB:-unknown}M >= 9500M; refusing to run."
    exit 1
fi

module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

echo "#####################################################################"
echo "BP-sched tuning task ${SLURM_ARRAY_TASK_ID}/2"
echo "Config: $CONFIG_FILE"
echo "Study:  $STUDY_NAME"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}   Start: $(date -Is)"
echo "GPU UUID: $(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null | paste -sd, -)"
echo "Host: $(hostname)   GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

DB_PATH="$OPTUNA_DIR/$STUDY_NAME.db"
RESUME_OUT=$(python scripts/jpdc/optuna_resume.py \
    --db "$DB_PATH" --study "$STUDY_NAME" --target 50 --shell) || {
    echo "Resume bookkeeping failed; refusing to start a trial."
    exit 1
}
eval "$RESUME_OUT"
echo "Resume check: complete=$COMPLETE remaining=$REMAINING failed_stale=$FAILED_STALE"
if [ "${REMAINING}" -le 0 ]; then
    echo "Budget already reached (COMPLETE=$COMPLETE >= 50); exiting without GPU work."
    exit 0
fi

srun python scripts/run_optuna_search.py \
    --config "$CONFIG_FILE" \
    --output-dir "$OPTUNA_DIR" \
    --study-name "$STUDY_NAME" \
    --n-trials "$REMAINING"
EXIT_CODE=$?
deactivate

# Rollback-journal integrity check and off-scratch copy, after every segment.
python - "$DB_PATH" <<'PY'
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1])
result = con.execute("PRAGMA integrity_check").fetchone()[0]
print(f"PRAGMA integrity_check: {result}")
raise SystemExit(0 if result == "ok" else 1)
PY
INTEGRITY_CODE=$?

DB_ARCHIVE="/net/pr2/projects/plgrid/plgglscclass/plgspyra_bbp_archive/w1_optuna"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
CONTAINER="${STUDY_NAME}_${SLURM_JOB_ID}_${STAMP}.db"
if mkdir -p "$DB_ARCHIVE" && cp -p "$DB_PATH" "$DB_ARCHIVE/$CONTAINER"; then
    ( cd "$DB_ARCHIVE" && sha256sum "$CONTAINER" > "$CONTAINER.sha256" )
    echo "DB archived: $DB_ARCHIVE/$CONTAINER"
else
    echo "DB archive failed; the scratch DB is intact."
    if [ "$EXIT_CODE" -eq 0 ]; then EXIT_CODE=1; fi
fi

if [ "$INTEGRITY_CODE" -ne 0 ] && [ "$EXIT_CODE" -eq 0 ]; then
    EXIT_CODE=$INTEGRITY_CODE
fi

echo "#####################################################################"
if [ $EXIT_CODE -eq 0 ]; then
    echo "BP-sched tuning task ${SLURM_ARRAY_TASK_ID} finished."
else
    echo "BP-sched tuning task ${SLURM_ARRAY_TASK_ID} failed ($EXIT_CODE)."
fi
date
exit $EXIT_CODE
