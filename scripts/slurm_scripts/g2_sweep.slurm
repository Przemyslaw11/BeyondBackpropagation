#!/bin/bash
# G2 E-S/E-D saturation sweep (PREREG-G2-DRAFT.md): one repeat per task, all
# (batch x width x arm) units of that repeat on one GPU in shuffled order.
# Tasks 1-20 = {cifar10, cifar100} x repeats 0-9 (author, 2026-11-11). Pilot: submit with --array=1
# and G2_EXTRA="--measure-sec 10 --idle-sec 60" (exploratory, never decisive).
# Account is passed by g2_submit.sh; this file carries none on purpose.
#SBATCH --job-name=BBP_G2Sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=02:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=1-20

DATASETS=(cifar10 cifar100)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
DATASET=${DATASETS[$((INDEX / 10))]}
REPEAT=$((INDEX % 10))

cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

OUT="results/g2_sweep${G2_TAG:+_$G2_TAG}/${DATASET}_r${REPEAT}_${SLURM_ARRAY_JOB_ID}.json"
echo "G2 sweep task $SLURM_ARRAY_TASK_ID: dataset=$DATASET repeat=$REPEAT out=$OUT"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}   Start: $(date -Is)   Commit: $(git rev-parse HEAD)"
echo "GPU UUID: $(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null | paste -sd, -)"

# shellcheck disable=SC2086
srun python scripts/jpdc/g2_sweep.py --dataset "$DATASET" --repeat "$REPEAT" --out "$OUT" ${G2_EXTRA:-}
EXIT_CODE=$?
echo "Exit: $EXIT_CODE   End: $(date -Is)"
exit $EXIT_CODE
