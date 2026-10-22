#!/bin/bash
# W2 diagnostic: one 1 MB NCCL point-to-point microbenchmark on two A100s.
# Short job (seconds of GPU time), not a W1 array and not a confirmatory W2 run.
# Not submitted by w1_submit.sh. Requires author approval before submission.
#
#SBATCH --job-name=BBP_W2NCCL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=00:10:00
#SBATCH -A plgoncotherapy2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

module purge
module load Python/3.10.4
module load CUDA/12.4.0
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

export WANDB_MODE=offline
mkdir -p jpdc_data/g1/w2_nccl

echo "Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

torchrun --standalone --nproc_per_node=2 \
    scripts/jpdc/nccl_p2p_benchmark.py \
    --tensor-bytes 1048576 \
    --iterations 200 \
    --warmup 20 \
    --output jpdc_data/g1/w2_nccl/nccl_p2p_1mb.json

echo "Wrote jpdc_data/g1/w2_nccl/nccl_p2p_1mb.json"
