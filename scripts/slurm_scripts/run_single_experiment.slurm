#!/bin/bash
#SBATCH --job-name=BBP_Experiment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=8:00:00
#SBATCH -A plgoncotherapy-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

CONFIG_FILE="$1"

if [ -z "$CONFIG_FILE" ]; then echo "Error: No configuration file path provided." >&2; exit 1; fi
if [ ! -f "$CONFIG_FILE" ]; then echo "Error: Configuration file not found at: $CONFIG_FILE" >&2; exit 1; fi

# --- Environment Setup ---
echo "#####################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Executing command in: $(pwd)"
echo "Using config file: $CONFIG_FILE"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

EXP_NAME=$(basename "$CONFIG_FILE" .yaml)
echo "Derived Experiment Name: $EXP_NAME"

cd $SLURM_SUBMIT_DIR || exit 1

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

echo "Loading modules..."
module purge
module load Python/3.10.4
module load CUDA/12.4.0

echo "Modules loaded:"
module list

echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate venv" >&2; exit 1; }

echo "Checking PyTorch CUDA..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}')" || { echo "PyTorch check failed" >&2; exit 1; }

echo "Running the experiment script..."
srun python scripts/run_experiment.py --config "$CONFIG_FILE"

EXIT_CODE=$?

# --- Cleanup ---
echo "Deactivating virtual environment..."
deactivate

echo "#####################################################################"
echo "Job finished with exit code $EXIT_CODE at $(date)"
echo "#####################################################################"

exit $EXIT_CODE