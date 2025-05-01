#!/bin/bash
#SBATCH --job-name=BBP_Experiment # Job name - will be prefixed with config later
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks-per-node=1       # Run 1 task (main Python script)
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=16        # Allocate 16 CPUs (proportional for 1 A100)
#SBATCH --mem=128000M             # Allocate 128GB RAM (proportional for 1 A100)
#SBATCH --time=2:00:00           # Max walltime (8 hours, adjust as needed)
#SBATCH -A plgoncotherapy-gpu-a100 # !!! YOUR GRANT ACCOUNT !!!
#SBATCH -p plgrid-gpu-a100       # PARTITION name for A100 nodes
#SBATCH --output=slurm_logs/%x-%j.out # Standard output log file (%x=jobname, %j=jobid)
#SBATCH --error=slurm_logs/%x-%j.err  # Standard error log file

# --- Configuration via Command Line Argument ---
# Example: sbatch scripts/slurm_scripts/run_single_experiment.slurm configs/ff/fashion_mnist_mlp_4x2000.yaml
CONFIG_FILE="$1"

if [ -z "$CONFIG_FILE" ]; then echo "Error: No configuration file path provided." >&2; exit 1; fi
if [ ! -f "$CONFIG_FILE" ]; then echo "Error: Configuration file not found at: $CONFIG_FILE" >&2; exit 1; fi

# --- Environment Setup ---
echo "#####################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Executing command in: $(pwd)" # Should be project root
echo "Using config file: $CONFIG_FILE"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

EXP_NAME=$(basename "$CONFIG_FILE" .yaml)
echo "Derived Experiment Name: $EXP_NAME"
# Dynamic job name via sbatch:
# sbatch --job-name="BBP_${EXP_NAME}" scripts/slurm_scripts/run_single_experiment.slurm "$CONFIG_FILE"

cd $SLURM_SUBMIT_DIR || exit 1

echo "Loading modules..."
module purge # Start with a clean environment
module load Python/3.10.4   # Version used for setup
module load CUDA/12.4.0     # Version compatible with PyTorch build (cu121)

echo "Modules loaded:"
module list

echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate venv" >&2; exit 1; }

# Set WANDB API Key if not in environment (best practice: use env vars)
# export WANDB_API_KEY="YOUR_API_KEY"

echo "Checking PyTorch CUDA..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}')" || { echo "PyTorch check failed" >&2; exit 1; }

# --- Run the Experiment ---
echo "Running the experiment script..."
# Use srun for Slurm integration
srun python scripts/run_experiment.py --config "$CONFIG_FILE"

EXIT_CODE=$?

# --- Cleanup ---
echo "Deactivating virtual environment..."
deactivate

echo "#####################################################################"
echo "Job finished with exit code $EXIT_CODE at $(date)"
echo "#####################################################################"

exit $EXIT_CODE