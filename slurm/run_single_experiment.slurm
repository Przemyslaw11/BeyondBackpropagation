#!/bin/bash
#SBATCH --job-name=BBP_Experiment # Job name - will be prefixed with config later
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks-per-node=1       # Run 1 task (main Python script)
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=16        # Allocate 16 CPUs (proportional for 1 A100)
#SBATCH --mem=128000M             # Allocate 128GB RAM (proportional for 1 A100)
#SBATCH --time=08:00:00           # Max walltime (8 hours, adjust as needed)
#SBATCH -A <your_grant_name>-gpu-a100 # ACCOUNT NAME - REPLACE <your_grant_name>
#SBATCH -p plgrid-gpu-a100        # PARTITION name for A100 nodes
#SBATCH --output=slurm_logs/%x-%j.out # Standard output log file (%x=jobname, %j=jobid)
#SBATCH --error=slurm_logs/%x-%j.err  # Standard error log file

# --- User Configuration ---
# REQUIRED: Set the path to the configuration file you want to run
CONFIG_FILE="configs/ff/fashion_mnist_mlp_4x2000.yaml" # <--- CHANGE THIS LINE

# --- Environment Setup ---
echo "#####################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Executing command in: $(pwd)"
echo "Using config file: $CONFIG_FILE"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"
# Set experiment name based on config file for better logging/output names
# Extract filename without extension: e.g., configs/ff/file.yaml -> file
EXP_NAME=$(basename "$CONFIG_FILE" .yaml)
# Update the SLURM job name dynamically if possible (might not reflect in scheduler immediately, but good for logs)
# Note: This won't change the name in `squeue` after submission, but helps identify logs.
# A better way is to set job name during submission (see run_array.slurm example).
echo "Setting log identifier to: BBP_$EXP_NAME"

# Navigate to the submission directory (where the code is)
cd $SLURM_SUBMIT_DIR || exit 1

# Load necessary modules
echo "Loading modules..."
module purge # Start with a clean environment
module load plgrid/tools/python/3.10 # Use the SAME version used for installation
module load plgrid/tools/cuda/11.8   # Use the SAME version used for installation
# module load plgrid/tools/gcc/X.Y   # Load if needed by any runtime dependency

echo "Modules loaded:"
module list

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Set WANDB API Key if needed (alternative to setting in .bashrc)
# export WANDB_API_KEY="YOUR_API_KEY"

# Check PyTorch and CUDA availability
echo "Checking PyTorch CUDA..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}')" || { echo "PyTorch check failed"; exit 1; }

# --- Run the Experiment ---
echo "Running the experiment script..."
# Use srun for better resource management within the allocation
srun python scripts/run_experiment.py --config "$CONFIG_FILE"

EXIT_CODE=$? # Capture the exit code of the python script

# --- Cleanup ---
echo "Deactivating virtual environment..."
deactivate

echo "#####################################################################"
echo "Job finished with exit code $EXIT_CODE"
date
echo "#####################################################################"

exit $EXIT_CODE