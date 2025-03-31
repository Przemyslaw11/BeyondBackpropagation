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

# --- Configuration via Command Line Argument ---
# The path to the configuration file should be passed as the first argument to sbatch
# Example: sbatch scripts/slurm_scripts/run_single_experiment.slurm configs/ff/fashion_mnist_mlp_4x2000.yaml
CONFIG_FILE="$1"

# Check if CONFIG_FILE argument is provided
if [ -z "$CONFIG_FILE" ]; then
  echo "Error: No configuration file path provided as the first argument."
  echo "Usage: sbatch $0 <path_to_config.yaml>"
  exit 1
fi

# Check if CONFIG_FILE exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at: $CONFIG_FILE"
    exit 1
fi

# --- Environment Setup ---
echo "#####################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Executing command in: $(pwd)"
echo "Using config file: $CONFIG_FILE"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"
# Set experiment name based on config file for potential use (e.g., logging)
# Extract filename without extension: e.g., configs/ff/file.yaml -> file
EXP_NAME=$(basename "$CONFIG_FILE" .yaml)
echo "Derived Experiment Name: $EXP_NAME"
# Note: To set the SLURM job name dynamically, use the --job-name flag during sbatch submission, e.g.:
# sbatch --job-name="BBP_${EXP_NAME}" scripts/slurm_scripts/run_single_experiment.slurm "$CONFIG_FILE"

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
