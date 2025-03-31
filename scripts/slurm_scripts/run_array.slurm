#!/bin/bash
#SBATCH --job-name=BBP_Array_%a # Job name includes array task ID (%a)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=08:00:00      # Time limit PER TASK
#SBATCH -A <your_grant_name>-gpu-a100 # ACCOUNT NAME - REPLACE <your_grant_name>
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%A_%a.out # Output file (%A=jobid, %a=taskid)
#SBATCH --error=slurm_logs/%x-%A_%a.err  # Error file
#SBATCH --array=1-6 # ARRAY: Run for task IDs 1 through 6 (adjust range)

# --- Configuration ---
# Define an array of configuration files
# List ALL config files you want to run in this array job
CONFIG_FILES=(
    "configs/ff/fashion_mnist_mlp_4x2000.yaml"
    "configs/bp_baselines/fashion_mnist_mlp_4x2000_bp.yaml" # Run BP after tuning!
    "configs/cafo/fashion_mnist_cnn_3block.yaml"
    "configs/bp_baselines/fashion_mnist_cnn_3block_bp.yaml" # Run BP after tuning!
    "configs/cafo/cifar10_cnn_3block.yaml"
    "configs/bp_baselines/cifar10_cnn_3block_bp.yaml" # Run BP after tuning!
    # Add MF configs and other CIFAR configs here
    # "configs/mf/fashion_mnist_mlp_2x1000.yaml"
    # "configs/bp_baselines/fashion_mnist_mlp_2x1000_bp.yaml"
    # ... etc.
)

# Select the config file based on the SLURM array task ID
# SLURM_ARRAY_TASK_ID starts from 1, Bash array indices start from 0
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
CONFIG_FILE=${CONFIG_FILES[$INDEX]}

# Check if config file selection is valid
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file found for array task ID $SLURM_ARRAY_TASK_ID (Index $INDEX)"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi


# --- Environment Setup ---
echo "#####################################################################"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Executing command in: $(pwd)"
echo "Using config file: $CONFIG_FILE"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

cd $SLURM_SUBMIT_DIR || exit 1

echo "Loading modules..."
module purge
module load plgrid/tools/python/3.10 # Use the SAME version
module load plgrid/tools/cuda/11.8   # Use the SAME version

echo "Modules loaded:"
module list

echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

echo "Checking PyTorch CUDA..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" || { echo "PyTorch check failed"; exit 1; }

# --- Run the Experiment ---
echo "Running the experiment script for task $SLURM_ARRAY_TASK_ID..."
srun python scripts/run_experiment.py --config "$CONFIG_FILE"

EXIT_CODE=$?

# --- Cleanup ---
echo "Deactivating virtual environment..."
deactivate

echo "#####################################################################"
echo "Array Task $SLURM_ARRAY_TASK_ID finished with exit code $EXIT_CODE"
date
echo "#####################################################################"

exit $EXIT_CODE