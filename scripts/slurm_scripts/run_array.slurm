#!/bin/bash
#SBATCH --job-name=BBP_Array_%a # Job name includes array task ID (%a)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=08:00:00      # Time limit PER TASK
#SBATCH -A plgoncotherapy-gpu-a100 # !!! YOUR GRANT ACCOUNT !!!
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%A_%a.out # Output file (%A=jobid, %a=taskid)
#SBATCH --error=slurm_logs/%x-%A_%a.err  # Error file
#SBATCH --array=1-6 # ARRAY: Run for task IDs 1 through N (adjust N based on CONFIG_FILES length)

# --- Configuration ---
# Define an array of configuration files (relative to project root)
CONFIG_FILES=(
    "configs/ff/fashion_mnist_mlp_4x2000.yaml"
    "configs/bp_baselines/fashion_mnist_mlp_4x2000_bp.yaml" # Make sure this is tuned!
    "configs/cafo/fashion_mnist_cnn_3block.yaml"
    "configs/bp_baselines/fashion_mnist_cnn_3block_bp.yaml" # Make sure this is tuned!
    "configs/cafo/cifar10_cnn_3block.yaml"
    "configs/bp_baselines/cifar10_cnn_3block_bp.yaml" # Make sure this is tuned!
    # Add the remaining configs here:
    # "configs/cafo/cifar100_cnn_3block.yaml"
    # "configs/bp_baselines/cifar100_cnn_3block_bp.yaml"
    # "configs/mf/fashion_mnist_mlp_2x1000.yaml"
    # "configs/bp_baselines/fashion_mnist_mlp_2x1000_bp.yaml"
    # "configs/mf/cifar10_mlp_3x2000.yaml"
    # "configs/bp_baselines/cifar10_mlp_3x2000_bp.yaml"
    # "configs/mf/cifar100_mlp_3x2000.yaml"
    # "configs/bp_baselines/cifar100_mlp_3x2000_bp.yaml"
)

# Get number of configs
NUM_CONFIGS=${#CONFIG_FILES[@]}
# Update the --array range above dynamically based on this number if possible,
# otherwise ensure the --array range matches the number of configs listed.
# Example: #SBATCH --array=1-${NUM_CONFIGS}

INDEX=$((SLURM_ARRAY_TASK_ID - 1)) # Bash array is 0-indexed

# Check if index is valid
if [ $INDEX -lt 0 ] || [ $INDEX -ge $NUM_CONFIGS ]; then
    echo "Error: Invalid SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) for ${NUM_CONFIGS} configs."
    exit 1
fi

CONFIG_FILE=${CONFIG_FILES[$INDEX]}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '${CONFIG_FILE}' not found for task ID $SLURM_ARRAY_TASK_ID."
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
module load Python/3.10.4   # Use the SAME version
module load CUDA/12.4.0     # Use the SAME version

echo "Modules loaded:"
module list

echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

echo "Checking PyTorch CUDA..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" || { echo "PyTorch check failed"; exit 1; }

# --- Run the Experiment ---
echo "Running experiment script for task $SLURM_ARRAY_TASK_ID..."
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