#!/bin/bash
#SBATCH --job-name=BBP_Array_%a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=08:00:00
#SBATCH -A plgoncotherapy-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --array=1-25

CONFIG_FILES=(
    # --- BP Baselines ---
    "configs/bp_baselines/cifar100_cnn_3block_bp.yaml"
    "configs/bp_baselines/cifar100_mlp_3x2000_bp.yaml"
    "configs/bp_baselines/cifar10_cnn_3block_bp.yaml"
    "configs/bp_baselines/cifar10_mlp_3x2000_bp.yaml"
    "configs/bp_baselines/fashion_mnist_cnn_3block_bp.yaml"
    "configs/bp_baselines/fashion_mnist_mlp_2x1000_bp.yaml"
    "configs/bp_baselines/fashion_mnist_mlp_4x2000_bp.yaml"
    "configs/bp_baselines/mnist_cnn_3block_bp.yaml"
    "configs/bp_baselines/mnist_mlp_3x1000_bp.yaml"
    "configs/bp_baselines/mnist_mlp_4x2000_bp.yaml"
    "configs/bp_baselines/mnist_mlp_2x1000_bp.yaml"

    # --- CaFo ---
    "configs/cafo/cafodfa_cifar100_cnn_3block.yaml"
    "configs/cafo/cafodfa_cifar10_cnn_3block.yaml"
    "configs/cafo/cafodfa_fashion_mnist_cnn_3block.yaml"
    "configs/cafo/cafodfa_mnist_cnn_3block.yaml"
    "configs/cafo/cifar100_cnn_3block.yaml"
    "configs/cafo/cifar10_cnn_3block.yaml"
    "configs/cafo/fashion_mnist_cnn_3block.yaml"
    "configs/cafo/mnist_cnn_3block.yaml"

    # --- FF ---
    "configs/ff/fashion_mnist_mlp_4x2000.yaml"
    "configs/ff/mnist_mlp_3x1000_ADAMW.yaml"
    "configs/ff/mnist_mlp_3x1000_SGD_ref.yaml"
    "configs/ff/mnist_mlp_4x2000.yaml"

    # --- MF ---
    "configs/mf/cifar100_mlp_3x2000.yaml"
    "configs/mf/cifar10_mlp_3x2000.yaml"
    "configs/mf/fashion_mnist_mlp_2x1000.yaml"
    "configs/mf/mnist_mlp_2x1000.yaml"
)

NUM_CONFIGS=${#CONFIG_FILES[@]}
echo "INFO: Number of config files listed: ${NUM_CONFIGS}. Slurm array range set to 1-${SLURM_ARRAY_TASK_MAX}"
if [ "${SLURM_ARRAY_TASK_MAX}" -ne "${NUM_CONFIGS}" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!! WARNING: Slurm --array range (1-${SLURM_ARRAY_TASK_MAX}) DOES NOT MATCH the number   !!"
    echo "!!          of config files listed in this script (${NUM_CONFIGS}). Check the script!            !!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi

INDEX=$((SLURM_ARRAY_TASK_ID - 1))

if [ $INDEX -lt 0 ] || [ $INDEX -ge $NUM_CONFIGS ]; then
    echo "Error: Invalid SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) for ${NUM_CONFIGS} configs." >&2
    exit 1
fi

CONFIG_FILE=${CONFIG_FILES[$INDEX]}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '${CONFIG_FILE}' not found for task ID $SLURM_ARRAY_TASK_ID." >&2
    exit 1
fi

echo "#####################################################################"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on host: $(hostname)"
echo "Executing command in: $(pwd)"
echo "Using config file (${INDEX}): $CONFIG_FILE"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "#####################################################################"

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "Loading modules..."
module purge
module load Python/3.10.4
module load CUDA/12.4.0

echo "Modules loaded:"
module list

echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Set WANDB to offline mode for compute nodes
export WANDB_MODE=offline
echo "Set WANDB_MODE=offline"

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
if [ $EXIT_CODE -eq 0 ]; then
    echo "Array Task $SLURM_ARRAY_TASK_ID finished successfully."
else
    echo "Array Task $SLURM_ARRAY_TASK_ID finished with errors (Exit Code: $EXIT_CODE)."
fi
date
echo "To sync WandB offline runs (after ALL tasks finish):"
echo "cd $SLURM_SUBMIT_DIR"
echo "source venv/bin/activate"
echo "wandb sync --sync-all # Syncs all runs in the local wandb directory"
echo "#####################################################################"

exit $EXIT_CODE