import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm # For progress bars
import time
from typing import Dict, Any, Optional, Tuple

# Assuming architectures and utils are accessible via src package
# We might need to import specific architectures if the BP baseline
# needs to adapt them (e.g., add a final classifier to CaFo blocks)
from src.architectures import FF_MLP, CaFo_CNN, MF_MLP # Import base architectures
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics
from src.utils.helpers import format_time

logger = logging.getLogger(__name__)

def train_bp_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int, # For logging
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    input_adapter: Optional[Callable] = None # Optional function to adapt input (e.g., flatten)
) -> Tuple[float, float]:
    """
    Performs one epoch of standard Backpropagation training.

    Args:
        model: The PyTorch model (nn.Module) to train.
        train_loader: DataLoader for the training data.
        optimizer: PyTorch optimizer.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        device: Device to perform training on ('cuda' or 'cpu').
        epoch: Current epoch number (for logging).
        wandb_run: Optional Weights & Biases run object for logging.
        log_interval: How often to log batch metrics.
        input_adapter: Optional function to apply to input images before model forward pass
                       (e.g., lambda x: x.view(x.shape[0], -1) for flattening).

    Returns:
        Tuple[float, float]: Average epoch loss, Average epoch accuracy.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    global_step = epoch * len(train_loader) # Estimate global step for logging

    pbar = tqdm(train_loader, desc=f"BP Epoch {epoch+1}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Adapt input if necessary (e.g., flatten for MLP)
        if input_adapter:
            images = input_adapter(images)

        # Standard BP forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        batch_accuracy = calculate_accuracy(outputs, labels)
        total_loss += loss.item() * images.size(0) # Accumulate total loss for epoch avg
        total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
        total_samples += labels.size(0)
        current_global_step = global_step + batch_idx

        # Logging
        if batch_idx % log_interval == 0:
            avg_loss_batch = loss.item()
            pbar.set_postfix(loss=f"{avg_loss_batch:.4f}", acc=f"{batch_accuracy:.2f}%")
            metrics = {
                'bp_train_loss_batch': avg_loss_batch,
                'bp_train_acc_batch': batch_accuracy,
            }
            log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

    avg_epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_epoch_accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

    return avg_epoch_loss, avg_epoch_accuracy


def evaluate_bp_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    input_adapter: Optional[Callable] = None # Optional function to adapt input
) -> Tuple[float, float]:
    """
    Evaluates a model trained with Backpropagation.

    Args:
        model: The trained PyTorch model (nn.Module).
        data_loader: DataLoader for the evaluation dataset (validation or test).
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        device: Device to perform evaluation on ('cuda' or 'cpu').
        input_adapter: Optional function to apply to input images before model forward pass.

    Returns:
        Tuple[float, float]: Average evaluation loss, Average evaluation accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating BP Model")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Adapt input if necessary
            if input_adapter:
                images = input_adapter(images)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total_samples += labels.size(0)

            current_acc = (total_correct / total_samples) * 100.0
            pbar.set_postfix(acc=f"{current_acc:.2f}%")

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

    return avg_loss, avg_accuracy


def train_bp_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None
):
    """
    Orchestrates the end-to-end training of a model using Backpropagation.

    Args:
        model: The PyTorch model instance.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation data.
        config: Dictionary with training configurations:
            - optimizer: Name of the optimizer (e.g., 'Adam', 'SGD').
            - optimizer_params: Dict of parameters for the optimizer (e.g., {'lr': 0.001, 'weight_decay': 0.0}).
            - criterion: Name of the loss function (e.g., 'CrossEntropyLoss').
            - epochs: Total number of training epochs.
            - scheduler (optional): Name of LR scheduler (e.g., 'StepLR', 'CosineAnnealingLR').
            - scheduler_params (optional): Dict of parameters for the scheduler.
            - log_interval: Frequency of logging batch metrics.
            - checkpoint_dir (optional): Directory to save model checkpoints.
        device: Device to run training on.
        wandb_run: Optional W&B run object.
        input_adapter: Optional function to adapt input images.
    """
    model.to(device)
    logger.info("Starting standard Backpropagation training.")
    start_time = time.time()

    # --- Get Training Configuration ---
    optimizer_name = config.get('optimizer', 'Adam')
    optimizer_params = config.get('optimizer_params', {'lr': 0.001})
    criterion_name = config.get('criterion', 'CrossEntropyLoss')
    epochs = config.get('epochs', 10)
    scheduler_name = config.get('scheduler', None)
    scheduler_params = config.get('scheduler_params', {})
    log_interval = config.get('log_interval', 100)
    # checkpoint_dir = config.get('checkpoint_dir', None) # TODO: Implement checkpointing

    # Select loss function
    if criterion_name.lower() == 'crossentropyloss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

    # Select optimizer (acts on all model parameters)
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Select LR scheduler (optional)
    scheduler = None
    if scheduler_name:
        if scheduler_name.lower() == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler_name.lower() == 'cosineannealinglr':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        # Add other schedulers as needed
        else:
            logger.warning(f"Unsupported scheduler: {scheduler_name}. Proceeding without scheduler.")

    # --- Training Loop ---
    best_val_accuracy = -1.0
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        train_loss, train_acc = train_bp_epoch(
            model, train_loader, optimizer, criterion, device, epoch, wandb_run, log_interval, input_adapter
        )

        # Validation phase
        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            val_loss, val_acc = evaluate_bp_model(
                model, val_loader, criterion, device, input_adapter
            )
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                # TODO: Save best model checkpoint if checkpoint_dir is provided
                logger.info(f"New best validation accuracy: {best_val_accuracy:.2f}%")

        epoch_duration = time.time() - epoch_start_time

        # Log epoch metrics
        epoch_metrics = {
            'bp_train_loss_epoch': train_loss,
            'bp_train_acc_epoch': train_acc,
            'bp_val_loss': val_loss,
            'bp_val_acc': val_acc,
            'bp_epoch': epoch + 1,
            'bp_epoch_duration_sec': epoch_duration,
            'bp_learning_rate': optimizer.param_groups[0]['lr'] # Log current LR
        }
        log_metrics(epoch_metrics, step=epoch + 1, wandb_run=wandb_run) # Log against epoch number

        logger.info(
            f"BP Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"Duration: {format_time(epoch_duration)}"
        )

        # Step the scheduler
        if scheduler:
            # Some schedulers are stepped after epoch (e.g., StepLR), others after batch
            # Assume epoch-based stepping here unless specified otherwise
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(val_loss) # Needs validation metric
            else:
                 scheduler.step()

    total_training_time = time.time() - start_time
    logger.info(f"Finished standard Backpropagation training. Total time: {format_time(total_training_time)}")
    log_metrics({'bp_total_training_time_sec': total_training_time}, wandb_run=wandb_run)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Testing BP baseline components...")

    # Setup
    batch_size = 4
    input_dim = 784 # F-MNIST like
    num_classes = 10
    hidden_dims = [100, 50]
    epochs = 2
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy Data
    num_batches = 10
    dummy_images = torch.randn(batch_size * num_batches, input_dim)
    dummy_labels = torch.randint(0, num_classes, (batch_size * num_batches,))
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=batch_size)
    # Dummy validation loader (can be same data for quick test)
    dummy_val_loader = DataLoader(dummy_dataset, batch_size=batch_size)

    # --- Test with MF_MLP architecture ---
    print("\n--- Testing BP with MF_MLP Architecture ---")
    try:
        # Use the standard forward pass of MF_MLP
        model_mf = MF_MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes
        ).to(device)
        print(model_mf)

        # Config for BP training
        config_bp_mf = {
            'optimizer': 'AdamW',
            'optimizer_params': {'lr': lr, 'weight_decay': 0.01},
            'criterion': 'CrossEntropyLoss',
            'epochs': epochs,
            'log_interval': 5,
            # 'scheduler': 'StepLR', # Example scheduler
            # 'scheduler_params': {'step_size': 1, 'gamma': 0.1}
        }

        # Input adapter for MLP (flatten) - MF_MLP already expects flat input
        # input_adapter_mlp = lambda x: x.view(x.shape[0], -1)

        train_bp_model(
            model=model_mf,
            train_loader=dummy_loader,
            val_loader=dummy_val_loader,
            config=config_bp_mf,
            device=device,
            input_adapter=None # No adapter needed for MF_MLP
        )
        print("BP training with MF_MLP architecture finished.")

        # Evaluate final model
        final_loss, final_acc = evaluate_bp_model(
            model_mf, dummy_val_loader, nn.CrossEntropyLoss(), device, input_adapter=None
        )
        print(f"Final evaluation - Loss: {final_loss:.4f}, Acc: {final_acc:.2f}%")

    except Exception as e:
        print(f"\nError during BP testing with MF_MLP: {e}", exc_info=True)


    # --- Test with CaFo_CNN architecture (Blocks + new Classifier) ---
    print("\n--- Testing BP with CaFo_CNN Architecture ---")
    try:
        # Need to adapt CaFo_CNN for BP: use blocks + add a final classifier
        cafo_base = CaFo_CNN(
            input_channels=1, # F-MNIST like
            block_channels=[16, 32],
            image_size=28,
            num_classes=num_classes # This predictor won't be used by BP
        )

        # Calculate output features from the last block
        with torch.no_grad():
             dummy_cafo_input = torch.randn(1, 1, 28, 28)
             last_block_output = cafo_base.forward_blocks_only(dummy_cafo_input)
             num_output_features = last_block_output.numel() # Flattened size

        # Create the BP model: CaFo blocks + new Linear layer
        model_cafo_bp = nn.Sequential(
            cafo_base.blocks, # Use the ModuleList of blocks
            nn.Flatten(),
            nn.Linear(num_output_features, num_classes)
        ).to(device)
        print(model_cafo_bp)

        # Config for BP training
        config_bp_cafo = {
            'optimizer': 'Adam',
            'optimizer_params': {'lr': lr},
            'criterion': 'CrossEntropyLoss',
            'epochs': epochs,
            'log_interval': 5
        }

        # Need spatial dummy data for CNN
        dummy_images_spatial = dummy_images.view(batch_size * num_batches, 1, 28, 28)
        dummy_dataset_spatial = TensorDataset(dummy_images_spatial, dummy_labels)
        dummy_loader_spatial = DataLoader(dummy_dataset_spatial, batch_size=batch_size)
        dummy_val_loader_spatial = DataLoader(dummy_dataset_spatial, batch_size=batch_size)

        train_bp_model(
            model=model_cafo_bp,
            train_loader=dummy_loader_spatial,
            val_loader=dummy_val_loader_spatial,
            config=config_bp_cafo,
            device=device,
            input_adapter=None # No adapter needed, model handles spatial input
        )
        print("BP training with CaFo_CNN architecture finished.")

        # Evaluate final model
        final_loss_cafo, final_acc_cafo = evaluate_bp_model(
            model_cafo_bp, dummy_val_loader_spatial, nn.CrossEntropyLoss(), device, input_adapter=None
        )
        print(f"Final evaluation (CaFo Arch) - Loss: {final_loss_cafo:.4f}, Acc: {final_acc_cafo:.2f}%")


    except Exception as e:
        print(f"\nError during BP testing with CaFo_CNN: {e}", exc_info=True)
