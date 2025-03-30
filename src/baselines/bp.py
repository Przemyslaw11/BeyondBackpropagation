# File: src/baselines/bp.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import time
from typing import Dict, Any, Optional, Tuple, Callable  # Added Callable

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
    epoch: int,  # For logging
    total_epochs: int,  # For progress bar display
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    input_adapter: Optional[Callable] = None,
) -> Tuple[float, float]:
    """Performs one epoch of standard Backpropagation training."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    epoch_step = epoch * len(train_loader)  # Base step for this epoch

    pbar = tqdm(train_loader, desc=f"BP Epoch {epoch+1}/{total_epochs}", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Adapt input if necessary (e.g., flatten for MLP)
        adapted_images = input_adapter(images) if input_adapter else images

        # Standard BP forward pass
        outputs = model(adapted_images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Calculate Metrics ---
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size  # Accumulate total loss
        # Accuracy calculation
        with torch.no_grad():
            predicted_labels = torch.argmax(outputs, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
        total_samples += batch_size
        current_global_step = epoch_step + batch_idx

        # Logging (batch level)
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
            batch_accuracy = calculate_accuracy(
                outputs, labels
            )  # Re-calculate for just this batch log
            avg_loss_batch = loss.item()
            pbar.set_postfix(loss=f"{avg_loss_batch:.4f}", acc=f"{batch_accuracy:.2f}%")
            metrics = {
                "bp_train_loss_batch": avg_loss_batch,
                "bp_train_acc_batch": batch_accuracy,
            }
            log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

    avg_epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_epoch_accuracy = (
        (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    )

    return avg_epoch_loss, avg_epoch_accuracy


def evaluate_bp_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    input_adapter: Optional[Callable] = None,
) -> Tuple[float, float]:
    """Evaluates a model trained with Backpropagation."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating BP Model", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Adapt input if necessary
            adapted_images = input_adapter(images) if input_adapter else images

            outputs = model(adapted_images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predicted_labels = torch.argmax(outputs, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

            # Optional: update progress bar postfix
            # current_acc = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
            # pbar.set_postfix(acc=f"{current_acc:.2f}%")

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
    input_adapter: Optional[Callable] = None,
):
    """Orchestrates the end-to-end training of a model using Backpropagation."""
    model.to(device)
    logger.info("Starting standard Backpropagation training.")
    start_time = time.time()

    # --- Get Training Configuration ---
    # Look inside 'training' sub-dictionary if it exists, else top level
    train_config = config.get("training", config)
    optimizer_config = config.get("optimizer", {})  # Optimizer params might be separate

    optimizer_name = optimizer_config.get("type", "Adam")  # Get optimizer type
    # Combine specific LR/WD from optimizer_config with others from optimizer_params if structure varies
    lr = optimizer_config.get("lr", 0.001)
    weight_decay = optimizer_config.get("weight_decay", 0.0)
    optimizer_params_extra = optimizer_config.get(
        "params", {}
    )  # Other params like momentum

    criterion_name = train_config.get("criterion", "CrossEntropyLoss")
    epochs = train_config.get("epochs", 10)
    scheduler_name = train_config.get("scheduler", None)
    scheduler_params = train_config.get("scheduler_params", {})
    log_interval = train_config.get("log_interval", 100)
    # checkpoint_dir = train_config.get('checkpoint_dir', None) # TODO: Implement checkpointing

    # Select loss function
    if criterion_name.lower() == "crossentropyloss":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

    # Select optimizer
    optimizer_params = {
        "lr": lr,
        "weight_decay": weight_decay,
        **optimizer_params_extra,
    }
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    logger.info(f"Using optimizer: {optimizer_name} with params: {optimizer_params}")

    # Select LR scheduler (optional)
    scheduler = None
    if scheduler_name:
        try:
            if scheduler_name.lower() == "steplr":
                scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
            elif scheduler_name.lower() == "cosineannealinglr":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, **scheduler_params
                )  # T_max often set to total epochs
            elif scheduler_name.lower() == "reducelronplateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **scheduler_params
                )
            # Add other schedulers as needed
            else:
                logger.warning(
                    f"Unsupported scheduler: {scheduler_name}. Proceeding without scheduler."
                )
        except Exception as e:
            logger.error(
                f"Failed to create scheduler '{scheduler_name}': {e}", exc_info=True
            )
            scheduler = None  # Disable scheduler if creation fails

    if scheduler:
        logger.info(
            f"Using LR scheduler: {scheduler_name} with params: {scheduler_params}"
        )

    # --- Training Loop ---
    best_val_accuracy = -1.0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        train_loss, train_acc = train_bp_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            epochs,
            wandb_run,
            log_interval,
            input_adapter,
        )

        # Validation phase
        val_loss, val_acc = float("nan"), float("nan")  # Default if no val_loader
        if val_loader:
            val_loss, val_acc = evaluate_bp_model(
                model, val_loader, criterion, device, input_adapter
            )
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                # TODO: Save best model checkpoint based on accuracy
                logger.info(
                    f"Epoch {epoch+1}: New best validation accuracy: {best_val_accuracy:.2f}%"
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # TODO: Save best model checkpoint based on loss (optional)

        epoch_duration = time.time() - epoch_start_time

        # Log epoch metrics
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_metrics = {
            "bp_train_loss_epoch": train_loss,
            "bp_train_acc_epoch": train_acc,
            "bp_val_loss": val_loss,
            "bp_val_acc": val_acc,
            "bp_epoch": epoch + 1,
            "bp_epoch_duration_sec": epoch_duration,
            "bp_learning_rate": current_lr,
        }
        # Use the epoch number as the step for epoch-level W&B logging
        log_metrics(epoch_metrics, step=epoch + 1, wandb_run=wandb_run)

        logger.info(
            f"BP Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"Duration: {format_time(epoch_duration)}"
        )

        # Step the scheduler
        if scheduler:
            try:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(
                        val_loss if val_loader else train_loss
                    )  # Needs metric
                else:
                    scheduler.step()  # Step based on epoch
            except Exception as e:
                logger.error(f"Failed to step scheduler: {e}", exc_info=True)

    total_training_time = time.time() - start_time
    logger.info(
        f"Finished standard Backpropagation training. Total time: {format_time(total_training_time)}"
    )
    log_metrics(
        {"bp_total_training_time_sec": total_training_time}, wandb_run=wandb_run
    )


# Removed the __main__ block for cleaner baseline file
