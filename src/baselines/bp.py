# File: src/baselines/bp.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import time
import os  # For checkpointing
from typing import Dict, Any, Optional, Tuple, Callable

from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics
from src.utils.helpers import format_time, save_checkpoint  # Import checkpoint helper

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
    total_loss, total_correct, total_samples = 0.0, 0, 0
    epoch_step = epoch * len(train_loader)  # Base step for this epoch

    pbar = tqdm(train_loader, desc=f"BP Epoch {epoch+1}/{total_epochs}", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        adapted_images = input_adapter(images) if input_adapter else images

        outputs = model(adapted_images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        with torch.no_grad():
            predicted_labels = torch.argmax(outputs, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
        total_samples += batch_size
        current_global_step = epoch_step + batch_idx + 1 # Use 1-based indexing for steps if preferred

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
            batch_accuracy = calculate_accuracy(outputs, labels)
            avg_loss_batch = loss.item()
            pbar.set_postfix(loss=f"{avg_loss_batch:.4f}", acc=f"{batch_accuracy:.2f}%")
            # Use consistent naming for W&B
            metrics = {
                "BP_Baseline/Train_Loss_Batch": avg_loss_batch,
                "BP_Baseline/Train_Acc_Batch": batch_accuracy,
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
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating BP Model", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            adapted_images = input_adapter(images) if input_adapter else images
            outputs = model(adapted_images)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predicted_labels = torch.argmax(outputs, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    return avg_loss, avg_accuracy


# --- Updated train_bp_model Function ---
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

    # --- Configuration ---
    train_config = config.get("training", config)
    optimizer_config = config.get("optimizer", {})
    checkpoint_config = config.get("checkpointing", {})  # Get checkpoint config

    logger.info("BP Training using optimizer config: %s", optimizer_config)

    optimizer_name = optimizer_config.get("type", "AdamW")
    lr = optimizer_config.get("lr", 0.001)
    weight_decay = optimizer_config.get("weight_decay", 0.0)
    optimizer_params_extra = optimizer_config.get("params", {})

    criterion_name = train_config.get("criterion", "CrossEntropyLoss")
    epochs = train_config.get("epochs", 10)
    scheduler_name = train_config.get("scheduler", None)
    scheduler_params = train_config.get("scheduler_params", {})
    log_interval = train_config.get("log_interval", 100)
    checkpoint_dir = checkpoint_config.get("checkpoint_dir", None)
    save_best_metric = checkpoint_config.get(
        "save_best_metric", "bp_val_accuracy"
    ).lower()
    save_best_metric_mode = "max" if "accuracy" in save_best_metric else "min"

    if criterion_name.lower() == "crossentropyloss":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

    optimizer_kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
        **optimizer_params_extra,
    }
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), **optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    logger.info(f"Using optimizer: {optimizer_name} with params: {optimizer_kwargs}")

    scheduler = None
    if scheduler_name:
        try:
            if scheduler_name.lower() == "steplr":
                scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
            elif scheduler_name.lower() == "cosineannealinglr":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, **scheduler_params
                )
            elif scheduler_name.lower() == "reducelronplateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode=save_best_metric_mode, **scheduler_params
                )
            else:
                logger.warning(f"Unsupported scheduler: {scheduler_name}.")
        except Exception as e:
            logger.error(
                f"Failed to create scheduler '{scheduler_name}': {e}", exc_info=True
            )
    if scheduler:
        logger.info(
            f"Using LR scheduler: {scheduler_name} with params: {scheduler_params}"
        )

    # --- Training Loop ---
    best_metric_value = (
        -float("inf") if save_best_metric_mode == "max" else float("inf")
    )
    num_batches_per_epoch = len(train_loader) # Store length for step calculation

    for epoch in range(epochs):
        epoch_start_time = time.time()
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

        val_loss, val_acc = float("nan"), float("nan")
        is_best = False
        current_metric_value = None
        if val_loader:
            val_loss, val_acc = evaluate_bp_model(
                model, val_loader, criterion, device, input_adapter
            )
            if save_best_metric == "bp_val_accuracy":
                current_metric_value = val_acc
            elif save_best_metric == "bp_val_loss":
                current_metric_value = val_loss
            else:
                logger.warning(
                    f"Unknown save_best_metric '{save_best_metric}', checkpointing disabled."
                )

            if current_metric_value is not None:
                if (
                    save_best_metric_mode == "max"
                    and current_metric_value > best_metric_value
                ):
                    best_metric_value = current_metric_value
                    is_best = True
                    logger.info(
                        f"Epoch {epoch+1}: New best validation metric ({save_best_metric}): {best_metric_value:.4f}"
                    )
                elif (
                    save_best_metric_mode == "min"
                    and current_metric_value < best_metric_value
                ):
                    best_metric_value = current_metric_value
                    is_best = True
                    logger.info(
                        f"Epoch {epoch+1}: New best validation metric ({save_best_metric}): {best_metric_value:.4f}"
                    )

            if checkpoint_dir:
                save_checkpoint(
                    state={
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "best_metric_value": best_metric_value,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    },
                    is_best=is_best,
                    checkpoint_dir=checkpoint_dir,
                    filename=f"bp_checkpoint_epoch_{epoch+1}.pth",
                    best_filename=f"bp_{config.get('experiment_name', 'model')}_best.pth",
                )

        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # *** FIX FOR W&B STEP LOGGING ***
        # Calculate the global step corresponding to the *end* of this epoch
        final_global_step_epoch = (epoch + 1) * num_batches_per_epoch

        epoch_metrics = {
            "BP_Baseline/Train_Loss_Epoch": train_loss,
            "BP_Baseline/Train_Acc_Epoch": train_acc,
            "BP_Baseline/Val_Loss_Epoch": val_loss,
            "BP_Baseline/Val_Acc_Epoch": val_acc,
            "BP_Baseline/Epoch": epoch + 1,
            "BP_Baseline/Epoch_Duration_Sec": epoch_duration,
            "BP_Baseline/Learning_Rate": current_lr,
        }
        # Log epoch metrics using the FINAL global step of that epoch
        log_metrics(epoch_metrics, step=final_global_step_epoch, wandb_run=wandb_run)
        # ********************************

        logger.info(
            f"BP Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | Duration: {format_time(epoch_duration)}"
        )

        if scheduler:
            try:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # Ensure current_metric_value is not None before stepping
                    metric_for_scheduler = current_metric_value if current_metric_value is not None else (float('inf') if save_best_metric_mode == 'min' else -float('inf'))
                    scheduler.step(metric_for_scheduler)
                else:
                    scheduler.step()
            except Exception as e:
                logger.error(f"Failed to step scheduler: {e}", exc_info=True)

    total_training_time = time.time() - start_time
    logger.info(
        f"Finished standard Backpropagation training. Total time: {format_time(total_training_time)}"
    )
    # Log total training time using the final step count
    final_total_step = epochs * num_batches_per_epoch
    log_metrics(
        {"BP_Baseline/Total_Training_Time_Sec": total_training_time},
        step=final_total_step, # Log against the final step
        wandb_run=wandb_run,
    )