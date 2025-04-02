# File: src/baselines/bp.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import time
import os
import pynvml # Import pynvml for type hinting handle
from typing import Dict, Any, Optional, Tuple, Callable, List

from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics
from src.utils.helpers import format_time, save_checkpoint, create_directory_if_not_exists
from src.utils.monitoring import get_gpu_memory_usage # Import memory usage function

logger = logging.getLogger(__name__)

# --- MODIFIED: Accept handle/active, sample memory, return peak_mem ---
def train_bp_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    input_adapter: Optional[Callable] = None,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, # ADDED
    nvml_active: bool = False, # ADDED
) -> Tuple[float, float, float]: # Returns Avg Loss, Avg Acc, Peak Mem Epoch
    """
    Performs one epoch of standard Backpropagation training.
    Logs BATCH metrics only. Returns epoch average loss, accuracy, and peak memory.
    """
    model.train()
    epoch_total_loss, epoch_total_correct, epoch_total_samples = 0.0, 0, 0 # Accumulators for epoch averages
    peak_mem_epoch = 0.0 # Initialize peak memory for this epoch

    pbar = tqdm(train_loader, desc=f"BP Epoch {epoch+1}/{total_epochs}", leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        step_ref[0] += 1
        current_global_step = step_ref[0]

        images, labels = images.to(device), labels.to(device)
        adapted_images = input_adapter(images) if input_adapter else images

        outputs = model(adapted_images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        batch_loss_value = loss.item()
        with torch.no_grad():
            predicted_labels = torch.argmax(outputs, dim=1)
            batch_correct = (predicted_labels == labels).sum().item()

        # Accumulate for epoch averages
        epoch_total_loss += batch_loss_value * batch_size
        epoch_total_correct += batch_correct
        epoch_total_samples += batch_size

        # --- Sample memory usage periodically or at end of batch ---
        current_mem_used = float('nan') # Default value if not measured
        if nvml_active and gpu_handle and ((batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1):
             mem_info = get_gpu_memory_usage(gpu_handle)
             if mem_info:
                 current_mem_used = mem_info[0] # Used memory in MiB
                 peak_mem_epoch = max(peak_mem_epoch, current_mem_used)
        # --- End memory sampling ---

        # Log batch metrics periodically
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
            batch_accuracy = (batch_correct / batch_size) * 100.0 if batch_size > 0 else 0.0
            pbar.set_postfix(loss=f"{batch_loss_value:.4f}", acc=f"{batch_accuracy:.2f}%")
            metrics_to_log = {
                "global_step": current_global_step,
                "BP_Baseline/Train_Loss_Batch": batch_loss_value,
                "BP_Baseline/Train_Acc_Batch": batch_accuracy,
            }
            # Log current memory usage alongside batch metrics if available
            if not torch.isnan(torch.tensor(current_mem_used)): # Check if valid value was obtained
                metrics_to_log["BP_Baseline/GPU_Mem_Used_MiB_Batch"] = current_mem_used
            log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)

    # --- End of Epoch ---
    avg_epoch_loss = epoch_total_loss / epoch_total_samples if epoch_total_samples > 0 else 0.0
    avg_epoch_accuracy = (epoch_total_correct / epoch_total_samples) * 100.0 if epoch_total_samples > 0 else 0.0

    # If memory was never sampled (e.g., very short epoch), sample it once at the end
    if nvml_active and gpu_handle and peak_mem_epoch == 0.0 and epoch_total_samples > 0:
        mem_info = get_gpu_memory_usage(gpu_handle)
        if mem_info:
             peak_mem_epoch = mem_info[0]

    return avg_epoch_loss, avg_epoch_accuracy, peak_mem_epoch # Return peak mem

# --- evaluate_bp_model (No changes needed here) ---
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

# --- train_bp_model (MODIFIED) ---
def train_bp_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, # ADDED
    nvml_active: bool = False, # ADDED
) -> float: # Returns overall peak memory
    """
    Orchestrates the end-to-end training of a model using Backpropagation.
    Returns the peak GPU memory observed during training.
    """
    model.to(device)
    logger.info("Starting standard Backpropagation training.")
    start_time = time.time()

    # --- Configuration ---
    train_config = config.get("training", config)
    optimizer_config = config.get("optimizer", {})
    checkpoint_config = config.get("checkpointing", {})

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
    save_best_metric = checkpoint_config.get("save_best_metric", "bp_val_accuracy").lower()
    save_best_metric_mode = "max" if "accuracy" in save_best_metric else "min"

    if criterion_name.lower() == "crossentropyloss": criterion = nn.CrossEntropyLoss()
    else: raise ValueError(f"Unsupported criterion: {criterion_name}")

    optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_params_extra}
    optimizer = getattr(optim, optimizer_name)(model.parameters(), **optimizer_kwargs)
    logger.info(f"Using optimizer: {optimizer_name} with params: {optimizer_kwargs}")

    scheduler = None
    if scheduler_name:
        try:
            if scheduler_name.lower() == "steplr": scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
            elif scheduler_name.lower() == "cosineannealinglr": scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, **scheduler_params)
            elif scheduler_name.lower() == "reducelronplateau": scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=save_best_metric_mode, **scheduler_params)
            else: logger.warning(f"Unsupported scheduler: {scheduler_name}.")
        except Exception as e: logger.error(f"Failed to create scheduler '{scheduler_name}': {e}", exc_info=True)
    if scheduler: logger.info(f"Using LR scheduler: {scheduler_name} with params: {scheduler_params}")

    best_metric_value = -float("inf") if save_best_metric_mode == "max" else float("inf")
    peak_mem_train = 0.0 # Initialize overall peak memory for the run

    for epoch in range(epochs):
        epoch_start_time = time.time()
        # Pass handle/active down, get avg train loss/acc and epoch peak mem back
        avg_epoch_train_loss, avg_epoch_train_acc, peak_mem_epoch = train_bp_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, epochs, wandb_run, log_interval, input_adapter, step_ref,
            gpu_handle, nvml_active # Pass handle and active status
        )
        # Update overall peak memory
        peak_mem_train = max(peak_mem_train, peak_mem_epoch)

        val_loss, val_acc = float("nan"), float("nan")
        is_best = False
        current_metric_value = None
        current_global_step = step_ref[0]
        logger.debug(f"End of Epoch {epoch+1} training. Current global_step: {current_global_step}. Peak Mem Epoch: {peak_mem_epoch:.2f} MiB")

        if val_loader:
            val_loss, val_acc = evaluate_bp_model(model, val_loader, criterion, device, input_adapter)
            if save_best_metric == "bp_val_accuracy": current_metric_value = val_acc
            elif save_best_metric == "bp_val_loss": current_metric_value = val_loss
            else: logger.warning(f"Unknown save_best_metric '{save_best_metric}'.")

            if current_metric_value is not None:
                metric_improved = (save_best_metric_mode == "max" and current_metric_value > best_metric_value) or \
                                  (save_best_metric_mode == "min" and current_metric_value < best_metric_value)
                if metric_improved:
                    best_metric_value = current_metric_value
                    is_best = True
                    logger.info(f"Epoch {epoch+1}: New best metric ({save_best_metric}): {best_metric_value:.4f}")

            if checkpoint_dir:
                create_directory_if_not_exists(checkpoint_dir)
                save_checkpoint(
                    state={
                        "epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "best_metric_value": best_metric_value, "val_loss": val_loss, "val_accuracy": val_acc
                    },
                    is_best=is_best, checkpoint_dir=checkpoint_dir,
                    filename=f"bp_checkpoint_epoch_{epoch+1}.pth",
                    best_filename=f"bp_{config.get('experiment_name', 'model')}_best.pth",
                )

        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_summary_metrics = {
            "global_step": current_global_step,
            "BP_Baseline/Train_Loss_Epoch": avg_epoch_train_loss,
            "BP_Baseline/Train_Acc_Epoch": avg_epoch_train_acc,
            "BP_Baseline/Val_Loss_Epoch": val_loss,
            "BP_Baseline/Val_Acc_Epoch": val_acc,
            "BP_Baseline/Epoch_Duration_Sec": epoch_duration,
            "BP_Baseline/Learning_Rate": current_lr,
            "BP_Baseline/Epoch": epoch + 1,
            "BP_Baseline/Peak_GPU_Mem_Epoch_MiB": peak_mem_epoch, # Log epoch peak mem
        }
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)
        logger.debug(f"Logged combined epoch summary at global_step {current_global_step}")

        logger.info(
            f"BP Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
            f"Train Loss: {avg_epoch_train_loss:.4f}, Acc: {avg_epoch_train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
            f"Peak Mem: {peak_mem_epoch:.1f} MiB | " # Add peak mem to console log
            f"Duration: {format_time(epoch_duration)}"
        )


        if scheduler:
            try:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    metric_for_scheduler = current_metric_value if current_metric_value is not None else (float('inf') if save_best_metric_mode == 'min' else -float('inf'))
                    scheduler.step(metric_for_scheduler)
                else:
                    scheduler.step()
            except Exception as e: logger.error(f"Failed to step scheduler: {e}", exc_info=True)

    total_training_time = time.time() - start_time
    logger.info(f"Finished standard Backpropagation training. Total time: {format_time(total_training_time)}")

    return peak_mem_train # Return the overall peak memory observed