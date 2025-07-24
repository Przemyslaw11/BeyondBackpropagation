# File: ./src/baselines/bp.py
"""Implementation of standard backpropagation training and evaluation."""

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import pynvml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.helpers import (
    create_directory_if_not_exists,
    format_time,
    save_checkpoint,
)
from src.utils.logging_utils import log_metrics
from src.utils.monitoring import get_gpu_memory_usage

if TYPE_CHECKING:
    # Use type-checking block to avoid circular import and an install-time dependency
    import wandb.sdk.wandb_run

logger = logging.getLogger(__name__)


def train_bp_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    wandb_run: "Optional[wandb.sdk.wandb_run.Run]" = None,
    log_interval: int = 100,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    step_ref: Optional[List[int]] = None,
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> Tuple[float, float, float]:
    """Performs one epoch of standard Backpropagation training.

    Logs BATCH metrics only. Returns epoch average loss, accuracy, and peak memory.
    """
    if step_ref is None:
        step_ref = [-1]

    model.train()
    epoch_total_loss, epoch_total_correct, epoch_total_samples = 0.0, 0, 0
    peak_mem_epoch = 0.0

    pbar = tqdm(train_loader, desc=f"BP Epoch {epoch + 1}/{total_epochs}", leave=False)

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

        epoch_total_loss += batch_loss_value * batch_size
        epoch_total_correct += batch_correct
        epoch_total_samples += batch_size

        current_mem_used = float("nan")
        is_log_time = (batch_idx + 1) % log_interval == 0
        is_last_batch = batch_idx == len(train_loader) - 1

        if nvml_active and gpu_handle and (is_log_time or is_last_batch):
            mem_info = get_gpu_memory_usage(gpu_handle)
            if mem_info:
                current_mem_used = mem_info[0]
                peak_mem_epoch = max(peak_mem_epoch, current_mem_used)

        if is_log_time or is_last_batch:
            batch_accuracy = (
                (batch_correct / batch_size) * 100.0 if batch_size > 0 else 0.0
            )
            pbar.set_postfix(
                loss=f"{batch_loss_value:.4f}", acc=f"{batch_accuracy:.2f}%"
            )
            metrics_to_log = {
                "global_step": current_global_step,
                "BP_Baseline/Train_Loss_Batch": batch_loss_value,
                "BP_Baseline/Train_Acc_Batch": batch_accuracy,
            }
            if not torch.isnan(torch.tensor(current_mem_used)):
                metrics_to_log["BP_Baseline/GPU_Mem_Used_MiB_Batch"] = current_mem_used
            log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)

    avg_epoch_loss = (
        epoch_total_loss / epoch_total_samples if epoch_total_samples > 0 else 0.0
    )
    avg_epoch_accuracy = (
        (epoch_total_correct / epoch_total_samples) * 100.0
        if epoch_total_samples > 0
        else 0.0
    )

    if nvml_active and gpu_handle and peak_mem_epoch == 0.0 and epoch_total_samples > 0:
        mem_info = get_gpu_memory_usage(gpu_handle)
        if mem_info:
            peak_mem_epoch = mem_info[0]

    return avg_epoch_loss, avg_epoch_accuracy, peak_mem_epoch


def evaluate_bp_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
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


def train_bp_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: "Optional[wandb.sdk.wandb_run.Run]" = None,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    step_ref: Optional[List[int]] = None,
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float:
    """Orchestrates the end-to-end training of a model using Backpropagation.

    Returns the peak GPU memory observed during training.
    """
    if step_ref is None:
        step_ref = [-1]

    model.to(device)
    logger.info("Starting standard Backpropagation training.")
    start_time = time.time()

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
    save_best_metric = checkpoint_config.get("save_best_metric", "bp_val_loss").lower()
    save_best_metric_mode = "max" if "accuracy" in save_best_metric else "min"

    es_enabled = train_config.get("early_stopping_enabled", True)
    es_metric = train_config.get("early_stopping_metric", "bp_val_loss").lower()
    es_patience = train_config.get("early_stopping_patience", 10)
    es_mode = train_config.get("early_stopping_mode", "min").lower()
    es_min_delta = train_config.get("early_stopping_min_delta", 0.0)

    if es_enabled:
        if val_loader is None:
            logger.warning(
                "Early stopping enabled but no validation loader provided. "
                "Disabling early stopping."
            )
            es_enabled = False
        else:
            logger.info(
                f"Early stopping enabled: Metric='{es_metric}', "
                f"Patience={es_patience}, Mode='{es_mode}', MinDelta={es_min_delta}"
            )
            metric_is_accuracy = "accuracy" in es_metric
            metric_is_loss = "loss" in es_metric
            if (es_mode == "min" and metric_is_accuracy) or (
                es_mode == "max" and metric_is_loss
            ):
                logger.error(
                    f"Early stopping mode '{es_mode}' is incompatible with metric "
                    f"'{es_metric}'. Disabling."
                )
                es_enabled = False
            epochs_no_improve = 0
            best_es_metric_value = float("inf") if es_mode == "min" else -float("inf")
    else:
        logger.info("Early stopping disabled.")

    if criterion_name.lower() == "crossentropyloss":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

    optimizer_kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
        **optimizer_params_extra,
    }
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    if not params_to_optimize:
        logger.error(
            "BP Baseline: No parameters found requiring gradients. Cannot train."
        )
        return 0.0

    optimizer = getattr(optim, optimizer_name)(params_to_optimize, **optimizer_kwargs)
    logger.info(f"Using optimizer: {optimizer_name} with params: {optimizer_kwargs}")

    optimized_param_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    logger.debug("BP Baseline Optimizer - Parameters being optimized:")
    for name in optimized_param_names:
        logger.debug(f"  - {name}")
    if any("projection_matrices" in name for name in optimized_param_names):
        logger.warning("BP Baseline Optimizer - WARNING: 'projection_matrices' found.")
    else:
        logger.debug(
            "BP Baseline Optimizer - Verified: 'projection_matrices' NOT optimized."
        )

    scheduler = None
    if scheduler_name:
        try:
            s_name = scheduler_name.lower()
            if s_name == "steplr":
                scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
            elif s_name == "cosineannealinglr":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, **scheduler_params
                )
            elif s_name == "reducelronplateau":
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

    best_checkpoint_metric_value = (
        -float("inf") if save_best_metric_mode == "max" else float("inf")
    )
    peak_mem_train = 0.0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        avg_epoch_train_loss, avg_epoch_train_acc, peak_mem_epoch = train_bp_epoch(
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
            step_ref,
            gpu_handle,
            nvml_active,
        )
        peak_mem_train = max(peak_mem_train, peak_mem_epoch)

        val_loss, val_acc = float("nan"), float("nan")
        is_best_for_checkpointing = False
        current_checkpoint_metric_value = None
        current_es_metric_value = None
        current_global_step = step_ref[0]
        logger.debug(
            f"End of Epoch {epoch + 1} training. Current global_step: "
            f"{current_global_step}. Peak Mem Epoch: {peak_mem_epoch:.2f} MiB"
        )

        if val_loader:
            val_loss, val_acc = evaluate_bp_model(
                model, val_loader, criterion, device, input_adapter
            )
            if save_best_metric == "bp_val_accuracy":
                current_checkpoint_metric_value = val_acc
            elif save_best_metric == "bp_val_loss":
                current_checkpoint_metric_value = val_loss
            else:
                logger.warning(f"Unknown save_best_metric '{save_best_metric}'.")

            metric_is_valid = (
                current_checkpoint_metric_value is not None
                and not torch.isnan(torch.tensor(current_checkpoint_metric_value))
            )
            if metric_is_valid:
                checkpoint_metric_improved = (
                    save_best_metric_mode == "max"
                    and current_checkpoint_metric_value > best_checkpoint_metric_value
                ) or (
                    save_best_metric_mode == "min"
                    and current_checkpoint_metric_value < best_checkpoint_metric_value
                )
                if checkpoint_metric_improved:
                    best_checkpoint_metric_value = current_checkpoint_metric_value
                    is_best_for_checkpointing = True
                    logger.info(
                        f"Epoch {epoch + 1}: New best checkpoint metric "
                        f"({save_best_metric}): {best_checkpoint_metric_value:.4f}"
                    )

            if checkpoint_dir:
                create_directory_if_not_exists(checkpoint_dir)
                checkpoint_filename = f"bp_checkpoint_epoch_{epoch + 1}.pth"
                experiment_name = config.get("experiment_name", "model")
                best_checkpoint_filename = f"bp_{experiment_name}_best.pth"
                save_checkpoint(
                    state={
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None,
                        "best_metric_value": best_checkpoint_metric_value,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    },
                    is_best=is_best_for_checkpointing,
                    checkpoint_dir=checkpoint_dir,
                    filename=checkpoint_filename,
                    best_filename=best_checkpoint_filename,
                )

            if es_enabled:
                if es_metric == "bp_val_accuracy":
                    current_es_metric_value = val_acc
                elif es_metric == "bp_val_loss":
                    current_es_metric_value = val_loss

                metric_is_invalid = current_es_metric_value is None or torch.isnan(
                    torch.tensor(current_es_metric_value)
                )
                if metric_is_invalid:
                    logger.warning(
                        f"Epoch {epoch + 1}: Early stopping metric '{es_metric}' is "
                        "None or NaN. Treating as no improvement."
                    )
                    epochs_no_improve += 1
                else:
                    if es_mode == "min":
                        improved = (
                            current_es_metric_value
                            < best_es_metric_value - es_min_delta
                        )
                    else:  # es_mode == "max"
                        improved = (
                            current_es_metric_value
                            > best_es_metric_value + es_min_delta
                        )

                    if improved:
                        best_es_metric_value = current_es_metric_value
                        epochs_no_improve = 0
                        logger.debug(
                            f"Epoch {epoch + 1}: Early stopping metric improved to "
                            f"{best_es_metric_value:.4f}. Reset patience."
                        )
                    else:
                        epochs_no_improve += 1
                        logger.debug(
                            f"Epoch {epoch + 1}: Early stopping metric did not "
                            f"improve. Patience: {epochs_no_improve}/{es_patience}."
                        )

                if epochs_no_improve >= es_patience:
                    logger.info("--- Early Stopping Triggered ---")
                    logger.info(
                        f"Metric '{es_metric}' did not improve for {es_patience} epochs."
                    )
                    logger.info(f"Stopping training at epoch {epoch + 1}.")
                    break

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
            "BP_Baseline/Peak_GPU_Mem_Epoch_MiB": peak_mem_epoch,
        }
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)
        logger.debug(
            f"Logged combined epoch summary at global_step {current_global_step}"
        )

        logger.info(
            f"BP Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f} | "
            f"Train Loss: {avg_epoch_train_loss:.4f}, "
            f"Acc: {avg_epoch_train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
            f"Peak Mem: {peak_mem_epoch:.1f} MiB | "
            f"Duration: {format_time(epoch_duration)}"
        )

        if scheduler:
            try:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    metric_for_scheduler = (
                        current_checkpoint_metric_value
                        if current_checkpoint_metric_value is not None
                        else (
                            float("inf")
                            if save_best_metric_mode == "min"
                            else -float("inf")
                        )
                    )
                    if torch.isnan(torch.tensor(metric_for_scheduler)):
                        logger.warning(
                            "Metric for ReduceLROnPlateau scheduler is NaN, skipping step."
                        )
                    else:
                        scheduler.step(metric_for_scheduler)
                else:
                    scheduler.step()
            except Exception as e:
                logger.error(f"Failed to step scheduler: {e}", exc_info=True)

    total_training_time = time.time() - start_time
    logger.info(
        "Finished standard Backpropagation training. Total time: "
        f"{format_time(total_training_time)}"
    )

    if checkpoint_dir:
        experiment_name = config.get("experiment_name", "model")
        best_checkpoint_path = os.path.join(
            checkpoint_dir, f"bp_{experiment_name}_best.pth"
        )
        if os.path.exists(best_checkpoint_path):
            try:
                logger.info(f"Loading best model state from: {best_checkpoint_path}")
                best_state_dict = torch.load(best_checkpoint_path, map_location=device)
                model.load_state_dict(best_state_dict)
                logger.info(
                    "Successfully loaded best model weights for final evaluation."
                )
            except Exception as e:
                logger.error(
                    f"Failed to load best checkpoint from {best_checkpoint_path}: {e}",
                    exc_info=True,
                )
                logger.warning("Proceeding with model state from the last epoch.")
        else:
            logger.warning(
                f"Best checkpoint file not found at {best_checkpoint_path}. "
                "Using model state from the last epoch."
            )
    else:
        logger.warning(
            "Checkpoint directory not specified. Using model state from the last epoch."
        )

    return peak_mem_train
