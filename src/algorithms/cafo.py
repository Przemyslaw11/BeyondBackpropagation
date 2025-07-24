"""Implementation of the Cascaded Forward (CaFo) algorithm."""

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architectures.cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from src.utils.helpers import (
    create_directory_if_not_exists,
    format_time,
    save_checkpoint,
)
from src.utils.logging_utils import log_metrics
from src.utils.metrics import calculate_accuracy
from src.utils.monitoring import get_gpu_memory_usage

if TYPE_CHECKING:
    import wandb.sdk.wandb_run

logger = logging.getLogger(__name__)


def train_cafo_dfa_blocks(
    model: CaFo_CNN,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    step_ref: Optional[List[int]] = None,
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float:
    """Trains the blocks of the CaFo_CNN model using Direct Feedback Alignment (DFA).

    Uses the fully-connected approximation for Conv layer gradient updates.
    """
    if step_ref is None:
        step_ref = [-1]

    logger.info("--- Starting CaFo Block Training (DFA) Phase ---")
    model.to(device)
    model.train()

    algo_config = config.get("algorithm_params", {})
    data_config = config.get("data", {})
    num_classes = data_config.get("num_classes", 10)
    epochs = algo_config.get("block_training_epochs", 10)
    optimizer_name = algo_config.get("block_optimizer_type", "Adam")
    lr = algo_config.get("block_lr", 0.0001)
    weight_decay = algo_config.get("block_weight_decay", 0.0)
    log_interval = algo_config.get("log_interval", 100)
    dfa_feedback_matrix_type = algo_config.get("dfa_feedback_matrix_type", "gaussian")

    try:
        last_block_idx = len(model.blocks) - 1
        last_block_flat_dim = model.get_predictor_input_dim(last_block_idx)
        aux_layer = nn.Linear(last_block_flat_dim, num_classes).to(device)
        nn.init.kaiming_uniform_(aux_layer.weight, a=math.sqrt(5))
        if aux_layer.bias is not None:
            nn.init.zeros_(aux_layer.bias)
        logger.info(
            f"Created auxiliary layer for DFA block training: "
            f"Linear({last_block_flat_dim}, {num_classes})"
        )
    except Exception as e:
        logger.error(f"Failed to create auxiliary layer for DFA: {e}", exc_info=True)
        return float("nan")

    # --- Setup Feedback Matrices (B_i) ---
    feedback_matrices = []
    error_dim = num_classes
    for i in range(len(model.blocks)):  # Need B_0 to B_{L-1}
        block_output_flat_dim = model.get_predictor_input_dim(i)
        b_matrix = torch.randn(error_dim, block_output_flat_dim, device=device)
        if dfa_feedback_matrix_type == "uniform":
            b_matrix.uniform_(-1.0, 1.0)
        with torch.no_grad():
            b_matrix /= torch.norm(b_matrix, dim=1, keepdim=True) + 1e-8
        b_matrix.requires_grad_(False)  # Ensure they are fixed
        feedback_matrices.append(b_matrix)
        logger.debug(f"Created fixed feedback matrix B_{i} with shape {b_matrix.shape}")

    # --- Setup Optimizer for Blocks AND Auxiliary Layer ---
    block_params = list(model.blocks.parameters())
    aux_params = list(aux_layer.parameters())
    all_params_to_train = block_params + aux_params
    if not all_params_to_train:
        logger.error("DFA Block Training: No parameters found to train.")
        return 0.0  # No memory used if nothing to train

    optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay}
    optimizer = getattr(optim, optimizer_name)(all_params_to_train, **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()  # For the auxiliary layer loss

    # --- Training Loop ---
    peak_mem_block_train = 0.0
    block_train_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        aux_layer.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        peak_mem_block_epoch = 0.0
        pbar = tqdm(
            train_loader, desc=f"DFA Block Epoch {epoch + 1}/{epochs}", leave=False
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1
            current_global_step = step_ref[0]
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

            # --- Forward Pass (Store Activations) ---
            activations = [images]  # h_0 = input
            current_h = images
            for _, block in enumerate(model.blocks):
                current_h = block(current_h)
                activations.append(current_h)  # Store h_1, h_2, ..., h_L

            last_block_output = activations[-1]
            last_block_output_flat = last_block_output.view(batch_size, -1)
            aux_output_logits = aux_layer(last_block_output_flat)

            # --- Loss Calculation ---
            loss = criterion(aux_output_logits, labels)

            # --- Gradient Calculation & DFA Update ---
            optimizer.zero_grad()
            loss.backward()

            # --- DFA Gradient Calculation ---
            with torch.no_grad():
                global_error = (
                    F.softmax(aux_output_logits, dim=1) - labels_one_hot
                ).detach()

            for i in range(len(model.blocks) - 1):
                block_index = i
                h_i = activations[block_index + 1]
                delta_h_i_flat = torch.matmul(
                    global_error, feedback_matrices[block_index]
                )
                delta_h_i_spatial = delta_h_i_flat.view_as(h_i)
                target_block = model.blocks[block_index]
                h_prev = activations[block_index]
                try:
                    original_mode = target_block.training
                    target_block.train()
                    h_i_recompute = target_block(h_prev.detach())
                    grads = torch.autograd.grad(
                        outputs=h_i_recompute,
                        inputs=target_block.parameters(),
                        grad_outputs=delta_h_i_spatial,
                        allow_unused=True,
                        retain_graph=False,
                    )
                    for param, grad in zip(
                        target_block.parameters(), grads, strict=False
                    ):
                        if grad is not None:
                            param.grad = grad
                    target_block.train(original_mode)
                except Exception as e_dfa_grad:
                    logger.error(
                        f"Error computing/assigning DFA gradient for block "
                        f"{block_index}: {e_dfa_grad}",
                        exc_info=True,
                    )

            optimizer.step()

            # --- Logging & Metrics ---
            with torch.no_grad():
                total_loss += loss.item() * batch_size
                predicted_labels = torch.argmax(aux_output_logits, dim=1)
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += batch_size

            # --- Sample memory usage ---
            current_mem_used = float("nan")
            if nvml_active and gpu_handle:
                mem_info = get_gpu_memory_usage(gpu_handle)
                if mem_info:
                    current_mem_used = mem_info[0]
                    peak_mem_block_epoch = max(peak_mem_block_epoch, current_mem_used)

            is_log_time = (batch_idx + 1) % log_interval == 0
            is_last_batch = batch_idx == len(train_loader) - 1
            if is_log_time or is_last_batch:
                batch_accuracy = calculate_accuracy(aux_output_logits, labels)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", acc=f"{batch_accuracy:.2f}%"
                )
                metrics_to_log = {
                    "global_step": current_global_step,
                    "CaFo_DFA/BlockTrain_Loss_Batch": loss.item(),
                    "CaFo_DFA/BlockTrain_Acc_Batch": batch_accuracy,
                }
                if not torch.isnan(torch.tensor(current_mem_used)):
                    metrics_to_log["CaFo_DFA/BlockTrain_GPU_Mem_MiB_Batch"] = (
                        current_mem_used
                    )
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)
        # --- End Batch Loop ---

        avg_loss = total_loss / total_samples if total_samples > 0 else float("nan")
        avg_acc = (
            (total_correct / total_samples) * 100.0
            if total_samples > 0
            else float("nan")
        )
        peak_mem_block_train = max(peak_mem_block_train, peak_mem_block_epoch)
        epoch_duration = time.time() - epoch_start_time

        logger.info(
            f"DFA Block Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f}, "
            f"Avg Acc: {avg_acc:.2f}% | "
            f"Peak Mem Epoch: {peak_mem_block_epoch:.1f} MiB | "
            f"Duration: {format_time(epoch_duration)}"
        )
        epoch_summary_metrics = {
            "global_step": current_global_step,  # Use last step
            "CaFo_DFA/BlockTrain_Loss_EpochAvg": avg_loss,
            "CaFo_DFA/BlockTrain_Acc_EpochAvg": avg_acc,
            "CaFo_DFA/BlockTrain_Peak_GPU_Mem_Epoch_MiB": peak_mem_block_epoch,
        }
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)

    # --- End Epoch Loop ---
    block_train_duration = time.time() - block_train_start_time
    logger.info(
        f"--- Finished CaFo Block Training (DFA) Phase. "
        f"Duration: {format_time(block_train_duration)} ---"
    )

    if nvml_active and gpu_handle:
        mem_info = get_gpu_memory_usage(gpu_handle)
        if mem_info:
            peak_mem_block_train = max(peak_mem_block_train, mem_info[0])

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    del aux_layer, feedback_matrices

    return peak_mem_block_train


@torch.no_grad()
def evaluate_cafo_predictor(
    block: CaFoBlock,
    predictor: CaFoPredictor,
    val_loader: DataLoader,
    device: torch.device,
    get_block_input_fn: Callable[[torch.Tensor], torch.Tensor],
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """Evaluates a single predictor on a validation set."""
    block.eval()
    predictor.eval()
    block.to(device)
    predictor.to(device)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        block_input = get_block_input_fn(images)
        block_output = block(block_input)
        predictions = predictor(block_output)

        if criterion:
            loss = criterion(predictions, labels)
            total_loss += loss.item() * images.size(0)

        pred_labels = torch.argmax(predictions, dim=1)
        total_correct += (pred_labels == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = (
        total_loss / total_samples if criterion and total_samples > 0 else float("nan")
    )
    avg_accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

    return avg_loss, avg_accuracy


def train_cafo_predictor_only(
    block: CaFoBlock,
    predictor: CaFoPredictor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    device: torch.device,
    get_block_input_fn: Callable[[torch.Tensor], torch.Tensor],
    early_stopping_config: Dict[str, Any],
    wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    log_interval: int = 100,
    block_index: int = 0,
    step_ref: Optional[List[int]] = None,
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> Tuple[float, float, float, int]:
    """Trains a single CaFoPredictor, keeping its corresponding CaFoBlock frozen.

    Includes early stopping based on validation performance.

    Returns:
        A tuple of (Avg Train Loss, Avg Train Acc, Peak Memory, Epochs Trained).
    """
    if step_ref is None:
        step_ref = [-1]

    block.eval()
    predictor.train()
    block.to(device)
    predictor.to(device)
    log_prefix = f"Predictor_{block_index + 1}"
    logger.info(
        f"Starting CaFo training for {log_prefix} (Block {block_index + 1} frozen)"
    )

    peak_mem_predictor_train = 0.0
    final_avg_epoch_loss = float("nan")
    final_avg_epoch_accuracy = float("nan")
    epochs_trained = 0

    es_enabled = early_stopping_config.get("enabled", False)
    es_metric_name = early_stopping_config.get("metric", "val_loss").lower()
    es_patience = early_stopping_config.get("patience", 10)
    es_mode = early_stopping_config.get("mode", "min").lower()
    es_min_delta = early_stopping_config.get("min_delta", 0.0)
    epochs_no_improve = 0
    best_es_metric_value = float("inf") if es_mode == "min" else -float("inf")

    if es_enabled:
        if val_loader is None:
            logger.warning(
                f"{log_prefix}: Early stopping enabled but no val_loader provided. Disabling."
            )
            es_enabled = False
        else:
            is_acc = "accuracy" in es_metric_name
            is_loss = "loss" in es_metric_name
            if (es_mode == "min" and is_acc) or (es_mode == "max" and is_loss):
                logger.error(
                    f"{log_prefix}: Early stopping mode '{es_mode}' incompatible "
                    f"with metric '{es_metric_name}'. Disabling."
                )
                es_enabled = False
            else:
                log_msg = (
                    f"{log_prefix}: Early Stopping Enabled - "
                    f"Metric: '{es_metric_name}', Patience: {es_patience}, "
                    f"Mode: '{es_mode}', MinDelta: {es_min_delta}"
                )
                logger.info(log_msg)
    else:
        logger.info(f"{log_prefix}: Early Stopping Disabled.")

    for epoch in range(epochs):
        epochs_trained = epoch + 1
        predictor.train()
        epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
        peak_mem_predictor_epoch = 0.0
        pbar = tqdm(
            train_loader, desc=f"{log_prefix} Epoch {epoch + 1}/{epochs}", leave=False
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1
            current_global_step = step_ref[0]
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                block_input = get_block_input_fn(images)
                block_output = block(block_input)
            predictions = predictor(block_output.detach())
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_labels = torch.argmax(predictions, dim=1)
                batch_correct = (pred_labels == labels).sum().item()
            batch_accuracy = (
                (batch_correct / labels.size(0)) * 100.0 if labels.size(0) > 0 else 0.0
            )
            epoch_loss += loss.item() * labels.size(0)
            epoch_correct += batch_correct
            epoch_samples += labels.size(0)

            current_mem_used = float("nan")
            if nvml_active and gpu_handle:
                mem_info = get_gpu_memory_usage(gpu_handle)
                if mem_info:
                    current_mem_used = mem_info[0]
                    peak_mem_predictor_epoch = max(
                        peak_mem_predictor_epoch, current_mem_used
                    )

            is_log_time = (batch_idx + 1) % log_interval == 0
            is_last_batch = batch_idx == len(train_loader) - 1
            if is_log_time or is_last_batch:
                avg_loss_batch = loss.item()
                pbar.set_postfix(
                    loss=f"{avg_loss_batch:.4f}", acc=f"{batch_accuracy:.2f}%"
                )
                metrics_to_log = {
                    "global_step": current_global_step,
                    f"{log_prefix}/Train_Loss_Batch": avg_loss_batch,
                    f"{log_prefix}/Train_Acc_Batch": batch_accuracy,
                }
                if not torch.isnan(torch.tensor(current_mem_used)):
                    metrics_to_log[f"{log_prefix}/GPU_Mem_Used_MiB_Batch"] = (
                        current_mem_used
                    )
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)

        final_avg_epoch_loss = (
            epoch_loss / epoch_samples if epoch_samples > 0 else float("nan")
        )
        final_avg_epoch_accuracy = (
            (epoch_correct / epoch_samples) * 100.0
            if epoch_samples > 0
            else float("nan")
        )
        peak_mem_predictor_train = max(
            peak_mem_predictor_train, peak_mem_predictor_epoch
        )

        logger.info(
            f"{log_prefix} Epoch {epoch + 1}/{epochs} - Train Loss: "
            f"{final_avg_epoch_loss:.4f}, Train Acc: {final_avg_epoch_accuracy:.2f}%, "
            f"Peak Mem Epoch: {peak_mem_predictor_epoch:.1f} MiB"
        )
        epoch_summary_metrics = {
            "global_step": step_ref[0],
            f"{log_prefix}/Train_Loss_EpochAvg": final_avg_epoch_loss,
            f"{log_prefix}/Train_Acc_EpochAvg": final_avg_epoch_accuracy,
            f"{log_prefix}/Peak_GPU_Mem_Epoch_MiB": peak_mem_predictor_epoch,
        }
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)

        if es_enabled:
            val_loss, val_acc = evaluate_cafo_predictor(
                block, predictor, val_loader, device, get_block_input_fn, criterion
            )
            logger.info(
                f"{log_prefix} Epoch {epoch + 1}/{epochs} - Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.2f}%"
            )
            val_metrics = {
                "global_step": step_ref[0],
                f"{log_prefix}/Val_Loss_Epoch": val_loss,
                f"{log_prefix}/Val_Acc_Epoch": val_acc,
            }
            log_metrics(val_metrics, wandb_run=wandb_run, commit=True)

            current_es_metric_value = val_acc if "acc" in es_metric_name else val_loss

            if torch.isnan(torch.tensor(current_es_metric_value)):
                logger.warning(
                    f"{log_prefix} Epoch {epoch + 1}: Early stopping metric "
                    f"'{es_metric_name}' is NaN. Treating as no improvement."
                )
                epochs_no_improve += 1
            else:
                improved = False
                if es_mode == "min":
                    improved = (
                        current_es_metric_value < best_es_metric_value - es_min_delta
                    )
                else:
                    improved = (
                        current_es_metric_value > best_es_metric_value + es_min_delta
                    )

                if improved:
                    best_es_metric_value = current_es_metric_value
                    epochs_no_improve = 0
                    logger.debug(
                        f"{log_prefix} Epoch {epoch + 1}: Early stopping metric improved "
                        f"to {best_es_metric_value:.4f}. Reset patience."
                    )
                else:
                    epochs_no_improve += 1
                    logger.debug(
                        f"{log_prefix} Epoch {epoch + 1}: Early stopping metric did not "
                        f"improve. Patience: {epochs_no_improve}/{es_patience}."
                    )

            if epochs_no_improve >= es_patience:
                logger.info(
                    f"{log_prefix}: Early Stopping Triggered at Epoch {epoch + 1}!"
                )
                logger.info(
                    f"  Metric '{es_metric_name}' did not improve for {es_patience} "
                    f"epochs (Best: {best_es_metric_value:.4f})."
                )
                break

    if nvml_active and gpu_handle:
        mem_info = get_gpu_memory_usage(gpu_handle)
        if mem_info:
            peak_mem_predictor_train = max(peak_mem_predictor_train, mem_info[0])

    logger.info(
        f"Finished CaFo training for {log_prefix} after {epochs_trained} epochs. "
        f"Overall Peak Mem Predictor: {peak_mem_predictor_train:.1f} MiB"
    )
    predictor.eval()
    return (
        final_avg_epoch_loss,
        final_avg_epoch_accuracy,
        peak_mem_predictor_train,
        epochs_trained,
    )


def train_cafo_model(
    model: CaFo_CNN,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    input_adapter: Optional[Callable] = None,
    step_ref: Optional[List[int]] = None,
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float:
    """Orchestrates the training of CaFo_CNN.

    Optionally trains blocks using DFA first, then trains predictors layer-wise.
    Returns the overall peak GPU memory observed across all training phases.
    """
    if step_ref is None:
        step_ref = [-1]

    model.to(device)
    algo_config = config.get("algorithm_params", {})
    train_blocks_flag = algo_config.get("train_blocks", False)
    num_blocks = len(model.blocks)
    peak_mem_train = 0.0

    # --- Optional: Block Training Phase (DFA) ---
    if train_blocks_flag:
        peak_mem_block_phase = train_cafo_dfa_blocks(
            model=model,
            train_loader=train_loader,
            config=config,
            device=device,
            wandb_run=wandb_run,
            step_ref=step_ref,
            gpu_handle=gpu_handle,
            nvml_active=nvml_active,
        )
        peak_mem_train = max(peak_mem_train, peak_mem_block_phase)
    else:
        logger.info("Skipping block training phase. Blocks remain frozen.")
        model.eval()
        for p in model.blocks.parameters():
            p.requires_grad_(False)

    logger.info(
        f"--- Starting CaFo Predictor Training Phase for {num_blocks} blocks ---"
    )
    if input_adapter:
        logger.warning("CaFo Training: 'input_adapter' ignored for CNNs.")

    predictor_optimizer_name = algo_config.get("predictor_optimizer_type", "Adam")
    predictor_lr = algo_config.get("predictor_lr", 0.001)
    predictor_weight_decay = algo_config.get("predictor_weight_decay", 0.0)
    criterion_name = algo_config.get("loss_type", "CrossEntropyLoss")
    epochs_per_block = algo_config.get("num_epochs_per_block", 10)
    log_interval = algo_config.get("log_interval", 100)
    optimizer_params_extra = algo_config.get("optimizer_params", {})
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None)

    predictor_es_config = {
        "enabled": algo_config.get("predictor_early_stopping_enabled", True),
        "metric": algo_config.get("predictor_early_stopping_metric", "val_loss"),
        "patience": algo_config.get("predictor_early_stopping_patience", 10),
        "mode": algo_config.get("predictor_early_stopping_mode", "min"),
        "min_delta": algo_config.get("predictor_early_stopping_min_delta", 0.0),
    }

    if criterion_name.lower() == "crossentropyloss":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion for CaFo Predictors: {criterion_name}")

    predictors = nn.ModuleList()
    for i in range(num_blocks):
        try:
            in_features = model.get_predictor_input_dim(i)
            predictor = CaFoPredictor(in_features, model.num_classes).to(device)
            predictors.append(predictor)
            logger.info(f"Created predictor {i + 1} with input dim {in_features}")
        except Exception as e:
            logger.error(
                f"Failed to create predictor for block {i}: {e}", exc_info=True
            )
            raise RuntimeError("Predictor creation failed.") from e

    def initial_input_fn(img: torch.Tensor) -> torch.Tensor:
        return img.to(device)

    current_block_input_fn = initial_input_fn
    peak_mem_predictor_phase_overall = 0.0
    total_epochs_trained_all_predictors = 0

    for i in range(num_blocks):
        predictor_log_idx = i + 1
        log_prefix = f"Predictor_{predictor_log_idx}"
        logger.info(f"--- Training {log_prefix} ---")
        block = model.blocks[i]
        block.eval()
        predictor = predictors[i]
        for p in predictor.parameters():
            p.requires_grad_(True)
        params_to_optimize = list(predictor.parameters())

        predictor_peak_mem_this = 0.0
        final_avg_loss = float("nan")
        final_avg_acc = float("nan")
        epochs_trained_this_predictor = 0

        if not params_to_optimize:
            logger.warning(
                f"{log_prefix} has no parameters requiring gradients. Skipping training."
            )
        else:
            optimizer_kwargs = {
                "lr": predictor_lr,
                "weight_decay": predictor_weight_decay,
                **optimizer_params_extra,
            }
            optimizer = getattr(optim, predictor_optimizer_name)(
                params_to_optimize, **optimizer_kwargs
            )

            (
                final_avg_loss,
                final_avg_acc,
                predictor_peak_mem_this,
                epochs_trained_this_predictor,
            ) = train_cafo_predictor_only(
                block=block,
                predictor=predictor,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs_per_block,
                device=device,
                get_block_input_fn=current_block_input_fn,
                early_stopping_config=predictor_es_config,
                wandb_run=wandb_run,
                log_interval=log_interval,
                block_index=i,
                step_ref=step_ref,
                gpu_handle=gpu_handle,
                nvml_active=nvml_active,
            )
            total_epochs_trained_all_predictors += epochs_trained_this_predictor
            peak_mem_predictor_phase_overall = max(
                peak_mem_predictor_phase_overall, predictor_peak_mem_this
            )
            for p in predictor.parameters():
                p.requires_grad_(False)
            predictor.eval()

        current_global_step = step_ref[0]
        predictor_summary_metrics = {
            "global_step": current_global_step,
            f"{log_prefix}/Train_Loss_LayerAvg": final_avg_loss,
            f"{log_prefix}/Train_Acc_LayerAvg": final_avg_acc,
            f"{log_prefix}/Peak_GPU_Mem_Layer_MiB": predictor_peak_mem_this,
            f"{log_prefix}/Epochs_Trained": epochs_trained_this_predictor,
        }
        log_metrics(predictor_summary_metrics, wandb_run=wandb_run, commit=True)
        logger.debug(
            f"Logged CaFo {log_prefix} summary at global_step {current_global_step}"
        )

        if checkpoint_dir and params_to_optimize:
            create_directory_if_not_exists(checkpoint_dir)
            chkpt_filename = f"cafo_predictor_{i}_complete.pth"
            state_to_save = {
                "state_dict": predictor.state_dict(),
                "predictor_index": i,
                "epochs_trained": epochs_trained_this_predictor,
            }
            save_checkpoint(
                state=state_to_save,
                is_best=False,
                filename=chkpt_filename,
                checkpoint_dir=checkpoint_dir,
            )

        # Prepare input function for next predictor
        def create_next_input_fn(
            trained_block_idx: int, previous_input_fn: Callable
        ) -> Callable[[torch.Tensor], torch.Tensor]:
            block_k = model.blocks[trained_block_idx]
            block_k.eval()
            block_k.to(device)

            @torch.no_grad()
            def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                block_input = previous_input_fn(img_batch).to(device)
                block_output = block_k(block_input)
                return block_output.detach()

            return next_input_fn

        current_block_input_fn = create_next_input_fn(i, current_block_input_fn)

    logger.info(
        f"Finished all layer-wise CaFo predictor training. Total Epochs Trained "
        f"(Sum): {total_epochs_trained_all_predictors}"
    )
    model.trained_predictors = predictors  # Attach predictors for evaluation
    peak_mem_train = max(peak_mem_train, peak_mem_predictor_phase_overall)

    return peak_mem_train


def evaluate_cafo_model(
    model: CaFo_CNN,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    predictors: Optional[nn.ModuleList] = None,
    aggregation_method: str = "sum",
    input_adapter: Optional[Callable] = None,
) -> Dict[str, float]:
    """Evaluates the CaFo model by aggregating predictor outputs."""
    model.eval()
    model.to(device)

    if predictors is None:
        if hasattr(model, "trained_predictors") and isinstance(
            model.trained_predictors, nn.ModuleList
        ):
            predictors = model.trained_predictors
        else:
            raise ValueError("Trained predictors are required for CaFo evaluation.")
    if not predictors:
        raise ValueError("Predictor list is empty or None.")

    predictors.to(device)
    predictors.eval()

    num_predictors = len(predictors)
    num_blocks = len(model.blocks)
    if num_predictors != num_blocks:
        logger.warning(
            f"Num predictors ({num_predictors}) != num blocks ({num_blocks})."
        )
    total_loss, total_correct, total_samples = 0.0, 0, 0
    aggregation_method = aggregation_method.lower()
    logger.info(
        f"Evaluating CaFo model using {num_predictors} predictors "
        f"('{aggregation_method}' aggregation)."
    )
    with torch.no_grad():
        pbar_desc = f"Evaluating CaFo ({aggregation_method})"
        pbar = tqdm(data_loader, desc=pbar_desc, leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            adapted_images = images
            block_outputs = model.forward(adapted_images)
            predictor_outputs = []

            if aggregation_method == "last":
                can_eval_last = (
                    num_blocks > 0
                    and len(block_outputs) == num_blocks
                    and num_predictors == num_blocks
                )
                if can_eval_last:
                    last_block_idx = num_blocks - 1
                    try:
                        pred_out = predictors[last_block_idx](
                            block_outputs[last_block_idx]
                        )
                        predictor_outputs.append(pred_out)
                    except Exception as e_pred:
                        logger.error(
                            f"Error LAST predictor ({last_block_idx}): {e_pred}",
                            exc_info=True,
                        )
                        return {
                            "eval_accuracy": float("nan"),
                            "eval_loss": float("nan"),
                        }
                else:
                    logger.error("Cannot eval 'last': mismatch counts/outputs.")
                    return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
            else:
                for i, block_out in enumerate(block_outputs):
                    if i < len(predictors):
                        try:
                            pred_out = predictors[i](block_out)
                            predictor_outputs.append(pred_out)
                        except Exception as e_pred:
                            logger.error(
                                f"Error predictor {i}: {e_pred}", exc_info=True
                            )
                            return {
                                "eval_accuracy": float("nan"),
                                "eval_loss": float("nan"),
                            }

            if not predictor_outputs:
                logger.error("No predictor outputs.")
                return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}

            try:
                if aggregation_method == "sum":
                    final_prediction_logits = torch.stack(predictor_outputs, dim=0).sum(
                        dim=0
                    )
                elif aggregation_method == "last":
                    final_prediction_logits = predictor_outputs[0]
                elif aggregation_method == "average":
                    final_prediction_logits = torch.stack(
                        predictor_outputs, dim=0
                    ).mean(dim=0)
                else:
                    raise ValueError(
                        f"Unsupported aggregation method: {aggregation_method}"
                    )

                if criterion:
                    loss = criterion(final_prediction_logits, labels)
                    total_loss += loss.item() * adapted_images.size(0)

                predicted_labels = torch.argmax(final_prediction_logits, dim=1)
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)
            except Exception as e_agg:
                logger.error(f"Error during aggregation/loss: {e_agg}", exc_info=True)
                return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}

    avg_loss = (
        total_loss / total_samples if criterion and total_samples > 0 else float("nan")
    )
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

    log_msg = f"Eval Results (Agg: {aggregation_method}): Accuracy: {accuracy:.2f}%"
    if criterion and not torch.isnan(torch.tensor(avg_loss)):
        log_msg += f", Loss: {avg_loss:.4f}"
    logger.info(log_msg)

    return {"eval_accuracy": accuracy, "eval_loss": avg_loss}
