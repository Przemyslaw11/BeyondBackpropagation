# File: ./src/algorithms/mf.py (MODIFIED for MLP only)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import logging
from tqdm import tqdm
import pynvml # Import for type hint
from typing import Dict, Any, Optional, Callable, List, Tuple # Removed Union
import os
import time

from src.architectures.mf_mlp import MF_MLP # MF_CNN import removed
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint, format_time, create_directory_if_not_exists
from src.utils.monitoring import get_gpu_memory_usage

logger = logging.getLogger(__name__)

# --- mf_local_loss_fn (Remains the same) ---
def mf_local_loss_fn(
    activation_i: torch.Tensor,
    projection_matrix_i: nn.Parameter,
    targets: torch.Tensor,
    criterion: nn.Module = nn.CrossEntropyLoss(),
) -> torch.Tensor:
    """Calculates the Mono-Forward local cross-entropy loss for activation a_i using M_i."""
    if activation_i.dim() != 2:
        raise ValueError(f"Activation must be flattened (2D) for local loss. Got shape: {activation_i.shape}")
    goodness_scores_i = torch.matmul(activation_i, projection_matrix_i.t())
    loss = criterion(goodness_scores_i, targets)
    return loss

# --- evaluate_mf_local_loss (MODIFIED for MLP) ---
@torch.no_grad()
def evaluate_mf_local_loss(
    model: MF_MLP, # <<< MODIFIED >>> Model is MF_MLP
    matrix_index: int,
    criterion: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    input_adapter: Callable[[torch.Tensor], torch.Tensor], # <<< MODIFIED: No longer Optional for MLP
    log_prefix: str = "Layer",
) -> float:
    """
    Evaluates the local loss for a specific MF layer (using M_i) on a validation set for an MF_MLP model.
    """
    if matrix_index < 0 or matrix_index >= len(model.projection_matrices):
        logger.error(f"{log_prefix} Eval: Matrix index {matrix_index} out of bounds.")
        return float('nan')

    model.eval()
    model.to(device)
    projection_matrix = model.get_projection_matrix(matrix_index).to(device)

    total_loss = 0.0
    total_samples = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        adapted_input = input_adapter(images) # Apply adapter for MLP
        all_activations_flat = model.forward_with_intermediate_activations(adapted_input)
        if len(all_activations_flat) <= matrix_index:
            logger.error(f"{log_prefix} Eval: Activation list too short ({len(all_activations_flat)}) for index {matrix_index}.")
            continue
        activation_a_i_flat = all_activations_flat[matrix_index]

        batch_loss = mf_local_loss_fn(activation_a_i_flat, projection_matrix, labels, criterion)
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
    return avg_loss


# --- train_mf_matrix_only (MODIFIED for MLP) ---
def train_mf_matrix_only(
    model: MF_MLP, # <<< MODIFIED >>>
    matrix_index: int,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    device: torch.device,
    input_adapter: Callable[[torch.Tensor], torch.Tensor], # <<< MODIFIED: No longer Optional for MLP
    early_stopping_config: Dict[str, Any],
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> Tuple[float, float, int]:
    """
    Trains a single projection matrix (M_i) using local loss for an MF_MLP model.
    """
    log_prefix = f"Layer_M{matrix_index}"
    if matrix_index < 0 or matrix_index >= len(model.projection_matrices):
         raise IndexError(f"Matrix index {matrix_index} out of bounds.")

    projection_matrix = model.get_projection_matrix(matrix_index)
    if not projection_matrix.requires_grad: logger.error(f"{log_prefix} requires_grad is False."); return float('nan'), 0.0, 0

    model.to(device); model.eval()
    logger.info(f"--- Starting MF training for {log_prefix} ---")

    peak_mem_matrix_train = 0.0; final_avg_epoch_loss = float('nan'); epochs_trained = 0
    es_enabled = early_stopping_config.get("mf_early_stopping_enabled", False)
    es_patience = early_stopping_config.get("mf_early_stopping_patience", 10)
    es_min_delta = early_stopping_config.get("mf_early_stopping_min_delta", 0.0)
    epochs_no_improve = 0; best_es_metric_value = float('inf')

    if es_enabled:
        if val_loader is None: logger.warning(f"{log_prefix}: ES enabled but no val_loader. Disabling."); es_enabled = False
        else: logger.info(f"{log_prefix}: Early Stopping Enabled - Patience: {es_patience}, MinDelta: {es_min_delta}")
    else: logger.info(f"{log_prefix}: Early Stopping Disabled.")

    for epoch in range(epochs):
        epochs_trained = epoch + 1
        epoch_loss = 0.0; epoch_samples = 0; peak_mem_matrix_epoch = 0.0
        projection_matrix.requires_grad_(True)

        pbar = tqdm(train_loader, desc=f"{log_prefix} Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1; current_global_step = step_ref[0]
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                adapted_input = input_adapter(images)
                all_activations_flat = model.forward_with_intermediate_activations(adapted_input)
                if len(all_activations_flat) <= matrix_index: logger.error(f"{log_prefix} Batch {batch_idx}: Act list too short."); continue
                activation_a_i_flat = all_activations_flat[matrix_index]

            loss = mf_local_loss_fn(activation_a_i_flat, projection_matrix, labels, criterion)
            if torch.isnan(loss) or torch.isinf(loss): logger.error(f"NaN/Inf loss at {log_prefix}, Epoch {epoch+1}, Batch {batch_idx}. Stop."); break
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            batch_size = images.size(0); epoch_loss += loss.item() * batch_size; epoch_samples += batch_size
            current_mem_used = float('nan')
            if nvml_active and gpu_handle and ((batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1):
                 mem_info = get_gpu_memory_usage(gpu_handle)
                 if mem_info: current_mem_used = mem_info[0]; peak_mem_matrix_epoch = max(peak_mem_matrix_epoch, current_mem_used)
            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                metrics_to_log = {"global_step": current_global_step, f"{log_prefix}/Train_Loss_Batch": loss.item()}
                if not torch.isnan(torch.tensor(current_mem_used)): metrics_to_log[f"{log_prefix}/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True); pbar.set_postfix(loss=f"{loss.item():.6f}")

        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)): logger.error(f"Terminating {log_prefix} training due to invalid loss."); break

        final_avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float('nan')
        peak_mem_matrix_train = max(peak_mem_matrix_train, peak_mem_matrix_epoch)
        logger.info(f"{log_prefix} Epoch {epoch+1}/{epochs} - Train Loss: {final_avg_epoch_loss:.6f}, Peak Mem: {peak_mem_matrix_epoch:.1f} MiB")
        epoch_summary_metrics = {"global_step": step_ref[0], f"{log_prefix}/Train_Loss_EpochAvg": final_avg_epoch_loss, f"{log_prefix}/Peak_GPU_Mem_Epoch_MiB": peak_mem_matrix_epoch}
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)

        if es_enabled:
            projection_matrix.requires_grad_(False)
            val_loss = evaluate_mf_local_loss(
                model=model, matrix_index=matrix_index, criterion=criterion,
                val_loader=val_loader, device=device,
                input_adapter=input_adapter, log_prefix=log_prefix
            )
            logger.info(f"{log_prefix} Epoch {epoch+1}/{epochs} - Val Local Loss: {val_loss:.6f}")
            val_metrics = {"global_step": step_ref[0], f"{log_prefix}/Val_LocalLoss_Epoch": val_loss}
            log_metrics(val_metrics, wandb_run=wandb_run, commit=True)
            if torch.isnan(torch.tensor(val_loss)): logger.warning(f"{log_prefix} Epoch {epoch+1}: ES metric NaN."); epochs_no_improve += 1
            else:
                if val_loss < best_es_metric_value - es_min_delta: best_es_metric_value = val_loss; epochs_no_improve = 0; logger.debug(f"{log_prefix} Epoch {epoch+1}: ES metric improved to {best_es_metric_value:.6f}.")
                else: epochs_no_improve += 1; logger.debug(f"{log_prefix} Epoch {epoch+1}: ES did not improve. Patience: {epochs_no_improve}/{es_patience}.")
            if epochs_no_improve >= es_patience: logger.info(f"--- {log_prefix}: Early Stopping Triggered at Epoch {epoch+1}! ---"); break

    if nvml_active and gpu_handle: mem_info = get_gpu_memory_usage(gpu_handle); peak_mem_matrix_train = max(peak_mem_matrix_train, mem_info[0] if mem_info else 0.0)
    projection_matrix.requires_grad_(False)
    logger.info(f"--- Finished MF training for {log_prefix} after {epochs_trained} epochs. Peak Mem: {peak_mem_matrix_train:.1f} MiB ---")
    return final_avg_epoch_loss, peak_mem_matrix_train, epochs_trained


# --- train_mf_model (MODIFIED - Orchestrates MLP) ---
def train_mf_model(
    model: MF_MLP, # <<< MODIFIED >>>
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Callable[[torch.Tensor], torch.Tensor], # <<< MODIFIED: No longer Optional for MLP
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float:
    """
    Orchestrates the layer-wise training of an MF_MLP model.
    Trains M0 first, then trains W_i+1 and M_i+1 together for i=0 to L-1.
    """
    model.to(device)
    num_layers_W = model.num_hidden_layers
    num_matrices_M = len(model.projection_matrices)

    logger.info(f"Starting layer-wise MF training for MLP with {num_layers_W} W-layers and {num_matrices_M} M-matrices.")

    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    lr = algo_config.get("lr", 0.001)
    weight_decay = algo_config.get("weight_decay", 0.0)
    optimizer_extra_kwargs = {}
    epochs_per_layer = algo_config.get("epochs_per_layer", 5)
    log_interval = algo_config.get("log_interval", 100)
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None)
    mf_criterion = nn.CrossEntropyLoss()
    mf_early_stopping_config = {
        "mf_early_stopping_enabled": algo_config.get("mf_early_stopping_enabled", False),
        "mf_early_stopping_patience": algo_config.get("mf_early_stopping_patience", 10),
        "mf_early_stopping_min_delta": algo_config.get("mf_early_stopping_min_delta", 0.0),
    }

    peak_mem_train = 0.0
    total_epochs_trained_all_layers = 0

    logger.debug("Freezing all model parameters initially."); [p.requires_grad_(False) for p in model.parameters()]; model.eval()

    m0_peak_mem, epochs_trained_m0 = 0.0, 0
    if num_matrices_M > 0:
        m0_params = [model.get_projection_matrix(0)]
        model.get_projection_matrix(0).requires_grad_(True)
        m0_optimizer = getattr(optim, optimizer_name)(m0_params, lr=lr, weight_decay=weight_decay, **optimizer_extra_kwargs)
        _, m0_peak_mem, epochs_trained_m0 = train_mf_matrix_only(
            model=model, matrix_index=0, optimizer=m0_optimizer,
            criterion=mf_criterion, train_loader=train_loader, val_loader=val_loader, epochs=epochs_per_layer, device=device,
            input_adapter=input_adapter,
            early_stopping_config=mf_early_stopping_config, wandb_run=wandb_run, log_interval=log_interval, step_ref=step_ref,
            gpu_handle=gpu_handle, nvml_active=nvml_active
        )
        total_epochs_trained_all_layers += epochs_trained_m0
        peak_mem_train = max(peak_mem_train, m0_peak_mem)
        model.get_projection_matrix(0).requires_grad_(False)
        if checkpoint_dir: save_checkpoint(state={"state_dict": model.state_dict(), "layer_trained_index": -1, "epochs_trained": epochs_trained_m0}, is_best=False, filename="mf_matrix_M0_complete.pth", checkpoint_dir=checkpoint_dir,)

    for i in range(num_layers_W):
        w_layer_log_idx = i + 1; m_matrix_log_idx = i + 1
        log_prefix = f"Layer_W{w_layer_log_idx}_M{m_matrix_log_idx}"
        logger.info(f"--- Starting MF training for {log_prefix} ---")

        params_to_optimize = []
        # MLP case
        if i*2 < len(model.layers): # Linear layer W_{i+1}
            linear_layer = model.layers[i*2]
            linear_params = list(linear_layer.parameters())
            for p in linear_params: p.requires_grad_(True)
            params_to_optimize.extend(linear_params)
            linear_layer.train()
            model.layers[i*2+1].train() # Corresponding activation function
        else: logger.error(f"{log_prefix}: Linear layer index {i*2} out of range."); continue

        if m_matrix_log_idx < len(model.projection_matrices): # Projection matrix M_{i+1}
            projection_matrix = model.get_projection_matrix(m_matrix_log_idx)
            projection_matrix.requires_grad_(True)
            params_to_optimize.append(projection_matrix)
        else: logger.error(f"{log_prefix}: Projection matrix index {m_matrix_log_idx} out of range."); continue

        if not params_to_optimize: logger.error(f"{log_prefix}: No parameters to optimize."); continue
        optimizer = getattr(optim, optimizer_name)(params_to_optimize, lr=lr, weight_decay=weight_decay, **optimizer_extra_kwargs)

        peak_mem_layer_train = 0.0; final_avg_epoch_loss = float('nan'); epochs_trained_layer_i = 0
        es_enabled_layer = mf_early_stopping_config.get("mf_early_stopping_enabled", False)
        es_patience_layer = mf_early_stopping_config.get("mf_early_stopping_patience", 10)
        es_min_delta_layer = mf_early_stopping_config.get("mf_early_stopping_min_delta", 0.0)
        epochs_no_improve_layer = 0; best_es_metric_value_layer = float('inf')

        if es_enabled_layer and val_loader is None: logger.warning(f"{log_prefix}: ES enabled but no val_loader."); es_enabled_layer = False

        for epoch in range(epochs_per_layer):
            epochs_trained_layer_i = epoch + 1
            epoch_loss = 0.0; epoch_samples = 0; peak_mem_layer_epoch = 0.0
            model.layers[i*2].train(); model.layers[i*2+1].train() # W_{i+1} and activation
            projection_matrix.requires_grad_(True) # M_{i+1}

            pbar = tqdm(train_loader, desc=f"{log_prefix} Epoch {epoch+1}/{epochs_per_layer}", leave=False)
            for batch_idx, (images, labels) in enumerate(pbar):
                step_ref[0] += 1; current_global_step = step_ref[0]
                images, labels = images.to(device), labels.to(device)

                with torch.no_grad():
                    adapted_input = input_adapter(images)
                    all_activations_flat_prev = model.forward_with_intermediate_activations(adapted_input)
                    if len(all_activations_flat_prev) <= i: logger.error(f"{log_prefix} Batch {batch_idx}: Prev Act list too short."); continue
                    prev_activation = all_activations_flat_prev[i] # This is a_i (flattened)

                # Forward through W_{i+1} and activation (MLP specific)
                pre_activation_z_next = model.layers[i*2](prev_activation) # W_{i+1}(a_i)
                activation_a_next_flat = model.layers[i*2+1](pre_activation_z_next) # sigma(z_{i+1}), already flat

                loss = mf_local_loss_fn(activation_a_next_flat, projection_matrix, labels, mf_criterion)
                if torch.isnan(loss) or torch.isinf(loss): logger.error(f"NaN/Inf loss {log_prefix}, Epoch {epoch+1}, Batch {batch_idx}."); break

                optimizer.zero_grad(); loss.backward(); optimizer.step()

                batch_size = images.size(0); epoch_loss += loss.item() * batch_size; epoch_samples += batch_size
                current_mem_used = float('nan')
                if nvml_active and gpu_handle and ((batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1):
                     mem_info = get_gpu_memory_usage(gpu_handle)
                     if mem_info: current_mem_used = mem_info[0]; peak_mem_layer_epoch = max(peak_mem_layer_epoch, current_mem_used)
                if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                     metrics_to_log = {"global_step": current_global_step, f"{log_prefix}/Train_Loss_Batch": loss.item()}
                     if not torch.isnan(torch.tensor(current_mem_used)): metrics_to_log[f"{log_prefix}/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                     log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True); pbar.set_postfix(loss=f"{loss.item():.6f}")

            if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)): logger.error(f"Terminating {log_prefix} epoch due to invalid loss."); break

            final_avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float('nan')
            peak_mem_layer_train = max(peak_mem_layer_train, peak_mem_layer_epoch)
            logger.info(f"{log_prefix} Epoch {epoch+1}/{epochs_per_layer} - Train Loss: {final_avg_epoch_loss:.6f}, Peak Mem: {peak_mem_layer_epoch:.1f} MiB")
            epoch_summary_metrics = {"global_step": step_ref[0], f"{log_prefix}/Train_Loss_EpochAvg": final_avg_epoch_loss, f"{log_prefix}/Peak_GPU_Mem_Epoch_MiB": peak_mem_layer_epoch}
            log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)

            if es_enabled_layer:
                 model.layers[i*2].eval(); model.layers[i*2+1].eval()
                 projection_matrix.requires_grad_(False)
                 val_loss = evaluate_mf_local_loss(model, m_matrix_log_idx, mf_criterion, val_loader, device, input_adapter, log_prefix)
                 logger.info(f"{log_prefix} Epoch {epoch+1}/{epochs_per_layer} - Val Local Loss: {val_loss:.6f}")
                 val_metrics = {"global_step": step_ref[0], f"{log_prefix}/Val_LocalLoss_Epoch": val_loss}
                 log_metrics(val_metrics, wandb_run=wandb_run, commit=True)
                 if torch.isnan(torch.tensor(val_loss)): epochs_no_improve_layer += 1
                 else:
                      if val_loss < best_es_metric_value_layer - es_min_delta_layer: best_es_metric_value_layer = val_loss; epochs_no_improve_layer = 0; logger.debug(f"{log_prefix} Epoch {epoch+1}: ES improved.")
                      else: epochs_no_improve_layer += 1; logger.debug(f"{log_prefix} Epoch {epoch+1}: ES no improve. Patience: {epochs_no_improve_layer}/{es_patience_layer}.")
                 if epochs_no_improve_layer >= es_patience_layer: logger.info(f"--- {log_prefix}: Early Stopping Triggered at Epoch {epoch+1}! ---"); break

        total_epochs_trained_all_layers += epochs_trained_layer_i
        peak_mem_train = max(peak_mem_train, peak_mem_layer_train)
        for p in params_to_optimize: p.requires_grad_(False)
        model.layers[i*2].eval(); model.layers[i*2+1].eval()

        layer_summary_metrics = {
            "global_step": step_ref[0], f"{log_prefix}/Train_Loss_LayerAvg": final_avg_epoch_loss,
            f"{log_prefix}/Peak_GPU_Mem_Layer_MiB": peak_mem_layer_train, f"{log_prefix}/Epochs_Trained": epochs_trained_layer_i,
        }
        log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=True)
        logger.debug(f"Logged MF {log_prefix} summary at step {step_ref[0]}")

        if checkpoint_dir:
            create_directory_if_not_exists(checkpoint_dir); chkpt_filename = f"mf_layer_{w_layer_log_idx}_complete.pth"
            save_checkpoint(state={"state_dict": model.state_dict(), "layer_trained_index": i, "epochs_trained": epochs_trained_layer_i}, is_best=False, filename=chkpt_filename, checkpoint_dir=checkpoint_dir,)

    logger.info(f"Finished all layer-wise MF training. Total Epochs Trained (Sum): {total_epochs_trained_all_layers}")
    model.eval()
    return peak_mem_train


# --- evaluate_mf_model (MODIFIED for MLP) ---
def evaluate_mf_model(
    model: MF_MLP, # <<< MODIFIED >>>
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    input_adapter: Callable[[torch.Tensor], torch.Tensor], # <<< MODIFIED: No longer Optional for MLP
) -> Dict[str, float]:
    """
    Evaluates the trained MF_MLP model using the paper's "BP-style" approach.
    Uses the activation from the last layer (a_L) and the last projection matrix (M_L).
    """
    model.eval(); model.to(device)
    total_correct, total_samples = 0, 0

    num_layers = model.num_hidden_layers
    last_activation_index = num_layers
    last_projection_matrix_index = num_layers

    logger.info(f"Evaluating MF (MLP) using a_{last_activation_index} and M_{last_projection_matrix_index}.")
    if last_projection_matrix_index >= len(model.projection_matrices):
        logger.error(f"Index M_{last_projection_matrix_index} out of bounds ({len(model.projection_matrices)} matrices)."); return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
    last_projection_matrix = model.get_projection_matrix(last_projection_matrix_index)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Evaluating MF MLP (BP-style)", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            eval_input = input_adapter(images) # Apply adapter for MLP
            all_activations_flat = model.forward_with_intermediate_activations(eval_input)
            if len(all_activations_flat) <= last_activation_index: logger.error(f"Activation list len ({len(all_activations_flat)}) too short for a_{last_activation_index}."); continue
            last_activation_flat = all_activations_flat[last_activation_index].to(device)
            last_projection_matrix = last_projection_matrix.to(device)
            goodness_scores = torch.matmul(last_activation_flat, last_projection_matrix.t())
            predicted_labels = torch.argmax(goodness_scores, dim=1)
            total_correct += (predicted_labels == labels).sum().item(); total_samples += labels.size(0)
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"MF Evaluation Results (MLP, BP-style): Accuracy: {accuracy:.2f}%"); results = {"eval_accuracy": accuracy, "eval_loss": float("nan")} # Criterion not used for loss here
    return results