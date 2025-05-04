# File: ./src/algorithms/mf.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import logging
from tqdm import tqdm
import pynvml # Import for type hint
from typing import Dict, Any, Optional, Callable, List, Tuple
import os
import time

from src.architectures.mf_mlp import MF_MLP
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint, format_time, create_directory_if_not_exists
from src.utils.monitoring import get_gpu_memory_usage # Import memory usage function

logger = logging.getLogger(__name__)

# --- mf_local_loss_fn (No changes needed) ---
def mf_local_loss_fn(
    activation_i: torch.Tensor,
    projection_matrix_i: nn.Parameter,
    targets: torch.Tensor,
    criterion: nn.Module = nn.CrossEntropyLoss(),
) -> torch.Tensor:
    """Calculates the Mono-Forward local cross-entropy loss for activation a_i using M_i."""
    goodness_scores_i = torch.matmul(activation_i, projection_matrix_i.t())
    loss = criterion(goodness_scores_i, targets)
    return loss


# --- NEW: evaluate_mf_local_loss ---
@torch.no_grad()
def evaluate_mf_local_loss(
    model: MF_MLP, # Model reference (eval mode)
    matrix_index: int, # Index of M_i (0 to L)
    criterion: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    get_matrix_input_fn: Callable[[torch.Tensor], torch.Tensor], # Function to get a_i
    log_prefix: str = "Layer", # Prefix for logging
) -> float:
    """
    Evaluates the local loss for a specific MF layer (using M_i) on a validation set.
    """
    if matrix_index < 0 or matrix_index >= len(model.projection_matrices):
        logger.error(f"{log_prefix} Eval: Matrix index {matrix_index} out of bounds.")
        return float('nan')

    model.eval() # Ensure model is in eval mode
    model.to(device)
    projection_matrix = model.get_projection_matrix(matrix_index).to(device)

    total_loss = 0.0
    total_samples = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        # Use the provided function to get the correct activation (a_i)
        activation_a_i = get_matrix_input_fn(images) # Already detached from training graph

        # Calculate loss for this batch
        batch_loss = mf_local_loss_fn(activation_a_i, projection_matrix, labels, criterion)
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
    return avg_loss


# --- train_mf_matrix_only (MODIFIED - Added Early Stopping Logic) ---
def train_mf_matrix_only(
    model: MF_MLP,
    matrix_index: int,
    optimizer: optim.Optimizer, # Optimizer should *only* contain M_i
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader], # <<< ADDED >>>
    epochs: int,
    device: torch.device,
    get_matrix_input_fn: Callable[[torch.Tensor], torch.Tensor], # Function to get a_i
    early_stopping_config: Dict[str, Any], # <<< ADDED >>>
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> Tuple[float, float, int]: # Return avg loss, peak memory, epochs trained
    """
    Trains a single projection matrix (M_i) using local loss.
    <<< MODIFIED: Added early stopping based on validation local loss. >>>
    """
    log_prefix = f"Layer_M{matrix_index}"
    if matrix_index < 0 or matrix_index >= len(model.projection_matrices):
         raise IndexError(f"Matrix index {matrix_index} out of bounds.")

    projection_matrix = model.get_projection_matrix(matrix_index)
    if not projection_matrix.requires_grad:
        logger.error(f"{log_prefix} requires_grad is False. Check orchestrator.")
        return float('nan'), 0.0, 0 # Return NaN loss, 0 peak memory, 0 epochs

    # Keep the rest of the model in eval mode
    model.to(device); model.eval()

    logger.info(f"--- Starting MF training for {log_prefix} ---")

    peak_mem_matrix_train = 0.0
    final_avg_epoch_loss = float('nan')
    epochs_trained = 0

    # <<< Early Stopping Setup >>>
    es_enabled = early_stopping_config.get("mf_early_stopping_enabled", False)
    es_metric_name = "val_local_loss" # Hardcoded for MF
    es_patience = early_stopping_config.get("mf_early_stopping_patience", 10)
    es_mode = "min" # Always minimize loss
    es_min_delta = early_stopping_config.get("mf_early_stopping_min_delta", 0.0)
    epochs_no_improve = 0
    best_es_metric_value = float('inf')

    if es_enabled:
        if val_loader is None:
            logger.warning(f"{log_prefix}: Early stopping enabled but no val_loader. Disabling.")
            es_enabled = False
        else:
            logger.info(f"{log_prefix}: Early Stopping Enabled - Patience: {es_patience}, MinDelta: {es_min_delta}")
    else:
        logger.info(f"{log_prefix}: Early Stopping Disabled.")
    # <<< End Early Stopping Setup >>>

    for epoch in range(epochs):
        epochs_trained = epoch + 1 # Track how many epochs actually ran
        epoch_loss = 0.0; epoch_samples = 0; peak_mem_matrix_epoch = 0.0
        # <<< Ensure M_i is in train mode >>>
        projection_matrix.requires_grad_(True) # Ensure it's trainable for this epoch

        pbar = tqdm(train_loader, desc=f"{log_prefix} Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1; current_global_step = step_ref[0]
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad(): activation_a_i = get_matrix_input_fn(images) # Already detached

            loss = mf_local_loss_fn(activation_a_i, projection_matrix, labels, criterion)

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

        # --- Early Stopping Check ---
        if es_enabled:
            # <<< Ensure M_i is eval mode for evaluation >>>
            projection_matrix.requires_grad_(False)
            val_loss = evaluate_mf_local_loss(model, matrix_index, criterion, val_loader, device, get_matrix_input_fn, log_prefix)
            logger.info(f"{log_prefix} Epoch {epoch+1}/{epochs} - Val Local Loss: {val_loss:.6f}")
            val_metrics = {"global_step": step_ref[0], f"{log_prefix}/Val_LocalLoss_Epoch": val_loss}
            log_metrics(val_metrics, wandb_run=wandb_run, commit=True)

            if torch.isnan(torch.tensor(val_loss)):
                 logger.warning(f"{log_prefix} Epoch {epoch+1}: Early stopping metric is NaN. Treating as no improvement.")
                 epochs_no_improve += 1
            else:
                if val_loss < best_es_metric_value - es_min_delta: # Always minimize loss
                    best_es_metric_value = val_loss
                    epochs_no_improve = 0
                    logger.debug(f"{log_prefix} Epoch {epoch+1}: ES metric improved to {best_es_metric_value:.6f}. Reset patience.")
                else:
                    epochs_no_improve += 1
                    logger.debug(f"{log_prefix} Epoch {epoch+1}: ES metric did not improve. Patience: {epochs_no_improve}/{es_patience}.")

            if epochs_no_improve >= es_patience:
                logger.info(f"--- {log_prefix}: Early Stopping Triggered at Epoch {epoch+1}! ---")
                logger.info(f"  Val Local Loss did not improve for {es_patience} epochs (Best: {best_es_metric_value:.6f}).")
                break # Exit the training loop for this matrix
        # --- End Early Stopping Check ---

    if nvml_active and gpu_handle:
        mem_info = get_gpu_memory_usage(gpu_handle)
        if mem_info: peak_mem_matrix_train = max(peak_mem_matrix_train, mem_info[0])

    # <<< Ensure M_i is frozen after training loop finishes >>>
    projection_matrix.requires_grad_(False)
    logger.info(f"--- Finished MF training for {log_prefix} after {epochs_trained} epochs. Peak Mem: {peak_mem_matrix_train:.1f} MiB ---")
    return final_avg_epoch_loss, peak_mem_matrix_train, epochs_trained # <<< Return epochs_trained >>>


# --- train_mf_layer (MODIFIED - Added Early Stopping Logic) ---
def train_mf_layer(
    model: MF_MLP,
    layer_index: int,  # Index i=0..L-1. Trains W_{i+1} and M_{i+1}.
    optimizer: optim.Optimizer, # Optimizer should *only* contain W_{i+1} and M_{i+1}
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader], # <<< ADDED >>>
    epochs: int,
    device: torch.device,
    get_layer_input_fn: Callable[[torch.Tensor], torch.Tensor],  # Function to get a_i
    early_stopping_config: Dict[str, Any], # <<< ADDED >>>
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> Tuple[float, float, int]: # Return avg loss, peak memory, epochs trained
    """
    Trains a single layer 'i+1' (W_{i+1} and M_{i+1}) of an MF_MLP using local loss.
    <<< MODIFIED: Added early stopping based on validation local loss. >>>
    """
    if not (0 <= layer_index < model.num_hidden_layers): raise IndexError(f"Layer index {layer_index} out of bounds.")

    w_layer_log_idx = layer_index + 1
    m_matrix_log_idx = layer_index + 1
    log_prefix = f"Layer_W{w_layer_log_idx}_M{m_matrix_log_idx}"

    linear_layer = model.layers[layer_index * 2]  # W_{i+1}
    act_layer = model.layers[layer_index * 2 + 1]  # sigma_{i+1}
    projection_matrix = model.get_projection_matrix(layer_index + 1)  # M_{i+1}

    params_require_grad = [p.requires_grad for p in linear_layer.parameters()] + [projection_matrix.requires_grad]
    if not any(params_require_grad):
        logger.error(f"{log_prefix} requires_grad is False. Check orchestrator.")
        return float('nan'), 0.0, 0

    # Set relevant layers to train mode, keep rest in eval
    model.to(device); model.eval()
    linear_layer.train(); act_layer.train() # Train mode for layers being optimized
    projection_matrix.requires_grad_(True) # Ensure M is trainable

    logger.info(f"--- Starting MF training for {log_prefix} ---")

    peak_mem_layer_train = 0.0
    final_avg_epoch_loss = float('nan')
    epochs_trained = 0

    # <<< Early Stopping Setup >>>
    es_enabled = early_stopping_config.get("mf_early_stopping_enabled", False)
    es_metric_name = "val_local_loss" # Hardcoded for MF
    es_patience = early_stopping_config.get("mf_early_stopping_patience", 10)
    es_mode = "min" # Always minimize loss
    es_min_delta = early_stopping_config.get("mf_early_stopping_min_delta", 0.0)
    epochs_no_improve = 0
    best_es_metric_value = float('inf')

    if es_enabled:
        if val_loader is None: logger.warning(f"{log_prefix}: Early stopping enabled but no val_loader. Disabling."); es_enabled = False
        else: logger.info(f"{log_prefix}: Early Stopping Enabled - Patience: {es_patience}, MinDelta: {es_min_delta}")
    else: logger.info(f"{log_prefix}: Early Stopping Disabled.")
    # <<< End Early Stopping Setup >>>

    # Function to get activation a_{i+1} for validation loss calculation
    @torch.no_grad()
    def get_a_next_validation(img_batch: torch.Tensor) -> torch.Tensor:
        model.eval() # Ensure model is in eval mode
        linear_layer.eval(); act_layer.eval() # Ensure specific layers are eval
        prev_activation_a_i = get_layer_input_fn(img_batch).to(device)
        activation_a_next = act_layer(linear_layer(prev_activation_a_i))
        return activation_a_next.detach()

    for epoch in range(epochs):
        epochs_trained = epoch + 1
        epoch_loss = 0.0; epoch_samples = 0; peak_mem_layer_epoch = 0.0
        # Ensure layers are in train mode for this epoch
        linear_layer.train(); act_layer.train(); projection_matrix.requires_grad_(True)

        pbar = tqdm(train_loader, desc=f"{log_prefix} Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1; current_global_step = step_ref[0]
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad(): prev_activation_a_i = get_layer_input_fn(images) # Already detached

            pre_activation_z_next = linear_layer(prev_activation_a_i) # z_{i+1}
            activation_a_next = act_layer(pre_activation_z_next) # a_{i+1}
            loss = mf_local_loss_fn(activation_a_next, projection_matrix, labels, criterion)

            if torch.isnan(loss) or torch.isinf(loss): logger.error(f"NaN/Inf loss at {log_prefix}, Epoch {epoch+1}, Batch {batch_idx}. Stop."); break

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

        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)): logger.error(f"Terminating {log_prefix} training due to invalid loss."); break

        final_avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float('nan')
        peak_mem_layer_train = max(peak_mem_layer_train, peak_mem_layer_epoch)
        logger.info(f"{log_prefix} Epoch {epoch+1}/{epochs} - Train Loss: {final_avg_epoch_loss:.6f}, Peak Mem: {peak_mem_layer_epoch:.1f} MiB")
        epoch_summary_metrics = {"global_step": step_ref[0], f"{log_prefix}/Train_Loss_EpochAvg": final_avg_epoch_loss, f"{log_prefix}/Peak_GPU_Mem_Epoch_MiB": peak_mem_layer_epoch}
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)

        # --- Early Stopping Check ---
        if es_enabled:
            # <<< Evaluate using the local loss for THIS layer (M_{i+1}) >>>
            # Ensure model/layers are in eval mode for evaluation
            model.eval(); linear_layer.eval(); act_layer.eval(); projection_matrix.requires_grad_(False)
            val_loss = evaluate_mf_local_loss(model, layer_index + 1, criterion, val_loader, device, get_a_next_validation, log_prefix)
            logger.info(f"{log_prefix} Epoch {epoch+1}/{epochs} - Val Local Loss: {val_loss:.6f}")
            val_metrics = {"global_step": step_ref[0], f"{log_prefix}/Val_LocalLoss_Epoch": val_loss}
            log_metrics(val_metrics, wandb_run=wandb_run, commit=True)

            if torch.isnan(torch.tensor(val_loss)):
                 logger.warning(f"{log_prefix} Epoch {epoch+1}: ES metric is NaN. No improvement.")
                 epochs_no_improve += 1
            else:
                if val_loss < best_es_metric_value - es_min_delta:
                    best_es_metric_value = val_loss; epochs_no_improve = 0
                    logger.debug(f"{log_prefix} Epoch {epoch+1}: ES metric improved to {best_es_metric_value:.6f}.")
                else:
                    epochs_no_improve += 1
                    logger.debug(f"{log_prefix} Epoch {epoch+1}: ES did not improve. Patience: {epochs_no_improve}/{es_patience}.")

            if epochs_no_improve >= es_patience:
                logger.info(f"--- {log_prefix}: Early Stopping Triggered at Epoch {epoch+1}! ---")
                logger.info(f"  Val Local Loss did not improve for {es_patience} epochs (Best: {best_es_metric_value:.6f}).")
                break # Exit the training loop for this layer
        # --- End Early Stopping Check ---

    if nvml_active and gpu_handle:
        mem_info = get_gpu_memory_usage(gpu_handle)
        if mem_info: peak_mem_layer_train = max(peak_mem_layer_train, mem_info[0])

    # <<< Ensure layer params are frozen after training >>>
    linear_layer.requires_grad_(False); projection_matrix.requires_grad_(False)
    linear_layer.eval(); act_layer.eval() # Ensure eval mode after training

    logger.info(f"--- Finished MF training for {log_prefix} after {epochs_trained} epochs. Peak Mem: {peak_mem_layer_train:.1f} MiB ---")
    return final_avg_epoch_loss, peak_mem_layer_train, epochs_trained # <<< Return epochs_trained >>>


# --- train_mf_model (MODIFIED - Orchestrates and passes val_loader/ES config) ---
def train_mf_model(
    model: MF_MLP,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader], # <<< ADDED >>>
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float: # Returns overall peak memory
    """
    Orchestrates the layer-wise training of an MF_MLP model.
    <<< MODIFIED: Passes validation loader and early stopping config down. >>>
    """
    model.to(device)
    num_hidden_layers = model.num_hidden_layers
    logger.info(f"Starting layer-wise MF training for M0 and {num_hidden_layers} hidden layers.")

    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    optimizer_params_config = algo_config
    lr = optimizer_params_config.get("lr", 0.001)
    weight_decay = optimizer_params_config.get("weight_decay", 0.0)
    optimizer_extra_kwargs = {}

    epochs_per_layer = algo_config.get("epochs_per_layer", 5)
    log_interval = algo_config.get("log_interval", 100)
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None)
    mf_criterion = nn.CrossEntropyLoss()

    # <<< Get Early Stopping Config for MF Layers >>>
    mf_early_stopping_config = {
        "mf_early_stopping_enabled": algo_config.get("mf_early_stopping_enabled", False),
        # "mf_early_stopping_metric" is implicitly "val_local_loss"
        "mf_early_stopping_patience": algo_config.get("mf_early_stopping_patience", 10),
        "mf_early_stopping_min_delta": algo_config.get("mf_early_stopping_min_delta", 0.0),
    }

    def get_a0_input(img_batch):
        adapted_input = input_adapter(img_batch) if input_adapter else img_batch.view(img_batch.shape[0], -1)
        return adapted_input.to(device)

    peak_mem_train = 0.0
    total_epochs_trained_all_layers = 0 # Track total effective epochs

    # --- Freeze All Parameters Initially ---
    logger.debug("Freezing all model parameters initially."); [p.requires_grad_(False) for p in model.parameters()]; model.eval()

    # --- Train M0 ---
    log_prefix_m0 = "Layer_M0"
    logger.info(f"--- Training {log_prefix_m0} ---")
    projection_matrix_0 = model.get_projection_matrix(0)
    m0_peak_mem = 0.0; final_avg_loss_m0 = float('nan'); epochs_trained_m0 = 0
    projection_matrix_0.requires_grad_(True)
    optimizer_0 = getattr(optim, optimizer_name)([projection_matrix_0], lr=lr, weight_decay=weight_decay, **optimizer_extra_kwargs)

    # <<< Pass val_loader and ES config >>>
    final_avg_loss_m0, m0_peak_mem, epochs_trained_m0 = train_mf_matrix_only(
        model=model, matrix_index=0, optimizer=optimizer_0, criterion=mf_criterion,
        train_loader=train_loader, val_loader=val_loader, epochs=epochs_per_layer, device=device,
        get_matrix_input_fn=get_a0_input, early_stopping_config=mf_early_stopping_config,
        wandb_run=wandb_run, log_interval=log_interval, step_ref=step_ref,
        gpu_handle=gpu_handle, nvml_active=nvml_active
    )
    total_epochs_trained_all_layers += epochs_trained_m0
    peak_mem_train = max(peak_mem_train, m0_peak_mem)
    projection_matrix_0.requires_grad_(False)

    layer_summary_metrics = {
        "global_step": step_ref[0], f"{log_prefix_m0}/Train_Loss_LayerAvg": final_avg_loss_m0,
        f"{log_prefix_m0}/Peak_GPU_Mem_Layer_MiB": m0_peak_mem, f"{log_prefix_m0}/Epochs_Trained": epochs_trained_m0,
    }
    log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=True)
    logger.debug(f"Logged MF {log_prefix_m0} summary at step {step_ref[0]}")

    if checkpoint_dir:
        create_directory_if_not_exists(checkpoint_dir)
        save_checkpoint(state={"state_dict": model.state_dict(), "layer_trained_index": -1, "epochs_trained": epochs_trained_m0}, is_best=False, filename="mf_matrix_M0_complete.pth", checkpoint_dir=checkpoint_dir,)

    current_layer_input_fn = get_a0_input

    # --- Train Hidden Layers (W_i+1, M_i+1) ---
    for i in range(num_hidden_layers):
        w_layer_log_idx = i + 1; m_matrix_log_idx = i + 1
        log_prefix_layer = f"Layer_W{w_layer_log_idx}_M{m_matrix_log_idx}"
        logger.info(f"--- Training {log_prefix_layer} ---")

        linear_layer = model.layers[i * 2]; act_layer = model.layers[i * 2 + 1]; projection_matrix = model.get_projection_matrix(i + 1)
        layer_i_peak_mem = 0.0; final_avg_loss_layer_i = float('nan'); epochs_trained_layer_i = 0

        params_to_optimize = []; [p.requires_grad_(True) for p in linear_layer.parameters()]; params_to_optimize.extend(linear_layer.parameters()); projection_matrix.requires_grad_(True); params_to_optimize.append(projection_matrix)
        optimizer = getattr(optim, optimizer_name)(params_to_optimize, lr=lr, weight_decay=weight_decay, **optimizer_extra_kwargs)

        # <<< Pass val_loader and ES config >>>
        final_avg_loss_layer_i, layer_i_peak_mem, epochs_trained_layer_i = train_mf_layer(
            model=model, layer_index=i, optimizer=optimizer, criterion=mf_criterion,
            train_loader=train_loader, val_loader=val_loader, epochs=epochs_per_layer, device=device,
            get_layer_input_fn=current_layer_input_fn, early_stopping_config=mf_early_stopping_config,
            wandb_run=wandb_run, log_interval=log_interval, step_ref=step_ref,
            gpu_handle=gpu_handle, nvml_active=nvml_active
        )
        total_epochs_trained_all_layers += epochs_trained_layer_i
        peak_mem_train = max(peak_mem_train, layer_i_peak_mem)
        for p in params_to_optimize: p.requires_grad_(False); linear_layer.eval(); act_layer.eval()

        layer_summary_metrics = {
            "global_step": step_ref[0], f"{log_prefix_layer}/Train_Loss_LayerAvg": final_avg_loss_layer_i,
            f"{log_prefix_layer}/Peak_GPU_Mem_Layer_MiB": layer_i_peak_mem, f"{log_prefix_layer}/Epochs_Trained": epochs_trained_layer_i,
        }
        log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=True)
        logger.debug(f"Logged MF {log_prefix_layer} summary at step {step_ref[0]}")

        if checkpoint_dir:
            create_directory_if_not_exists(checkpoint_dir); chkpt_filename = f"mf_layer_{w_layer_log_idx}_complete.pth"
            save_checkpoint(state={"state_dict": model.state_dict(), "layer_trained_index": i, "epochs_trained": epochs_trained_layer_i}, is_best=False, filename=chkpt_filename, checkpoint_dir=checkpoint_dir,)

        def create_next_input_fn(trained_layer_idx: int, previous_input_fn: Callable):
            lin_layer_k, act_layer_k = model.layers[trained_layer_idx*2], model.layers[trained_layer_idx*2+1]
            lin_layer_k.eval(); act_layer_k.eval()
            @torch.no_grad()
            def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                a_prev = previous_input_fn(img_batch); lin_layer_k.to(device); act_layer_k.to(device)
                a_current = act_layer_k(lin_layer_k(a_prev.to(device)))
                return a_current.detach()
            return next_input_fn
        current_layer_input_fn = create_next_input_fn(i, current_layer_input_fn)

    logger.info(f"Finished all layer-wise MF training. Total Epochs Trained (Sum): {total_epochs_trained_all_layers}")
    model.eval()
    return peak_mem_train


# --- evaluate_mf_model (No changes needed) ---
def evaluate_mf_model(
    model: MF_MLP,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None, # Keep signature consistent
    input_adapter: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Evaluates the trained MF_MLP model using the paper's "BP-style" approach.
    Uses the activation from the last hidden layer (a_L) and the last projection matrix (M_L).
    """
    model.eval(); model.to(device)
    total_correct, total_samples = 0, 0
    num_hidden_layers = model.num_hidden_layers
    last_activation_index = num_hidden_layers # a_L is at index L
    last_projection_matrix_index = num_hidden_layers # M_L is at index L

    logger.info(f"Evaluating MF model using activation a_{last_activation_index} and matrix M_{last_projection_matrix_index}.")
    if last_projection_matrix_index >= len(model.projection_matrices):
        logger.error(f"Index M_{last_projection_matrix_index} out of bounds ({len(model.projection_matrices)} matrices exist)."); return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
    last_projection_matrix = model.get_projection_matrix(last_projection_matrix_index)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating MF (BP-style)", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            eval_input = input_adapter(images) if input_adapter else images.view(images.shape[0], -1)
            all_activations = model.forward_with_intermediate_activations(eval_input)
            if len(all_activations) <= last_activation_index: logger.error(f"Activation list len ({len(all_activations)}) too short for a_{last_activation_index}."); continue
            last_hidden_activation = all_activations[last_activation_index].to(device)
            last_projection_matrix = last_projection_matrix.to(device)
            goodness_scores = torch.matmul(last_hidden_activation, last_projection_matrix.t())
            predicted_labels = torch.argmax(goodness_scores, dim=1)
            total_correct += (predicted_labels == labels).sum().item(); total_samples += labels.size(0)
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"MF Evaluation Results (BP-style): Accuracy: {accuracy:.2f}%"); results = {"eval_accuracy": accuracy, "eval_loss": float("nan")}
    return results