# File: ./src/algorithms/mf.py (MODIFIED for CNN)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import logging
from tqdm import tqdm
import pynvml # Import for type hint
from typing import Dict, Any, Optional, Callable, List, Tuple, Union # Add Union
import os
import time

# <<< MODIFIED >>> Import both MLP and CNN
from src.architectures.mf_mlp import MF_MLP
from src.architectures.mf_cnn import MF_CNN
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint, format_time, create_directory_if_not_exists
from src.utils.monitoring import get_gpu_memory_usage # Import memory usage function

logger = logging.getLogger(__name__)

# --- mf_local_loss_fn (Remains the same) ---
def mf_local_loss_fn(
    activation_i: torch.Tensor, # Should be flattened for matrix multiply
    projection_matrix_i: nn.Parameter,
    targets: torch.Tensor,
    criterion: nn.Module = nn.CrossEntropyLoss(),
) -> torch.Tensor:
    """Calculates the Mono-Forward local cross-entropy loss for activation a_i using M_i."""
    # Ensure activation_i is 2D (batch_size, features_flat)
    if activation_i.dim() != 2:
        raise ValueError(f"Activation must be flattened (2D) for local loss. Got shape: {activation_i.shape}")
    goodness_scores_i = torch.matmul(activation_i, projection_matrix_i.t())
    loss = criterion(goodness_scores_i, targets)
    return loss

# --- evaluate_mf_local_loss (MODIFIED for CNN/MLP) ---
@torch.no_grad()
def evaluate_mf_local_loss(
    model: Union[MF_MLP, MF_CNN], # <<< MODIFIED >>> Model can be MLP or CNN
    matrix_index: int, # Index of M_i (0 to L)
    criterion: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    # <<< MODIFIED: Removed get_matrix_input_fn, get activation inside >>>
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]],
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

        # <<< MODIFIED: Get activations internally >>>
        adapted_input = input_adapter(images) if input_adapter else images # Apply adapter if MLP
        all_activations_flat = model.forward_with_intermediate_activations(adapted_input)
        if len(all_activations_flat) <= matrix_index:
            logger.error(f"{log_prefix} Eval: Activation list too short ({len(all_activations_flat)}) for index {matrix_index}.")
            continue # Skip batch
        activation_a_i_flat = all_activations_flat[matrix_index] # Get the i-th FLATTENED activation
        # <<< END MODIFICATION >>>

        # Calculate loss for this batch
        batch_loss = mf_local_loss_fn(activation_a_i_flat, projection_matrix, labels, criterion)
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
    return avg_loss


# --- train_mf_matrix_only (MODIFIED for CNN/MLP) ---
def train_mf_matrix_only(
    model: Union[MF_MLP, MF_CNN], # <<< MODIFIED >>>
    matrix_index: int,
    optimizer: optim.Optimizer, # Optimizer should *only* contain M_i
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    device: torch.device,
    # <<< MODIFIED: Removed get_matrix_input_fn >>>
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]],
    early_stopping_config: Dict[str, Any],
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> Tuple[float, float, int]: # Return avg loss, peak memory, epochs trained
    """
    Trains a single projection matrix (M_i) using local loss. Handles MLP/CNN.
    """
    log_prefix = f"Layer_M{matrix_index}"
    if matrix_index < 0 or matrix_index >= len(model.projection_matrices):
         raise IndexError(f"Matrix index {matrix_index} out of bounds.")

    projection_matrix = model.get_projection_matrix(matrix_index)
    if not projection_matrix.requires_grad: logger.error(f"{log_prefix} requires_grad is False."); return float('nan'), 0.0, 0

    model.to(device); model.eval() # Keep base model (W layers) in eval
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
        projection_matrix.requires_grad_(True) # Ensure M_i is trainable

        pbar = tqdm(train_loader, desc=f"{log_prefix} Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1; current_global_step = step_ref[0]
            images, labels = images.to(device), labels.to(device)

            # <<< MODIFIED: Get activation internally >>>
            with torch.no_grad():
                adapted_input = input_adapter(images) if input_adapter else images
                all_activations_flat = model.forward_with_intermediate_activations(adapted_input)
                if len(all_activations_flat) <= matrix_index: logger.error(f"{log_prefix} Batch {batch_idx}: Act list too short."); continue
                activation_a_i_flat = all_activations_flat[matrix_index] # Get the i-th FLATTENED activation
            # <<< END MODIFICATION >>>

            loss = mf_local_loss_fn(activation_a_i_flat, projection_matrix, labels, criterion)
            if torch.isnan(loss) or torch.isinf(loss): logger.error(f"NaN/Inf loss at {log_prefix}, Epoch {epoch+1}, Batch {batch_idx}. Stop."); break
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            batch_size = images.size(0); epoch_loss += loss.item() * batch_size; epoch_samples += batch_size
            # ... (memory monitoring and logging as before) ...
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
        # ... (log epoch summary metrics as before) ...
        epoch_summary_metrics = {"global_step": step_ref[0], f"{log_prefix}/Train_Loss_EpochAvg": final_avg_epoch_loss, f"{log_prefix}/Peak_GPU_Mem_Epoch_MiB": peak_mem_matrix_epoch}
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)


        # --- Early Stopping Check ---
        if es_enabled:
            projection_matrix.requires_grad_(False) # Freeze M_i for eval
            # <<< MODIFIED: Call evaluate_mf_local_loss without get_input_fn >>>
            val_loss = evaluate_mf_local_loss(
                model=model, matrix_index=matrix_index, criterion=criterion,
                val_loader=val_loader, device=device,
                input_adapter=input_adapter, log_prefix=log_prefix
            )
            # <<< END MODIFICATION >>>
            logger.info(f"{log_prefix} Epoch {epoch+1}/{epochs} - Val Local Loss: {val_loss:.6f}")
            val_metrics = {"global_step": step_ref[0], f"{log_prefix}/Val_LocalLoss_Epoch": val_loss}
            log_metrics(val_metrics, wandb_run=wandb_run, commit=True)
            if torch.isnan(torch.tensor(val_loss)): logger.warning(f"{log_prefix} Epoch {epoch+1}: ES metric NaN."); epochs_no_improve += 1
            else:
                if val_loss < best_es_metric_value - es_min_delta: best_es_metric_value = val_loss; epochs_no_improve = 0; logger.debug(f"{log_prefix} Epoch {epoch+1}: ES metric improved to {best_es_metric_value:.6f}.")
                else: epochs_no_improve += 1; logger.debug(f"{log_prefix} Epoch {epoch+1}: ES did not improve. Patience: {epochs_no_improve}/{es_patience}.")
            if epochs_no_improve >= es_patience: logger.info(f"--- {log_prefix}: Early Stopping Triggered at Epoch {epoch+1}! ---"); break
        # --- End Early Stopping Check ---

    # --- Final Cleanup ---
    if nvml_active and gpu_handle: mem_info = get_gpu_memory_usage(gpu_handle); peak_mem_matrix_train = max(peak_mem_matrix_train, mem_info[0] if mem_info else 0.0)
    projection_matrix.requires_grad_(False) # Ensure frozen after loop
    logger.info(f"--- Finished MF training for {log_prefix} after {epochs_trained} epochs. Peak Mem: {peak_mem_matrix_train:.1f} MiB ---")
    return final_avg_epoch_loss, peak_mem_matrix_train, epochs_trained


# --- train_mf_layer (REMOVED - Merged into train_mf_model orchestrator) ---
# The logic to train W_i+1 and M_i+1 together is now directly in train_mf_model


# --- train_mf_model (MODIFIED - Orchestrates CNN/MLP, handles layers directly) ---
def train_mf_model(
    model: Union[MF_MLP, MF_CNN], # <<< MODIFIED >>>
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None, # Still needed for MLP case
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float: # Returns overall peak memory
    """
    Orchestrates the layer-wise training of an MF_MLP or MF_CNN model.
    Trains M0 first, then trains W_i+1 and M_i+1 together for i=0 to L-1.
    """
    model.to(device)
    is_cnn = isinstance(model, MF_CNN)
    num_layers_W = model.num_hidden_layers if isinstance(model, MF_MLP) else model.num_cnn_layers
    num_matrices_M = len(model.projection_matrices) # Should be num_layers_W + 1

    logger.info(f"Starting layer-wise MF training for {'CNN' if is_cnn else 'MLP'} with {num_layers_W} W-layers and {num_matrices_M} M-matrices.")

    # Config extraction (similar to before)
    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    lr = algo_config.get("lr", 0.001)
    weight_decay = algo_config.get("weight_decay", 0.0)
    optimizer_extra_kwargs = {} # Add betas etc. if needed
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

    # --- Freeze All Initially ---
    logger.debug("Freezing all model parameters initially."); [p.requires_grad_(False) for p in model.parameters()]; model.eval()

    # --- Train M0 ---
    m0_peak_mem, epochs_trained_m0 = 0.0, 0
    if num_matrices_M > 0:
        m0_peak_mem, epochs_trained_m0 = train_mf_matrix_only( # Call adapted function
            model=model, matrix_index=0, optimizer=getattr(optim, optimizer_name)([model.get_projection_matrix(0)], lr=lr, weight_decay=weight_decay, **optimizer_extra_kwargs),
            criterion=mf_criterion, train_loader=train_loader, val_loader=val_loader, epochs=epochs_per_layer, device=device,
            input_adapter=input_adapter, # Pass adapter for potential MLP input
            early_stopping_config=mf_early_stopping_config, wandb_run=wandb_run, log_interval=log_interval, step_ref=step_ref,
            gpu_handle=gpu_handle, nvml_active=nvml_active
        )[1:] # Get peak_mem and epochs_trained only
        total_epochs_trained_all_layers += epochs_trained_m0
        peak_mem_train = max(peak_mem_train, m0_peak_mem)
        model.get_projection_matrix(0).requires_grad_(False) # Re-freeze
        if checkpoint_dir: save_checkpoint(state={"state_dict": model.state_dict(), "layer_trained_index": -1, "epochs_trained": epochs_trained_m0}, is_best=False, filename="mf_matrix_M0_complete.pth", checkpoint_dir=checkpoint_dir,)

    # --- Train Layers (W_i+1, M_i+1) together ---
    for i in range(num_layers_W):
        w_layer_log_idx = i + 1; m_matrix_log_idx = i + 1
        log_prefix = f"Layer_W{w_layer_log_idx}_M{m_matrix_log_idx}"
        logger.info(f"--- Starting MF training for {log_prefix} ---")

        # Identify parameters for W_{i+1} and M_{i+1}
        params_to_optimize = []
        if is_cnn: # Get params from the i-th block (which corresponds to W_{i+1})
            if i < len(model.blocks):
                block_params = list(model.blocks[i].parameters())
                for p in block_params: p.requires_grad_(True)
                params_to_optimize.extend(block_params)
                # Set block to train mode
                model.blocks[i].train()
            else: logger.error(f"{log_prefix}: Block index {i} out of range."); continue
        else: # MLP case
            if i*2 < len(model.layers): # Check if linear layer exists
                linear_layer = model.layers[i*2] # W_{i+1}
                linear_params = list(linear_layer.parameters())
                for p in linear_params: p.requires_grad_(True)
                params_to_optimize.extend(linear_params)
                # Set layer to train mode
                linear_layer.train()
                model.layers[i*2+1].train() # Keep activation in train mode too
            else: logger.error(f"{log_prefix}: Linear layer index {i*2} out of range."); continue

        if m_matrix_log_idx < len(model.projection_matrices):
            projection_matrix = model.get_projection_matrix(m_matrix_log_idx)
            projection_matrix.requires_grad_(True)
            params_to_optimize.append(projection_matrix)
        else: logger.error(f"{log_prefix}: Projection matrix index {m_matrix_log_idx} out of range."); continue

        if not params_to_optimize: logger.error(f"{log_prefix}: No parameters to optimize."); continue

        optimizer = getattr(optim, optimizer_name)(params_to_optimize, lr=lr, weight_decay=weight_decay, **optimizer_extra_kwargs)

        # --- Inner Epoch Loop (Replaces train_mf_layer) ---
        peak_mem_layer_train = 0.0; final_avg_epoch_loss = float('nan'); epochs_trained_layer_i = 0
        es_enabled_layer = mf_early_stopping_config.get("mf_early_stopping_enabled", False)
        es_patience_layer = mf_early_stopping_config.get("mf_early_stopping_patience", 10)
        es_min_delta_layer = mf_early_stopping_config.get("mf_early_stopping_min_delta", 0.0)
        epochs_no_improve_layer = 0; best_es_metric_value_layer = float('inf')

        if es_enabled_layer and val_loader is None: logger.warning(f"{log_prefix}: ES enabled but no val_loader."); es_enabled_layer = False

        for epoch in range(epochs_per_layer):
            epochs_trained_layer_i = epoch + 1
            epoch_loss = 0.0; epoch_samples = 0; peak_mem_layer_epoch = 0.0
            # Ensure relevant parts are train mode
            if is_cnn: model.blocks[i].train()
            else: model.layers[i*2].train(); model.layers[i*2+1].train()
            projection_matrix.requires_grad_(True)

            pbar = tqdm(train_loader, desc=f"{log_prefix} Epoch {epoch+1}/{epochs_per_layer}", leave=False)
            for batch_idx, (images, labels) in enumerate(pbar):
                step_ref[0] += 1; current_global_step = step_ref[0]
                images, labels = images.to(device), labels.to(device)

                # Get activation a_{i+1} dynamically
                with torch.no_grad(): # Need previous layer's output detached
                    adapted_input = input_adapter(images) if input_adapter else images
                    all_activations_flat_prev = model.forward_with_intermediate_activations(adapted_input)
                    if len(all_activations_flat_prev) <= i: logger.error(f"{log_prefix} Batch {batch_idx}: Prev Act list too short."); continue
                    prev_activation = all_activations_flat_prev[i] # This is a_i (flattened)
                    # For CNN need unflattened a_i if input adapter is None
                    if is_cnn and input_adapter is None:
                         # Need to reconstruct spatial a_i from flattened a_i - COMPLEX
                         # Alternative: Modify forward_with_intermediate to return BOTH spatial and flat
                         # Easier: Recompute spatial a_i needed by the block
                         # Recompute a_i spatially (input to block i)
                         current_spatial_act = adapted_input
                         for block_idx in range(i):
                              current_spatial_act = model.blocks[block_idx](current_spatial_act)
                         prev_activation_spatial = current_spatial_act.detach() # a_i spatial
                    elif is_cnn: # MLP Adapter applied to CNN? Should not happen
                         logger.error("Inconsistent state: CNN with MLP input adapter?"); continue
                    else: # MLP Case
                         prev_activation_spatial = prev_activation # MLP uses flattened directly

                # Forward through W_{i+1} and activation
                if is_cnn:
                     pre_activation_z_next = model.blocks[i].conv(prev_activation_spatial)
                     activation_a_next = model.blocks[i].relu(pre_activation_z_next)
                     # Need to apply pooling and flatten for loss calculation
                     activation_a_next_pooled = model.blocks[i].pool(activation_a_next)
                     activation_a_next_flat = activation_a_next_pooled.view(activation_a_next_pooled.size(0), -1)
                else: # MLP
                     pre_activation_z_next = model.layers[i*2](prev_activation_spatial) # W_{i+1}(a_i)
                     activation_a_next = model.layers[i*2+1](pre_activation_z_next) # sigma(z_{i+1})
                     activation_a_next_flat = activation_a_next # MLP activation is already flat

                # Calculate Loss using a_{i+1}_flat and M_{i+1}
                loss = mf_local_loss_fn(activation_a_next_flat, projection_matrix, labels, mf_criterion)
                if torch.isnan(loss) or torch.isinf(loss): logger.error(f"NaN/Inf loss {log_prefix}, Epoch {epoch+1}, Batch {batch_idx}."); break

                optimizer.zero_grad(); loss.backward(); optimizer.step()

                batch_size = images.size(0); epoch_loss += loss.item() * batch_size; epoch_samples += batch_size
                # ... (memory monitoring and logging) ...
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

            # --- Early Stopping Check for this layer ---
            if es_enabled_layer:
                 # Ensure relevant parts are eval mode
                 if is_cnn: model.blocks[i].eval()
                 else: model.layers[i*2].eval(); model.layers[i*2+1].eval()
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
            # --- End Early Stopping Check ---
        # --- End Inner Epoch Loop ---

        total_epochs_trained_all_layers += epochs_trained_layer_i
        peak_mem_train = max(peak_mem_train, peak_mem_layer_train)
        # Freeze parameters after training the layer
        for p in params_to_optimize: p.requires_grad_(False)
        if is_cnn: model.blocks[i].eval()
        else: model.layers[i*2].eval(); model.layers[i*2+1].eval()

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


# --- evaluate_mf_model (MODIFIED for CNN/MLP) ---
def evaluate_mf_model(
    model: Union[MF_MLP, MF_CNN], # <<< MODIFIED >>>
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None, # Keep signature consistent
    input_adapter: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Evaluates the trained MF model using the paper's "BP-style" approach.
    Uses the activation from the last layer (a_L) and the last projection matrix (M_L).
    """
    model.eval(); model.to(device)
    total_correct, total_samples = 0, 0

    # Determine index L for a_L and M_L
    if isinstance(model, MF_CNN): num_layers = model.num_cnn_layers
    elif isinstance(model, MF_MLP): num_layers = model.num_hidden_layers
    else: raise TypeError("Model must be MF_MLP or MF_CNN")
    last_activation_index = num_layers # a_L is at index L
    last_projection_matrix_index = num_layers # M_L is at index L

    logger.info(f"Evaluating MF ({model.__class__.__name__}) using a_{last_activation_index} and M_{last_projection_matrix_index}.")
    if last_projection_matrix_index >= len(model.projection_matrices):
        logger.error(f"Index M_{last_projection_matrix_index} out of bounds ({len(model.projection_matrices)} matrices)."); return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
    last_projection_matrix = model.get_projection_matrix(last_projection_matrix_index)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Evaluating MF {model.__class__.__name__} (BP-style)", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            # Apply adapter only if it exists (for MLP)
            eval_input = input_adapter(images) if input_adapter else images
            # Get all flattened activations [a_0_flat, ..., a_L_flat]
            all_activations_flat = model.forward_with_intermediate_activations(eval_input)
            if len(all_activations_flat) <= last_activation_index: logger.error(f"Activation list len ({len(all_activations_flat)}) too short for a_{last_activation_index}."); continue
            last_activation_flat = all_activations_flat[last_activation_index].to(device) # a_L_flat
            last_projection_matrix = last_projection_matrix.to(device)
            goodness_scores = torch.matmul(last_activation_flat, last_projection_matrix.t())
            predicted_labels = torch.argmax(goodness_scores, dim=1)
            total_correct += (predicted_labels == labels).sum().item(); total_samples += labels.size(0)
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"MF Evaluation Results ({model.__class__.__name__}, BP-style): Accuracy: {accuracy:.2f}%"); results = {"eval_accuracy": accuracy, "eval_loss": float("nan")}
    return results