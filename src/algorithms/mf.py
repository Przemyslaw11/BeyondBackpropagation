# File: src/algorithms/mf.py
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
from src.utils.monitoring import get_gpu_memory_usage

logger = logging.getLogger(__name__)

# --- mf_local_loss_fn (Remains Unchanged) ---
def mf_local_loss_fn(
    activation_i: torch.Tensor, projection_matrix_i: nn.Parameter, targets: torch.Tensor,
    criterion: nn.Module = nn.CrossEntropyLoss(),
) -> torch.Tensor:
    goodness_scores_i = torch.matmul(activation_i, projection_matrix_i.t()); loss = criterion(goodness_scores_i, targets); return loss

# --- train_mf_matrix_only (Remains Unchanged) ---
def train_mf_matrix_only(
    model: MF_MLP, matrix_index: int, optimizer: optim.Optimizer, criterion: nn.Module, train_loader: DataLoader,
    epochs: int, device: torch.device, get_matrix_input_fn: Callable[[torch.Tensor], torch.Tensor],
    wandb_run: Optional[Any] = None, log_interval: int = 100, step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, nvml_active: bool = False,
) -> Tuple[float, float]: # Return avg loss, peak memory
    if matrix_index < 0 or matrix_index >= len(model.projection_matrices): raise IndexError(f"Matrix index {matrix_index} out of bounds.")
    projection_matrix = model.get_projection_matrix(matrix_index)
    if not projection_matrix.requires_grad: logger.error(f"M_{matrix_index} requires_grad is False."); return float('nan'), 0.0
    model.to(device); model.eval()
    logger.info(f"Starting MF training for Projection Matrix M_{matrix_index}")
    peak_mem_matrix_train = 0.0; final_avg_epoch_loss = float('nan')
    for epoch in range(epochs):
        epoch_loss = 0.0; epoch_samples = 0; peak_mem_matrix_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"M{matrix_index} Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1; current_global_step = step_ref[0]; images, labels = images.to(device), labels.to(device)
            activation_a_i = get_matrix_input_fn(images).detach(); projection_matrix = projection_matrix.to(device); loss = mf_local_loss_fn(activation_a_i, projection_matrix, labels, criterion)
            if torch.isnan(loss) or torch.isinf(loss): logger.error(f"NaN/Inf loss at M{matrix_index}. Stop."); break
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            batch_size = images.size(0); epoch_loss += loss.item() * batch_size; epoch_samples += batch_size
            current_mem_used = float('nan')
            if nvml_active and gpu_handle and ((batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1): mem_info = get_gpu_memory_usage(gpu_handle);
            if mem_info: current_mem_used = mem_info[0]; peak_mem_matrix_epoch = max(peak_mem_matrix_epoch, current_mem_used)
            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                log_prefix = f"Layer_M{matrix_index}"; pbar.set_postfix(loss=f"{loss.item():.6f}"); metrics_to_log = {"global_step": current_global_step, f"{log_prefix}/Train_Loss_Batch": loss.item()}
                if not torch.isnan(torch.tensor(current_mem_used)): metrics_to_log[f"{log_prefix}/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)
        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)): break
        final_avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float('nan'); peak_mem_matrix_train = max(peak_mem_matrix_train, peak_mem_matrix_epoch)
        logger.info(f"Matrix M{matrix_index} Epoch {epoch+1}/{epochs} - Avg Loss: {final_avg_epoch_loss:.6f}, Peak Mem: {peak_mem_matrix_epoch:.1f} MiB")
    if nvml_active and gpu_handle: mem_info = get_gpu_memory_usage(gpu_handle);
    if mem_info: peak_mem_matrix_train = max(peak_mem_matrix_train, mem_info[0])
    logger.info(f"Finished MF training for Matrix M_{matrix_index}. Peak Mem: {peak_mem_matrix_train:.1f} MiB")
    return final_avg_epoch_loss, peak_mem_matrix_train


# --- train_mf_layer (Remains Unchanged) ---
def train_mf_layer(
    model: MF_MLP, layer_index: int, optimizer: optim.Optimizer, criterion: nn.Module, train_loader: DataLoader,
    epochs: int, device: torch.device, get_layer_input_fn: Callable[[torch.Tensor], torch.Tensor],
    wandb_run: Optional[Any] = None, log_interval: int = 100, step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, nvml_active: bool = False,
) -> Tuple[float, float]: # Return avg loss, peak memory
    if not (0 <= layer_index < model.num_hidden_layers): raise IndexError(f"Layer index {layer_index} out of bounds.")
    linear_layer_idx = layer_index * 2; act_layer_idx = layer_index * 2 + 1; proj_matrix_idx = layer_index + 1; w_layer_log_idx = layer_index + 1; m_matrix_log_idx = proj_matrix_idx
    linear_layer = model.layers[linear_layer_idx]; act_layer = model.layers[act_layer_idx]; projection_matrix = model.get_projection_matrix(proj_matrix_idx)
    params_require_grad = [p.requires_grad for p in linear_layer.parameters()] + [projection_matrix.requires_grad]
    if not any(params_require_grad): logger.error(f"Layer W{w_layer_log_idx}/M{m_matrix_log_idx} has no trainable parameters."); return float('nan'), 0.0
    model.to(device); model.eval(); linear_layer.train(); act_layer.train()
    logger.info(f"Starting MF training for Layer W_{w_layer_log_idx} / Matrix M_{m_matrix_log_idx}")
    peak_mem_layer_train = 0.0; final_avg_epoch_loss = float('nan')
    for epoch in range(epochs):
        epoch_loss = 0.0; epoch_samples = 0; peak_mem_layer_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"W{w_layer_log_idx}/M{m_matrix_log_idx} Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1; current_global_step = step_ref[0]; images, labels = images.to(device), labels.to(device)
            prev_activation_a_i = get_layer_input_fn(images).detach()
            linear_layer.to(device); act_layer.to(device); projection_matrix.to(device)
            pre_activation_z_next = linear_layer(prev_activation_a_i); activation_a_next = act_layer(pre_activation_z_next); loss = mf_local_loss_fn(activation_a_next, projection_matrix, labels, criterion)
            if torch.isnan(loss) or torch.isinf(loss): logger.error(f"NaN/Inf loss at W{w_layer_log_idx}/M{m_matrix_log_idx}. Stop."); break
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            batch_size = images.size(0); epoch_loss += loss.item() * batch_size; epoch_samples += batch_size
            current_mem_used = float('nan')
            if nvml_active and gpu_handle and ((batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1): mem_info = get_gpu_memory_usage(gpu_handle);
            if mem_info: current_mem_used = mem_info[0]; peak_mem_layer_epoch = max(peak_mem_layer_epoch, current_mem_used)
            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                log_prefix = f"Layer_W{w_layer_log_idx}_M{m_matrix_log_idx}"; pbar.set_postfix(loss=f"{loss.item():.6f}"); metrics_to_log = {"global_step": current_global_step, f"{log_prefix}/Train_Loss_Batch": loss.item()}
                if not torch.isnan(torch.tensor(current_mem_used)): metrics_to_log[f"{log_prefix}/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)
        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)): break
        final_avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float('nan'); peak_mem_layer_train = max(peak_mem_layer_train, peak_mem_layer_epoch)
        logger.info(f"Layer (W{w_layer_log_idx}/M{m_matrix_log_idx}) Epoch {epoch+1}/{epochs} - Avg Loss: {final_avg_epoch_loss:.6f}, Peak Mem: {peak_mem_layer_epoch:.1f} MiB")
    if nvml_active and gpu_handle: mem_info = get_gpu_memory_usage(gpu_handle);
    if mem_info: peak_mem_layer_train = max(peak_mem_layer_train, mem_info[0])
    logger.info(f"Finished MF training for Layer (W{w_layer_log_idx}/M{m_matrix_log_idx}). Peak Mem: {peak_mem_layer_train:.1f} MiB")
    return final_avg_epoch_loss, peak_mem_layer_train


# --- train_mf_model (MODIFIED - Added Early Stopping Logic) ---
def train_mf_model(
    model: MF_MLP,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader], # <<< ADDED val_loader
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
    MODIFIED: Handles requires_grad centrally.
    <<< MODIFIED: Added Early Stopping logic based on BP-style validation accuracy. >>>
    Returns the overall peak GPU memory observed across all layer training phases.
    """
    model.to(device)
    num_hidden_layers = model.num_hidden_layers
    logger.info(f"Starting layer-wise MF training for M0 and {num_hidden_layers} hidden layers.")

    algo_config = config.get("algorithm_params", config.get("training", {}))
    train_config = config.get("training", {})
    optimizer_name = algo_config.get("optimizer_type", "Adam"); optimizer_params_config = algo_config
    lr = optimizer_params_config.get("lr", 0.001); weight_decay = optimizer_params_config.get("weight_decay", 0.0); optimizer_extra_kwargs = {}
    epochs_per_layer = algo_config.get("epochs_per_layer", 5); log_interval = algo_config.get("log_interval", 100)
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None); mf_criterion = nn.CrossEntropyLoss()

    # --- Early Stopping Configuration ---
    es_enabled = train_config.get("early_stopping_enabled", True)
    es_patience = train_config.get("early_stopping_patience", 10)
    es_min_delta = train_config.get("early_stopping_min_delta", 0.0)
    # MF uses BP-style eval which ONLY returns accuracy. Mode must be 'max'.
    es_mode = "max"
    es_metric_key = "mf_val_accuracy" # Use a consistent key for logging/config
    metric_name_for_eval = "eval_accuracy" # The key returned by evaluate_mf_model
    epochs_no_improve = 0
    best_es_metric_value = -float('inf') # Always maximizing accuracy

    if es_enabled:
        if val_loader is None: logger.warning("MF Early stopping enabled but no validation loader provided. Disabling."); es_enabled = False
        else: logger.info(f"MF Early stopping enabled: Metric='{es_metric_key}' (Accuracy), Mode='{es_mode}', Patience={es_patience}, MinDelta={es_min_delta}")
    else: logger.info("MF Early stopping disabled.")
    # --- End Early Stopping Config ---

    if input_adapter is None: input_adapter = lambda x: x.view(x.shape[0], -1); logger.debug("Using default flattening input adapter for MF.")
    def get_a0_input(img_batch): adapted_input = input_adapter(img_batch) if input_adapter else img_batch.view(img_batch.shape[0], -1); return adapted_input.to(device)
    peak_mem_train = 0.0

    logger.debug("Freezing all model parameters initially.");
    for param in model.parameters(): param.requires_grad_(False); model.eval()

    # Train M0
    logger.info("--- Training Projection Matrix M_0 ---")
    projection_matrix_0 = model.get_projection_matrix(0); m0_peak_mem = 0.0; final_avg_loss_m0 = float('nan')
    projection_matrix_0.requires_grad_(True); params_m0 = [projection_matrix_0]
    if params_m0:
        optimizer_0_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_extra_kwargs}
        optimizer_0 = getattr(optim, optimizer_name)(params_m0, **optimizer_0_kwargs)
        final_avg_loss_m0, m0_peak_mem = train_mf_matrix_only(
            model, 0, optimizer_0, mf_criterion, train_loader, epochs_per_layer, device,
            get_a0_input, wandb_run, log_interval, step_ref, gpu_handle, nvml_active
        )
    else: logger.warning("M0 has no parameters.")
    peak_mem_train = max(peak_mem_train, m0_peak_mem); projection_matrix_0.requires_grad_(False)

    current_global_step = step_ref[0]; log_prefix_m0 = "Layer_M0"
    layer_summary_metrics = {"global_step": current_global_step, f"{log_prefix_m0}/Train_Loss_LayerAvg": final_avg_loss_m0, f"{log_prefix_m0}/Peak_GPU_Mem_Layer_MiB": m0_peak_mem}
    log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=False)

    # Early Stopping Check after M0
    stop_training = False
    if es_enabled:
        logger.info("--- Evaluating Performance after Training M0 for Early Stopping ---")
        val_results = evaluate_mf_model(model, val_loader, device, input_adapter=input_adapter)
        current_metric_value = val_results.get(metric_name_for_eval, float('nan'))
        es_val_log = {"global_step": current_global_step, f"MF_ES/Val_Accuracy": current_metric_value, f"MF_ES/Layer_Trained": "M0"}
        log_metrics(es_val_log, wandb_run=wandb_run, commit=True) # Commit validation log
        logger.info(f"M0 Stage ES Check: Current Val Acc = {current_metric_value:.4f}, Best = {best_es_metric_value:.4f}, Patience = {epochs_no_improve}/{es_patience}")
        if torch.isnan(torch.tensor(current_metric_value)): logger.warning("M0 Stage: Early stopping metric is NaN."); epochs_no_improve += 1
        else:
            improved = current_metric_value > best_es_metric_value + es_min_delta
            if improved: best_es_metric_value = current_metric_value; epochs_no_improve = 0; logger.info("M0 Stage: ES metric improved. Reset patience.")
            else: epochs_no_improve += 1; logger.info("M0 Stage: ES metric did not improve.")
        if epochs_no_improve >= es_patience: logger.info(f"--- Early Stopping Triggered after M0 (Best Acc: {best_es_metric_value:.4f}) ---"); stop_training = True
    else: log_metrics({}, wandb_run=wandb_run, commit=True) # Commit M0 training log

    if checkpoint_dir and params_m0: create_directory_if_not_exists(checkpoint_dir); save_checkpoint(state={"state_dict": model.state_dict(), "layer_trained_index": -1}, is_best=False, filename="mf_matrix_M0_complete.pth", checkpoint_dir=checkpoint_dir)

    # Train Hidden Layers Loop
    current_layer_input_fn = get_a0_input
    if not stop_training:
        for i in range(num_hidden_layers):
            w_layer_log_idx = i + 1; m_matrix_log_idx = i + 1; log_prefix_layer = f"Layer_W{w_layer_log_idx}_M{m_matrix_log_idx}"
            logger.info(f"--- Training Hidden Layer W_{w_layer_log_idx} / Matrix M_{m_matrix_log_idx} ---")
            linear_layer = model.layers[i * 2]; projection_matrix = model.get_projection_matrix(i + 1); layer_i_peak_mem = 0.0; final_avg_loss_layer_i = float('nan')
            params_to_optimize = []
            for p in linear_layer.parameters(): p.requires_grad_(True); params_to_optimize.append(p)
            projection_matrix.requires_grad_(True); params_to_optimize.append(projection_matrix)
            if params_to_optimize:
                optimizer_i_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_extra_kwargs}
                optimizer = getattr(optim, optimizer_name)(params_to_optimize, **optimizer_i_kwargs)
                final_avg_loss_layer_i, layer_i_peak_mem = train_mf_layer(
                    model, i, optimizer, mf_criterion, train_loader, epochs_per_layer, device,
                    current_layer_input_fn, wandb_run, log_interval, step_ref, gpu_handle, nvml_active
                )
            else: logger.warning(f"Layer W{w_layer_log_idx}/M{m_matrix_log_idx} has no parameters.")
            peak_mem_train = max(peak_mem_train, layer_i_peak_mem)
            for p in params_to_optimize: p.requires_grad_(False); model.layers[i * 2].eval(); model.layers[i * 2 + 1].eval()

            current_global_step = step_ref[0]
            layer_summary_metrics = {"global_step": current_global_step, f"{log_prefix_layer}/Train_Loss_LayerAvg": final_avg_loss_layer_i, f"{log_prefix_layer}/Peak_GPU_Mem_Layer_MiB": layer_i_peak_mem}
            log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=False)

            # Early Stopping Check after Layer i
            if es_enabled:
                logger.info(f"--- Evaluating Performance after Training Layer {i+1} for Early Stopping ---")
                val_results = evaluate_mf_model(model, val_loader, device, input_adapter=input_adapter)
                current_metric_value = val_results.get(metric_name_for_eval, float('nan'))
                es_val_log = {"global_step": current_global_step, f"MF_ES/Val_Accuracy": current_metric_value, f"MF_ES/Layer_Trained": f"W{i+1}/M{i+1}"}
                log_metrics(es_val_log, wandb_run=wandb_run, commit=True)
                logger.info(f"Layer {i+1} ES Check: Current Val Acc = {current_metric_value:.4f}, Best = {best_es_metric_value:.4f}, Patience = {epochs_no_improve}/{es_patience}")
                if torch.isnan(torch.tensor(current_metric_value)): logger.warning(f"Layer {i+1}: Early stopping metric is NaN."); epochs_no_improve += 1
                else:
                    improved = current_metric_value > best_es_metric_value + es_min_delta
                    if improved: best_es_metric_value = current_metric_value; epochs_no_improve = 0; logger.info(f"Layer {i+1}: ES metric improved. Reset patience.")
                    else: epochs_no_improve += 1; logger.info(f"Layer {i+1}: ES metric did not improve.")
                if epochs_no_improve >= es_patience: logger.info(f"--- Early Stopping Triggered after Layer {i+1} (Best Acc: {best_es_metric_value:.4f}) ---"); stop_training = True
            else: log_metrics({}, wandb_run=wandb_run, commit=True)

            if checkpoint_dir and params_to_optimize: create_directory_if_not_exists(checkpoint_dir); chkpt_filename = f"mf_layer_{w_layer_log_idx}_complete.pth"; save_checkpoint(state={"state_dict": model.state_dict(), "layer_trained_index": i}, is_best=False, filename=chkpt_filename, checkpoint_dir=checkpoint_dir)
            if stop_training: break

            def create_next_input_fn(trained_layer_idx: int, previous_input_fn: Callable):
                lin_layer_k, act_layer_k = model.layers[trained_layer_idx*2], model.layers[trained_layer_idx*2+1]; lin_layer_k.eval(); act_layer_k.eval()
                @torch.no_grad()
                def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                    a_prev = previous_input_fn(img_batch); lin_layer_k.to(device); act_layer_k.to(device); a_current = act_layer_k(lin_layer_k(a_prev.to(device))); return a_current.detach()
                return next_input_fn
            current_layer_input_fn = create_next_input_fn(i, current_layer_input_fn)
        # --- End Hidden Layer Loop ---

    logger.info("Finished all layer-wise MF training.")
    model.eval()
    return peak_mem_train


# --- evaluate_mf_model (Remains Unchanged) ---
def evaluate_mf_model(
    model: MF_MLP, data_loader: DataLoader, device: torch.device, criterion: Optional[nn.Module] = None, input_adapter: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Evaluates the trained MF_MLP model using the paper's "BP-style" approach.
    (Function content remains the same as provided previously)
    """
    model.eval(); model.to(device); total_correct, total_samples = 0, 0; num_hidden_layers = model.num_hidden_layers; last_activation_index = num_hidden_layers; last_projection_matrix_index = num_hidden_layers
    logger.info(f"Evaluating MF model using activation a_{last_activation_index} and matrix M_{last_projection_matrix_index}.")
    if last_projection_matrix_index >= len(model.projection_matrices): logger.error(f"Index M_{last_projection_matrix_index} out of bounds."); return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
    last_projection_matrix = model.get_projection_matrix(last_projection_matrix_index)
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating MF (BP-style)", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device); eval_input = input_adapter(images) if input_adapter else images.view(images.shape[0], -1); all_activations = model.forward_with_intermediate_activations(eval_input)
            if len(all_activations) <= last_activation_index: logger.error(f"Activation list len ({len(all_activations)}) too short."); continue
            last_hidden_activation = all_activations[last_activation_index].to(device); last_projection_matrix = last_projection_matrix.to(device); goodness_scores = torch.matmul(last_hidden_activation, last_projection_matrix.t()); predicted_labels = torch.argmax(goodness_scores, dim=1)
            total_correct += (predicted_labels == labels).sum().item(); total_samples += labels.size(0)
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0; logger.info(f"MF Evaluation Results (BP-style): Accuracy: {accuracy:.2f}%")
    results = {"eval_accuracy": accuracy, "eval_loss": float("nan")} # Loss not calculated in this style
    return results