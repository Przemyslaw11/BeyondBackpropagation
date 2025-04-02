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
from src.utils.monitoring import get_gpu_memory_usage # Import memory usage function

logger = logging.getLogger(__name__)

# --- mf_local_loss_fn (no changes) ---
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


# --- train_mf_matrix_only (MODIFIED) ---
def train_mf_matrix_only(
    model: MF_MLP,
    matrix_index: int,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    get_matrix_input_fn: Callable[[torch.Tensor], torch.Tensor],
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, # ADDED
    nvml_active: bool = False, # ADDED
) -> Tuple[float, float]: # Return avg loss, peak memory
    """
    Trains a single projection matrix using local loss based on its corresponding activation.
    MODIFIED: Accepts handle/active, samples memory, returns avg loss and peak memory.
    """
    if matrix_index < 0 or matrix_index >= len(model.projection_matrices):
         raise IndexError(f"Matrix index {matrix_index} out of bounds.")

    projection_matrix = model.get_projection_matrix(matrix_index)
    # Ensure only this matrix requires grad for this phase
    for p in model.parameters(): p.requires_grad_(False)
    projection_matrix.requires_grad_(True)
    model.to(device)
    model.eval() # Keep model blocks in eval mode

    logger.info(f"Starting MF training for Projection Matrix M_{matrix_index}")

    peak_mem_matrix_train = 0.0 # Track peak memory for this matrix training
    final_avg_epoch_loss = float('nan')

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        peak_mem_matrix_epoch = 0.0 # Track peak memory for this epoch
        pbar = tqdm(
            train_loader, desc=f"M{matrix_index} Epoch {epoch+1}/{epochs}", leave=False
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1
            current_global_step = step_ref[0]

            images, labels = images.to(device), labels.to(device)
            # Use the provided function to get the correct activation (a_i)
            activation_a_i = get_matrix_input_fn(images).detach()
            loss = mf_local_loss_fn(activation_a_i, projection_matrix, labels, criterion)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss at M{matrix_index}, Epoch {epoch+1}, Batch {batch_idx}. Stop.")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size

            # --- Sample memory usage periodically or at end of batch ---
            current_mem_used = float('nan')
            if nvml_active and gpu_handle and ((batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1):
                 mem_info = get_gpu_memory_usage(gpu_handle)
                 if mem_info:
                     current_mem_used = mem_info[0]
                     peak_mem_matrix_epoch = max(peak_mem_matrix_epoch, current_mem_used)
            # --- End memory sampling ---

            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.6f}")
                metrics_to_log = {
                    "global_step": current_global_step,
                    f"Layer_M{matrix_index}/Train_Loss_Batch": avg_loss_batch
                }
                if not torch.isnan(torch.tensor(current_mem_used)):
                    metrics_to_log[f"Layer_M{matrix_index}/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)

        # Check for NaN/Inf loss again after loop
        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)):
            logger.error(f"Terminating M{matrix_index} training due to invalid loss.")
            break # Exit epoch loop

        # Calculate average loss for the epoch and update overall peak mem
        final_avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        peak_mem_matrix_train = max(peak_mem_matrix_train, peak_mem_matrix_epoch)

        logger.info(f"Matrix M{matrix_index} Epoch {epoch+1}/{epochs} - Avg Local Loss: {final_avg_epoch_loss:.6f}, Peak Mem Epoch: {peak_mem_matrix_epoch:.1f} MiB")
        # Log epoch summary (optional)

    # Sample memory one last time
    if nvml_active and gpu_handle:
        mem_info = get_gpu_memory_usage(gpu_handle)
        if mem_info:
            peak_mem_matrix_train = max(peak_mem_matrix_train, mem_info[0])

    logger.info(f"Finished MF training for Matrix M_{matrix_index}. Overall Peak Mem Matrix: {peak_mem_matrix_train:.1f} MiB")
    projection_matrix.requires_grad_(False)
    return final_avg_epoch_loss, peak_mem_matrix_train # Return loss and peak mem


# --- train_mf_layer (MODIFIED) ---
def train_mf_layer(
    model: MF_MLP,
    layer_index: int,  # Index i=0..L-1. Trains W_{i+1} and M_{i+1}.
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    get_layer_input_fn: Callable[[torch.Tensor], torch.Tensor],  # Function to get a_i
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, # ADDED
    nvml_active: bool = False, # ADDED
) -> Tuple[float, float]: # Return avg loss, peak memory
    """
    Trains a single layer 'i+1' (W_{i+1} and M_{i+1}) of an MF_MLP using local loss.
    MODIFIED: Accepts handle/active, samples memory, returns avg loss and peak memory.
    """
    if not (0 <= layer_index < model.num_hidden_layers):
        raise IndexError(f"Layer index {layer_index} out of bounds.")

    linear_layer_idx = layer_index * 2
    act_layer_idx = layer_index * 2 + 1
    proj_matrix_idx = layer_index + 1
    w_layer_log_idx = layer_index + 1
    m_matrix_log_idx = proj_matrix_idx

    linear_layer = model.layers[linear_layer_idx]  # W_{i+1}
    act_layer = model.layers[act_layer_idx]  # sigma_{i+1}
    projection_matrix = model.get_projection_matrix(proj_matrix_idx)  # M_{i+1}

    # Ensure only W_{i+1} and M_{i+1} require gradients
    for p in model.parameters(): p.requires_grad_(False)
    for p in linear_layer.parameters(): p.requires_grad_(True)
    projection_matrix.requires_grad_(True)

    model.to(device)
    # Set layers to train, others to eval
    model.eval()
    linear_layer.train()
    act_layer.train() # Activation doesn't have params but adheres to train/eval state

    logger.info(f"Starting MF training for Layer W_{w_layer_log_idx} / Matrix M_{m_matrix_log_idx}")

    peak_mem_layer_train = 0.0 # Track peak memory for this layer's training
    final_avg_epoch_loss = float('nan')

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        peak_mem_layer_epoch = 0.0 # Track peak memory for this epoch
        pbar = tqdm(
            train_loader,
            desc=f"W{w_layer_log_idx}/M{m_matrix_log_idx} Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1
            current_global_step = step_ref[0]

            images, labels = images.to(device), labels.to(device)
            prev_activation_a_i = get_layer_input_fn(images).detach() # Get a_i
            pre_activation_z_next = linear_layer(prev_activation_a_i) # z_{i+1}
            activation_a_next = act_layer(pre_activation_z_next) # a_{i+1}
            loss = mf_local_loss_fn(activation_a_next, projection_matrix, labels, criterion)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss at W{w_layer_log_idx}/M{m_matrix_log_idx}, Epoch {epoch+1}, Batch {batch_idx}. Stop.")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # Updates W_{i+1} and M_{i+1}

            batch_size = images.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size

            # --- Sample memory usage periodically or at end of batch ---
            current_mem_used = float('nan')
            if nvml_active and gpu_handle and ((batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1):
                 mem_info = get_gpu_memory_usage(gpu_handle)
                 if mem_info:
                     current_mem_used = mem_info[0]
                     peak_mem_layer_epoch = max(peak_mem_layer_epoch, current_mem_used)
            # --- End memory sampling ---

            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.6f}")
                metrics_to_log = {
                    "global_step": current_global_step,
                    f"Layer_M{m_matrix_log_idx}/Train_Loss_Batch": avg_loss_batch # Log loss under Matrix index for consistency
                }
                if not torch.isnan(torch.tensor(current_mem_used)):
                    metrics_to_log[f"Layer_M{m_matrix_log_idx}/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)

        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)):
            logger.error(f"Terminating W{w_layer_log_idx}/M{m_matrix_log_idx} training due to invalid loss.")
            break # Exit epoch loop

        final_avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        peak_mem_layer_train = max(peak_mem_layer_train, peak_mem_layer_epoch)

        logger.info(f"Layer (W{w_layer_log_idx}/M{m_matrix_log_idx}) Epoch {epoch+1}/{epochs} - Avg Local Loss: {final_avg_epoch_loss:.6f}, Peak Mem Epoch: {peak_mem_layer_epoch:.1f} MiB")
        # Log epoch summary (optional)

    # Sample memory one last time
    if nvml_active and gpu_handle:
        mem_info = get_gpu_memory_usage(gpu_handle)
        if mem_info:
            peak_mem_layer_train = max(peak_mem_layer_train, mem_info[0])

    logger.info(f"Finished MF training for Layer (W{w_layer_log_idx}/M{m_matrix_log_idx}). Overall Peak Mem Layer: {peak_mem_layer_train:.1f} MiB")
    linear_layer.eval()
    act_layer.eval()
    projection_matrix.requires_grad_(False)
    return final_avg_epoch_loss, peak_mem_layer_train # Return loss and peak mem


# --- train_mf_model (MODIFIED) ---
def train_mf_model(
    model: MF_MLP,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, # ADDED
    nvml_active: bool = False, # ADDED
) -> float: # Returns overall peak memory
    """
    Orchestrates the layer-wise training of an MF_MLP model.
    Returns the overall peak GPU memory observed across all layer training phases.
    """
    model.to(device)
    num_hidden_layers = model.num_hidden_layers
    logger.info(f"Starting layer-wise MF training for M0 and {num_hidden_layers} hidden layers.")

    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    optimizer_params_config = algo_config
    lr = optimizer_params_config.get("lr", 0.001)
    weight_decay = optimizer_params_config.get("weight_decay", 0.0)
    optimizer_extra_kwargs = {} # Add if MF needs specific optimizer args like betas

    epochs_per_layer = algo_config.get("epochs_per_layer", 5)
    log_interval = algo_config.get("log_interval", 100)
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None)
    mf_criterion = nn.CrossEntropyLoss()

    def get_a0_input(img_batch):
        # Apply adapter if provided (typically flattening)
        adapted_input = input_adapter(img_batch) if input_adapter else img_batch.view(img_batch.shape[0], -1)
        return adapted_input.to(device)

    peak_mem_train = 0.0 # Track overall peak memory

    logger.info("--- Training Projection Matrix M_0 ---")
    projection_matrix_0 = model.get_projection_matrix(0)
    params_m0 = [projection_matrix_0]
    m0_peak_mem = 0.0
    final_avg_loss_m0 = float('nan')

    # Check if requires_grad is True before trying to train
    if projection_matrix_0.requires_grad:
        optimizer_0_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_extra_kwargs}
        optimizer_0 = getattr(optim, optimizer_name)(params_m0, **optimizer_0_kwargs)
        final_avg_loss_m0, m0_peak_mem = train_mf_matrix_only(
            model=model, matrix_index=0, optimizer=optimizer_0, criterion=mf_criterion,
            train_loader=train_loader, epochs=epochs_per_layer, device=device,
            get_matrix_input_fn=get_a0_input, wandb_run=wandb_run, log_interval=log_interval,
            step_ref=step_ref,
            gpu_handle=gpu_handle, # Pass handle
            nvml_active=nvml_active # Pass status
        )
        peak_mem_train = max(peak_mem_train, m0_peak_mem) # Update overall peak
    else:
        logger.warning("M_0 does not require gradients. Skipping training.")
        # If skipped, peak mem for this part is effectively 0, no update needed

    # Log layer summary after training M0 (or skipping)
    current_global_step = step_ref[0]
    layer_summary_metrics = {
        "global_step": current_global_step,
        f"Layer_M0/Train_Loss_LayerAvg": final_avg_loss_m0,
        f"Layer_M0/Peak_GPU_Mem_Layer_MiB": m0_peak_mem,
    }
    log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=True)
    logger.debug(f"Logged MF Layer M0 summary at global_step {current_global_step}")

    if checkpoint_dir and projection_matrix_0.requires_grad: # Only save if trained
        create_directory_if_not_exists(checkpoint_dir)
        save_checkpoint(
            state={"state_dict": model.state_dict(), "layer_trained_index": -1}, # Use -1 for M0
            is_best=False, filename="mf_matrix_M0_complete.pth", checkpoint_dir=checkpoint_dir,
        )

    current_layer_input_fn = get_a0_input

    for i in range(num_hidden_layers):
        w_layer_log_idx = i + 1
        m_matrix_log_idx = i + 1
        logger.info(f"--- Training Hidden Layer W_{w_layer_log_idx} / Matrix M_{m_matrix_log_idx} ---")

        linear_layer = model.layers[i * 2]
        projection_matrix = model.get_projection_matrix(i + 1)
        layer_i_peak_mem = 0.0
        final_avg_loss_layer_i = float('nan')

        # Check if parameters require grad before setting up optimizer and training
        params_to_optimize = []
        if any(p.requires_grad for p in linear_layer.parameters()):
            params_to_optimize.extend(linear_layer.parameters())
        if projection_matrix.requires_grad:
            params_to_optimize.append(projection_matrix)

        if not params_to_optimize:
            logger.warning(f"Skipping W{w_layer_log_idx}/M{m_matrix_log_idx}: No parameters require gradients.")
            # Peak mem for this layer is 0 if skipped
        else:
            # Ensure grads are enabled on the parameters *before* optimizer creation
            for p in params_to_optimize: p.requires_grad_(True)

            optimizer_i_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_extra_kwargs}
            optimizer = getattr(optim, optimizer_name)(params_to_optimize, **optimizer_i_kwargs)

            final_avg_loss_layer_i, layer_i_peak_mem = train_mf_layer(
                model=model, layer_index=i, optimizer=optimizer, criterion=mf_criterion,
                train_loader=train_loader, epochs=epochs_per_layer, device=device,
                get_layer_input_fn=current_layer_input_fn, wandb_run=wandb_run, log_interval=log_interval,
                step_ref=step_ref,
                gpu_handle=gpu_handle, # Pass handle
                nvml_active=nvml_active # Pass status
            )
            peak_mem_train = max(peak_mem_train, layer_i_peak_mem) # Update overall peak

            # Freeze parameters after training
            for param in params_to_optimize: param.requires_grad_(False)
            linear_layer.eval(); model.layers[i * 2 + 1].eval()

        # Log layer summary after training (or skipping)
        current_global_step = step_ref[0]
        layer_summary_metrics = {
            "global_step": current_global_step,
            f"Layer_M{m_matrix_log_idx}/Train_Loss_LayerAvg": final_avg_loss_layer_i,
            f"Layer_M{m_matrix_log_idx}/Peak_GPU_Mem_Layer_MiB": layer_i_peak_mem,
        }
        log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=True)
        logger.debug(f"Logged MF Layer W{w_layer_log_idx}/M{m_matrix_log_idx} summary at global_step {current_global_step}")

        if checkpoint_dir and params_to_optimize: # Only save if trained
            create_directory_if_not_exists(checkpoint_dir)
            chkpt_filename = f"mf_hidden_layer_W{w_layer_log_idx}_M{m_matrix_log_idx}_complete.pth"
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained_index": i},
                is_best=False, filename=chkpt_filename, checkpoint_dir=checkpoint_dir,
            )

        # Prepare input function for the *next* layer
        def create_next_input_fn(trained_layer_idx: int, previous_input_fn: Callable):
            # Capture current layer's index correctly
            lin_layer_k, act_layer_k = model.layers[trained_layer_idx*2], model.layers[trained_layer_idx*2+1]
            lin_layer_k.eval(); act_layer_k.eval() # Ensure they are in eval mode
            @torch.no_grad()
            def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                a_prev = previous_input_fn(img_batch)
                # Ensure model components are on the correct device inside the closure
                lin_layer_k.to(device); act_layer_k.to(device)
                a_current = act_layer_k(lin_layer_k(a_prev.to(device)))
                return a_current.detach()
            return next_input_fn
        current_layer_input_fn = create_next_input_fn(i, current_layer_input_fn)

    logger.info("Finished all layer-wise MF training.")
    model.eval()
    return peak_mem_train # Return overall peak memory


# --- evaluate_mf_model (no changes needed) ---
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
    model.eval()
    model.to(device)
    total_correct, total_samples = 0, 0
    num_hidden_layers = model.num_hidden_layers
    last_activation_index = num_hidden_layers # a_L is at index L
    last_projection_matrix_index = num_hidden_layers # M_L is at index L

    logger.info(
        f"Evaluating MF model using activation a_{last_activation_index} and matrix M_{last_projection_matrix_index}."
    )

    if last_projection_matrix_index >= len(model.projection_matrices):
        logger.error(f"Index M_{last_projection_matrix_index} out of bounds ({len(model.projection_matrices)} matrices exist).")
        return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}

    last_projection_matrix = model.get_projection_matrix(last_projection_matrix_index)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating MF (BP-style)", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            # Apply input adapter (flattening) before getting activations
            eval_input = input_adapter(images) if input_adapter else images.view(images.shape[0], -1)

            all_activations = model.forward_with_intermediate_activations(eval_input)

            if len(all_activations) <= last_activation_index:
                logger.error(f"Activation list len ({len(all_activations)}) too short for a_{last_activation_index}.")
                continue

            last_hidden_activation = all_activations[last_activation_index] # This is a_L
            goodness_scores = torch.matmul(last_hidden_activation, last_projection_matrix.t())
            predicted_labels = torch.argmax(goodness_scores, dim=1)

            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"MF Evaluation Results (BP-style): Accuracy: {accuracy:.2f}%")
    # MF evaluation doesn't typically calculate a loss in this manner
    results = {"eval_accuracy": accuracy, "eval_loss": float("nan")}
    return results