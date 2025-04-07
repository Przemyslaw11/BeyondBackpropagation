# File: src/algorithms/mf.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    # Ensure inputs are on the same device
    if activation_i.device != projection_matrix_i.device:
        projection_matrix_i = projection_matrix_i.to(activation_i.device)
    if activation_i.device != targets.device:
        targets = targets.to(activation_i.device)

    goodness_scores_i = torch.matmul(activation_i, projection_matrix_i.t())
    loss = criterion(goodness_scores_i, targets)
    return loss

# --- NEW: Loss Computation Helper Functions ---

def compute_mf_matrix_loss(
    model: MF_MLP,
    matrix_index: int,
    criterion: nn.Module,
    images: torch.Tensor, # Pass batch data directly
    labels: torch.Tensor, # Pass batch data directly
    device: torch.device,
    get_matrix_input_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Optional[torch.Tensor]: # Return loss tensor or None on error
    """Computes the local loss for a projection matrix."""
    if matrix_index < 0 or matrix_index >= len(model.projection_matrices):
        logger.error(f"compute_mf_matrix_loss: Matrix index {matrix_index} out of bounds.")
        return None

    projection_matrix = model.get_projection_matrix(matrix_index)
    if not projection_matrix.requires_grad:
        # This shouldn't happen if train_mf_model manages grads correctly, but good check.
        logger.warning(f"M_{matrix_index} requires_grad is False during loss computation.")
        # Return None, let the outer loop handle it (e.g., skip backward/step)
        return None

    try:
        # Keep model in eval, M_i grad status is handled externally
        model.eval()
        # Ensure get_matrix_input_fn returns tensor on the correct device
        activation_a_i = get_matrix_input_fn(images).detach().to(device)
        projection_matrix = projection_matrix.to(device) # Ensure matrix is on device

        loss = mf_local_loss_fn(activation_a_i, projection_matrix, labels, criterion)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN/Inf loss computed for M{matrix_index}.")
            return None # Indicate error

        return loss
    except Exception as e:
        logger.error(f"Error computing loss for M{matrix_index}: {e}", exc_info=True)
        return None


def compute_mf_layer_loss(
    model: MF_MLP,
    layer_index: int, # Index i=0..L-1. Computes loss involving W_{i+1} and M_{i+1}.
    criterion: nn.Module,
    images: torch.Tensor, # Pass batch data directly
    labels: torch.Tensor, # Pass batch data directly
    device: torch.device,
    get_layer_input_fn: Callable[[torch.Tensor], torch.Tensor], # Function to get a_i
) -> Optional[torch.Tensor]: # Return loss tensor or None on error
    """Computes the local loss for a hidden layer (W_next, M_next)."""
    if not (0 <= layer_index < model.num_hidden_layers):
         logger.error(f"compute_mf_layer_loss: Layer index {layer_index} out of bounds.")
         return None

    linear_layer_idx = layer_index * 2
    act_layer_idx = layer_index * 2 + 1
    proj_matrix_idx = layer_index + 1

    linear_layer = model.layers[linear_layer_idx] # W_{i+1}
    act_layer = model.layers[act_layer_idx] # sigma_{i+1}
    projection_matrix = model.get_projection_matrix(proj_matrix_idx) # M_{i+1}

    # Check if relevant parameters require grad (should be handled by orchestrator)
    if not any(p.requires_grad for p in linear_layer.parameters()) or not projection_matrix.requires_grad:
        logger.warning(f"W{layer_index+1}/M{proj_matrix_idx} requires_grad is False during loss computation.")
        return None

    try:
        # Keep model eval, but ensure current layers are treated appropriately if they have state (like BN, though not used here)
        model.eval()
        # If layers like BN were used, you might need model.train() here and manage freezing others carefully.
        # For pure Linear/ReLU, model.eval() is sufficient.

        # Ensure get_layer_input_fn returns tensor on the correct device
        prev_activation_a_i = get_layer_input_fn(images).detach().to(device)

        # Ensure layers are on device
        linear_layer.to(device); act_layer.to(device); projection_matrix.to(device)

        pre_activation_z_next = linear_layer(prev_activation_a_i) # z_{i+1}
        activation_a_next = act_layer(pre_activation_z_next) # a_{i+1}

        loss = mf_local_loss_fn(activation_a_next, projection_matrix, labels, criterion)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN/Inf loss computed for W{layer_index+1}/M{proj_matrix_idx}.")
            return None # Indicate error

        return loss
    except Exception as e:
         logger.error(f"Error computing loss for W{layer_index+1}/M{proj_matrix_idx}: {e}", exc_info=True)
         return None


# --- Refactored train_mf_model to use a SINGLE persistent optimizer ---
def train_mf_model(
    model: MF_MLP,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None, # Should provide flattened input
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float: # Returns overall peak memory
    """
    Orchestrates the layer-wise training of an MF_MLP model using a SINGLE persistent optimizer.
    Cycles through training phases for each layer group (M0, then W1/M1, W2/M2, ...).
    """
    model.to(device)
    num_hidden_layers = model.num_hidden_layers
    logger.info(f"Starting layer-wise MF training for M0 and {num_hidden_layers} hidden layers (Single Optimizer Strategy).")

    # --- Config ---
    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    lr = algo_config.get("lr", 0.001)
    weight_decay = algo_config.get("weight_decay", 0.0)
    optimizer_extra_kwargs = {} # Add if MF needs specific optimizer args

    # epochs_per_layer now defines the number of epochs spent focusing on each layer group
    epochs_per_layer_phase = algo_config.get("epochs_per_layer", 10)
    log_interval = algo_config.get("log_interval", 100)
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None)
    mf_criterion = nn.CrossEntropyLoss()

    # --- Input Adapter ---
    # Ensure the input adapter flattens the image for MLP
    if input_adapter is None:
         input_adapter = lambda x: x.view(x.shape[0], -1)
         logger.info("Using default flattening input adapter for MF_MLP.")

    # Define helper to get a0 (input activation) correctly adapted and on device
    def get_a0_input(img_batch):
        adapted_input = input_adapter(img_batch) if input_adapter else img_batch.view(img_batch.shape[0], -1)
        return adapted_input.to(device)

    # --- Create SINGLE Optimizer for ALL relevant parameters ---
    params_to_optimize = []
    # Add M0
    params_to_optimize.append(model.get_projection_matrix(0))
    # Add W_i+1 and M_i+1 for i=0..L-1
    for i in range(num_hidden_layers):
        # Ensure parameters of linear layers are correctly retrieved
        linear_layer_params = list(model.layers[i*2].parameters())
        if not linear_layer_params:
            logger.warning(f"Linear layer {i*2} (W_{i+1}) appears to have no parameters.")
        params_to_optimize.extend(linear_layer_params) # W_i+1
        params_to_optimize.append(model.get_projection_matrix(i + 1)) # M_i+1

    # Ensure all parameters added require grad initially (optimizer needs this)
    for p in params_to_optimize:
        if isinstance(p, nn.Parameter): # Check if it's a Parameter (like M_i)
            p.requires_grad_(True)
        # For parameters within ModuleList layers (W_i), requires_grad should be handled by the optimizer target list

    optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_extra_kwargs}
    try:
        optimizer = getattr(optim, optimizer_name)(params_to_optimize, **optimizer_kwargs)
    except Exception as e:
        logger.error(f"Failed to create optimizer '{optimizer_name}': {e}", exc_info=True)
        return float('nan') # Cannot train without optimizer

    logger.info(f"Created single '{optimizer_name}' optimizer for all {len(params_to_optimize)} MF parameter groups/tensors.")

    # --- Training Loop ---
    peak_mem_train = 0.0
    # Store activations needed as input for subsequent layers (per batch)
    activations_cache: Dict[int, torch.Tensor] = {}

    # Calculate total steps roughly for reference
    num_layer_phases = 1 + num_hidden_layers # M0 phase + L hidden layer phases
    total_iterations = num_layer_phases * epochs_per_layer_phase * len(train_loader)
    logger.info(f"Estimated total optimizer steps: {total_iterations} ({num_layer_phases} phases * {epochs_per_layer_phase} epochs/phase * {len(train_loader)} batches/epoch)")

    overall_start_time = time.time()

    # --- Loop through training phases (one focus per layer group) ---
    for layer_focus_idx in range(-1, num_hidden_layers): # -1 for M0, 0..L-1 for W{i+1}/M{i+1}

        is_m0_phase = (layer_focus_idx == -1)
        if is_m0_phase:
            log_prefix_phase = "Layer_M0"
            target_matrix_idx = 0
            params_to_enable_grad = [model.get_projection_matrix(0)]
            logger.info(f"--- Starting Training Phase for M_0 ---")
        else: # Hidden layer phase
            target_layer_idx = layer_focus_idx # This is 'i'
            target_w_idx = target_layer_idx + 1 # W_{i+1}
            target_m_idx = target_layer_idx + 1 # M_{i+1}
            log_prefix_phase = f"Layer_W{target_w_idx}_M{target_m_idx}"
            # Parameters to have gradients enabled for this phase
            params_to_enable_grad = list(model.layers[target_layer_idx*2].parameters()) + \
                                    [model.get_projection_matrix(target_m_idx)]
            logger.info(f"--- Starting Training Phase for W_{target_w_idx} / M_{target_m_idx} ---")

        # --- Enable gradients ONLY for the parameters of the current phase ---
        # It's crucial that the single optimizer was initialized with ALL parameters.
        # Here, we control which parameters receive gradients during the backward pass.
        for param in model.parameters(): # Iterate through ALL model parameters
            is_target_param = False
            for target_p in params_to_enable_grad:
                 if param is target_p: # Check for object identity
                     is_target_param = True
                     break
            param.requires_grad_(is_target_param)
        # Log which parameters have requires_grad = True for verification
        # logger.debug(f"Phase {log_prefix_phase}: Enabled grads for {[n for n, p in model.named_parameters() if p.requires_grad]}")

        # --- Inner loop for epochs dedicated to this layer phase ---
        for epoch in range(epochs_per_layer_phase):
            epoch_start_time = time.time()
            epoch_loss_acc = 0.0
            epoch_samples = 0
            peak_mem_epoch = 0.0
            model.eval() # Keep model in eval by default for forward passes

            pbar_desc = f"{log_prefix_phase} Epoch {epoch+1}/{epochs_per_layer_phase}"
            pbar = tqdm(train_loader, desc=pbar_desc, leave=False)

            for batch_idx, (images, labels) in enumerate(pbar):
                step_ref[0] += 1
                current_global_step = step_ref[0]
                images, labels = images.to(device), labels.to(device)

                # --- Forward pass to get needed activations ---
                # We always need a_0 (input)
                activations_cache[0] = get_a0_input(images)

                # Calculate activations up to the layer *before* the one being trained's INPUT
                # E.g., if training W2/M2 (layer_focus_idx=1), we need a_1.
                # M0 phase (idx=-1) needs a0. W1/M1 phase (idx=0) needs a0. W2/M2 phase (idx=1) needs a1...
                max_needed_prev_act_idx = layer_focus_idx if not is_m0_phase else -1
                current_act = activations_cache[0]
                for act_idx in range(1, max_needed_prev_act_idx + 1): # Calculate a_1 up to a_{max_needed}
                    lin_layer = model.layers[(act_idx-1)*2].to(device)
                    act_layer = model.layers[(act_idx-1)*2+1].to(device)
                    with torch.no_grad(): # Only need forward pass value
                        current_act = act_layer(lin_layer(current_act)).detach()
                    activations_cache[act_idx] = current_act

                # --- Compute Local Loss for the CURRENT FOCUSED layer/matrix ---
                loss = None
                if is_m0_phase:
                    # Input function gives a0
                    loss = compute_mf_matrix_loss(model, 0, mf_criterion, images, labels, device, lambda _: activations_cache[0])
                else: # Hidden layer phase (W_i+1 / M_i+1)
                    # Input function gives a_i (where i == layer_focus_idx)
                    input_act_idx = layer_focus_idx
                    if input_act_idx in activations_cache:
                        loss = compute_mf_layer_loss(model, layer_focus_idx, mf_criterion, images, labels, device, lambda _: activations_cache[input_act_idx])
                    else:
                         logger.error(f"Required activation a_{input_act_idx} not found in cache.")

                # --- Optimization Step ---
                if loss is not None and torch.isfinite(loss):
                    optimizer.zero_grad()
                    # Backward pass will only compute gradients for params with requires_grad=True
                    loss.backward()
                    optimizer.step() # Optimizer updates params with grads using persistent state

                    # Accumulate loss
                    batch_size = images.size(0)
                    loss_item = loss.item()
                    epoch_loss_acc += loss_item * batch_size
                    epoch_samples += batch_size
                    pbar.set_postfix(loss=f"{loss_item:.6f}")
                elif loss is None:
                    logger.warning(f"Skipping optimizer step at global step {current_global_step} due to loss computation error.")
                else: # NaN or Inf loss
                    logger.error(f"Skipping optimizer step at global step {current_global_step} due to invalid loss value: {loss.item()}")
                    # break # Optional: break inner loop on invalid loss

                # --- Memory Monitoring ---
                current_mem_used = float('nan')
                if nvml_active and gpu_handle:
                     mem_info = get_gpu_memory_usage(gpu_handle)
                     if mem_info:
                         current_mem_used = mem_info[0]
                         peak_mem_epoch = max(peak_mem_epoch, current_mem_used)

                # --- Logging ---
                if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                    metrics_to_log = {
                        "global_step": current_global_step,
                        f"{log_prefix_phase}/Train_Loss_Batch": loss_item if loss is not None else float('nan'),
                    }
                    if not torch.isnan(torch.tensor(current_mem_used)):
                        metrics_to_log[f"{log_prefix_phase}/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                    log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)

            # --- End of Epoch ---
            avg_epoch_loss = epoch_loss_acc / epoch_samples if epoch_samples > 0 else float('nan')
            peak_mem_train = max(peak_mem_train, peak_mem_epoch) # Update overall peak memory
            epoch_duration = time.time() - epoch_start_time

            logger.info(f"{pbar_desc} - Avg Loss: {avg_epoch_loss:.6f}, Peak Mem Epoch: {peak_mem_epoch:.1f} MiB, Duration: {format_time(epoch_duration)}")

            # Log epoch summary metrics
            epoch_summary_metrics = {
                "global_step": current_global_step, # Use last step of the epoch
                f"{log_prefix_phase}/Train_Loss_EpochAvg": avg_epoch_loss,
                f"{log_prefix_phase}/Peak_GPU_Mem_Epoch_MiB": peak_mem_epoch,
            }
            log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)

        # --- End of Epoch Loop for Layer Phase ---

        # --- Checkpointing (Optional: after each phase) ---
        if checkpoint_dir:
            create_directory_if_not_exists(checkpoint_dir)
            chkpt_filename = f"mf_phase_{layer_focus_idx}_complete.pth"
            # It's safer to save the optimizer state dict as well with the model
            save_checkpoint(
                state={
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), # Save optimizer state
                    "layer_phase_trained": layer_focus_idx
                },
                is_best=False, filename=chkpt_filename, checkpoint_dir=checkpoint_dir,
            )
            logger.info(f"Saved checkpoint after completing phase for {log_prefix_phase}")

    # --- End of Layer Phase Loop ---

    # --- Final cleanup ---
    # Set all trained parameters back to requires_grad=False? Optional, depends on evaluation needs.
    # model.eval() ensures dropout/BN are off.
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False) # Ensure grads are off for final eval

    logger.info(f"Finished all layer-wise MF training phases. Total time: {format_time(time.time() - overall_start_time)}")
    return peak_mem_train # Return overall peak memory


# --- evaluate_mf_model (No changes needed from previous version, ensure adapter logic) ---
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

    # Ensure adapter is defined if needed (e.g., for flattening)
    if input_adapter is None:
        input_adapter = lambda x: x.view(x.shape[0], -1)
        logger.debug("Using default flattening adapter during MF evaluation.")

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating MF (BP-style)", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            # Apply input adapter before getting activations
            eval_input = input_adapter(images).to(device) # Ensure output is on device

            all_activations = model.forward_with_intermediate_activations(eval_input)

            if len(all_activations) <= last_activation_index:
                logger.error(f"Activation list len ({len(all_activations)}) too short for a_{last_activation_index}.")
                continue

            last_hidden_activation = all_activations[last_activation_index].to(device) # Ensure activation is on device
            last_projection_matrix = last_projection_matrix.to(device) # Ensure matrix is on device

            try:
                goodness_scores = torch.matmul(last_hidden_activation, last_projection_matrix.t())
                predicted_labels = torch.argmax(goodness_scores, dim=1)

                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)
            except Exception as e:
                logger.error(f"Error during MF evaluation prediction/comparison: {e}", exc_info=True)
                # Skip batch or handle error appropriately

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"MF Evaluation Results (BP-style): Accuracy: {accuracy:.2f}%")
    # MF evaluation doesn't typically calculate a loss in this manner
    results = {"eval_accuracy": accuracy, "eval_loss": float("nan")}
    return results