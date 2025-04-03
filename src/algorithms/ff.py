# File: src/algorithms/ff.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
import pynvml # Import for type hint
from typing import Dict, Any, Optional, Tuple, List, Callable
import functools
import os
import time

from src.architectures.ff_mlp import FF_MLP
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint, format_time, create_directory_if_not_exists
from src.utils.monitoring import get_gpu_memory_usage # Import memory usage function

logger = logging.getLogger(__name__)

# --- ff_loss_fn (no changes) ---
def ff_loss_fn(
    pos_goodness: torch.Tensor, neg_goodness: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Calculates the Forward-Forward loss for a layer using the logistic function (implemented via softplus).
    """
    if not isinstance(pos_goodness, torch.Tensor) or not isinstance(
        neg_goodness, torch.Tensor
    ):
        raise TypeError("Goodness inputs must be PyTorch Tensors.")
    if pos_goodness.shape != neg_goodness.shape:
        raise ValueError(
            f"Positive ({pos_goodness.shape}) and negative ({neg_goodness.shape}) goodness tensors must have the same shape."
        )
    if pos_goodness.dim() > 1:
        logger.warning(
            f"Goodness tensors have dimension {pos_goodness.dim()}, expected 1 (batch size). Loss calculated element-wise then averaged."
        )
    # Loss based on pushing pos_goodness > threshold and neg_goodness < threshold
    # softplus(x) = log(1 + exp(x))
    # Loss_pos = log(1 + exp(-(pos_goodness - threshold))) -> small when pos_goodness > threshold
    # Loss_neg = log(1 + exp(neg_goodness - threshold))   -> small when neg_goodness < threshold
    loss_pos = F.softplus(-(pos_goodness - threshold))
    loss_neg = F.softplus(neg_goodness - threshold)
    loss = torch.mean(loss_pos + loss_neg) # Average loss over the batch
    return loss

# --- create_ff_pixel_label_input (no changes) ---
def create_ff_pixel_label_input(
    original_images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    replace_value_on: float = 1.0,
    replace_value_off: float = 0.0,
) -> torch.Tensor:
    """
    Creates FF input by replacing the initial pixels of the image (flattened view)
    with a representation of the label. Mimics Hinton's paper (Sec 3.3) for MNIST.
    Assumes images are channel-first (B, C, H, W). Replaces pixels in channel 0.
    """
    batch_size, channels, height, width = original_images.shape
    device = original_images.device

    # The number of pixels needed is num_classes. Check if it fits in one channel.
    if num_classes > height * width:
        raise ValueError(
            f"num_classes ({num_classes}) is larger than the number of pixels per channel ({height*width}). Cannot embed label."
        )

    # Create one-hot labels [B, num_classes]
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).to(
        device=device, dtype=torch.float
    )
    # Map one-hot 1s and 0s to desired pixel values (e.g., 1.0 and 0.0)
    label_patch = torch.where(one_hot_labels == 1, replace_value_on, replace_value_off)

    # Clone the original images to avoid modifying them in place
    modified_images = original_images.clone()

    # Get the flattened view of the first channel [B, H*W]
    first_channel_flat = modified_images[:, 0, :, :].view(batch_size, -1)

    # Replace the first 'num_classes' pixels with the label patch
    first_channel_flat[:, :num_classes] = label_patch

    # Reshape back and update the first channel
    modified_images[:, 0, :, :] = first_channel_flat.view(batch_size, height, width)

    # Flatten the entire modified image (all channels) for MLP input [B, C*H*W]
    flattened_modified_images = modified_images.view(batch_size, -1)
    return flattened_modified_images

# --- generate_ff_pos_neg_pixel_data (no changes) ---
def generate_ff_pos_neg_pixel_data(
    base_images: torch.Tensor,
    base_labels: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates positive and negative flattened image tensors for FF training
    using the pixel replacement method. Ensures negative label is different.
    """
    batch_size = base_images.shape[0]
    device = base_images.device

    # Positive examples: original image with correct label embedded
    pos_flattened_images = create_ff_pixel_label_input(
        base_images, base_labels, num_classes
    )

    # Negative examples: original image with incorrect label embedded
    # Generate random offsets (1 to num_classes-1) to add to the true label
    rand_offset = torch.randint(
        1, num_classes, (batch_size,), device=device, dtype=torch.long
    )
    # Calculate potential negative labels using modulo arithmetic
    neg_labels = (base_labels + rand_offset) % num_classes

    # Ensure negative label is truly different from the positive label
    collision = neg_labels == base_labels
    retries = 0
    max_retries = 5 # Safety mechanism
    while torch.any(collision) and retries < max_retries:
        num_collisions = collision.sum().item()
        # Only regenerate offsets for the colliding samples
        new_rand_offset = torch.randint(
            1, num_classes, (num_collisions,), device=device, dtype=torch.long
        )
        neg_labels[collision] = (base_labels[collision] + new_rand_offset) % num_classes
        collision = neg_labels == base_labels # Check again
        retries += 1

    # If collisions persist after retries (highly unlikely for reasonable num_classes), force a change
    if retries == max_retries and torch.any(collision):
        logger.warning(
            f"Could not guarantee distinct negative labels after {max_retries} retries for {collision.sum().item()} samples. Forcing difference."
        )
        neg_labels[collision] = (neg_labels[collision] + 1) % num_classes # Simple increment wrap-around

    # Create negative images with the ensured incorrect labels
    neg_flattened_images = create_ff_pixel_label_input(
        base_images, neg_labels, num_classes
    )

    return pos_flattened_images, neg_flattened_images


# --- train_ff_layer (no changes) ---
def train_ff_layer(
    model: FF_MLP, # Pass the whole model for accessing activation layers
    layer_module: nn.Module, # The specific Linear layer being trained
    is_input_adapter_layer: bool, # Flag to handle activation differently for layer 0
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    get_layer_input_fn: Callable[
        [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ], # Function returning (pos_input, neg_input) for this layer
    threshold: float,
    epochs: int,
    device: torch.device,
    layer_index: int, # 0-based index of the effective hidden layer being trained
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    step_ref: List[int] = [-1], # Reference for global step counter
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, # For memory monitoring
    nvml_active: bool = False, # Status of NVML
) -> Tuple[float, float]: # Returns avg loss of last epoch, peak memory during training
    """
    Trains a single effective layer (input adapter or subsequent FF_Layer) of the FF_MLP.
    Updates weights locally based on the FF loss calculated from positive/negative goodness.
    Returns the average loss of the last epoch and the peak memory observed during its training.
    """
    layer_module.train() # Set the specific linear module to train mode
    layer_module.to(device)
    logger.info(
        f"Starting FF training for Layer {layer_index + 1} ({'Input Adapter' if is_input_adapter_layer else 'Hidden'})"
    )
    peak_mem_layer_train = 0.0 # Track peak memory for this layer's training phase
    last_epoch_avg_loss = float('nan')

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        peak_mem_layer_epoch = 0.0 # Track peak memory for this specific epoch
        pbar = tqdm(
            train_loader,
            desc=f"FF Layer {layer_index+1} Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1
            current_global_step = step_ref[0]

            images, labels = images.to(device), labels.to(device)

            try:
                # Get the appropriate input activations (normalized from previous layer)
                pos_activation_input, neg_activation_input = get_layer_input_fn(
                    images, labels
                )
            except Exception as e:
                logger.error(
                    f"Error getting layer input at Layer {layer_index+1}, Epoch {epoch+1}, Batch {batch_idx}: {e}",
                    exc_info=True,
                )
                continue # Skip batch if input generation fails

            pos_activation_input = pos_activation_input.to(device)
            neg_activation_input = neg_activation_input.to(device)

            pos_goodness: torch.Tensor
            neg_goodness: torch.Tensor

            # Calculate goodness based on the layer being trained
            try:
                if is_input_adapter_layer:
                    # Layer 0: Pass input through the linear layer, then model's first activation
                    pos_lin = layer_module(pos_activation_input)
                    neg_lin = layer_module(neg_activation_input)
                    pos_act = model.first_layer_activation(pos_lin) # Use model's activation
                    neg_act = model.first_layer_activation(neg_lin)
                    # Goodness is sum of squares of these activations (before normalization)
                    pos_goodness = torch.sum(pos_act.pow(2), dim=1)
                    neg_goodness = torch.sum(neg_act.pow(2), dim=1)
                else:
                    # Subsequent layers: Use the FF_Layer's forward_with_goodness method
                    # which calculates goodness internally before normalization.
                    # The layer_module here is an FF_Layer instance.
                    _, pos_goodness = layer_module.forward_with_goodness( # type: ignore
                        pos_activation_input
                    )
                    _, neg_goodness = layer_module.forward_with_goodness( # type: ignore
                        neg_activation_input
                    )
            except Exception as e:
                logger.error(
                    f"Error during forward/goodness calculation at Layer {layer_index+1}, Epoch {epoch+1}, Batch {batch_idx}: {e}",
                    exc_info=True,
                )
                continue # Skip batch if forward pass fails

            # Calculate loss
            try:
                loss = ff_loss_fn(pos_goodness, neg_goodness, threshold)
            except Exception as e:
                logger.error(
                    f"Error calculating FF loss at Layer {layer_index+1}, Epoch {epoch+1}, Batch {batch_idx}: {e}",
                    exc_info=True,
                )
                continue # Skip batch if loss calculation fails

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"NaN or Inf loss detected at Layer {layer_index+1}, Epoch {epoch+1}, Batch {batch_idx}. Stopping layer training."
                )
                break # Exit epoch loop if loss is invalid

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # Updates only the parameters passed to this layer's optimizer

            # Accumulate loss for epoch average
            batch_size = images.size(0) # Use input batch size for tracking samples
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

            # Log batch metrics periodically
            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.4f}")
                metrics_to_log = {
                    "global_step": current_global_step,
                    f"FF/Layer_{layer_index+1}/Train_Loss_Batch": avg_loss_batch,
                    f"FF/Layer_{layer_index+1}/Pos_Goodness_Mean": pos_goodness.mean().item(),
                    f"FF/Layer_{layer_index+1}/Neg_Goodness_Mean": neg_goodness.mean().item(),
                }
                if not torch.isnan(torch.tensor(current_mem_used)): # Only log if valid
                    metrics_to_log[f"FF/Layer_{layer_index+1}/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)

        # Check again if loss became NaN/Inf during the epoch loop exit
        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)):
            logger.error(
                f"Terminating training for Layer {layer_index+1} due to invalid loss in epoch {epoch+1}."
            )
            break # Exit epoch loop prematurely

        # --- Calculate epoch loss and update layer peak memory ---
        last_epoch_avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float('nan')
        peak_mem_layer_train = max(peak_mem_layer_train, peak_mem_layer_epoch) # Update peak for the entire layer training duration

        logger.info(
            f"FF Layer {layer_index+1} Epoch {epoch+1}/{epochs} - Avg Loss: {last_epoch_avg_loss:.4f}, Peak Mem Epoch: {peak_mem_layer_epoch:.1f} MiB"
        )

    # Sample memory one last time at the very end of layer training
    if nvml_active and gpu_handle:
        mem_info = get_gpu_memory_usage(gpu_handle)
        if mem_info:
            peak_mem_layer_train = max(peak_mem_layer_train, mem_info[0])

    logger.info(f"Finished FF training for Layer {layer_index + 1}. Overall Peak Mem for Layer: {peak_mem_layer_train:.1f} MiB")
    layer_module.eval() # Set layer back to eval mode after training
    return last_epoch_avg_loss, peak_mem_layer_train # Return last avg loss and peak mem


# --- train_ff_model (MODIFIED - Added comment clarification) ---
def train_ff_model(
    model: FF_MLP, # The FF_MLP instance to be trained layer-wise
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None, # Added for signature consistency, but FF uses pixel embedding.
    step_ref: List[int] = [-1], # Mutable list reference for global step counter
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, # Handle for monitoring
    nvml_active: bool = False, # NVML status
) -> float: # Returns overall peak memory observed during training
    """
    Orchestrates the layer-wise training of an FF_MLP model using pixel label embedding.
    Logs layer summary metrics including peak memory for the layer.
    Returns the overall peak GPU memory observed across all layer training phases.
    """
    model.to(device)
    num_hidden_layers = len(model.hidden_dims)
    logger.info(
        f"Starting layer-wise FF training for {num_hidden_layers} hidden layers (using pixel label embedding)."
    )
    # Clarification: input_adapter is part of the general training engine signature,
    # but FF uses its internal pixel embedding method instead.
    if input_adapter is not None:
        logger.warning("FF Training: 'input_adapter' provided but FF uses internal pixel embedding. Adapter will not be used by FF training logic.")

    # --- Get Training Configuration ---
    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_type = algo_config.get("optimizer_type", "Adam")
    lr = algo_config.get("lr", 0.001)
    weight_decay = algo_config.get("weight_decay", 0.0)
    threshold = algo_config.get("threshold", 1.0)
    epochs_per_layer = algo_config.get("epochs_per_layer", 10)
    log_interval = algo_config.get("log_interval", 100)
    optimizer_params_extra = algo_config.get("optimizer_params", {})
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None)

    logger.info(
        f"FF Parameters: Optimizer={optimizer_type}, LR={lr}, WD={weight_decay}, Threshold={threshold}, Epochs/Layer={epochs_per_layer}"
    )

    # --- Define Input Generation Functions (Closures) ---

    # Function to get initial pos/neg pixel-embedded inputs for training Layer 0
    @torch.no_grad()
    def get_initial_input_data(images, labels):
        pos_flat, neg_flat = generate_ff_pos_neg_pixel_data(
            images, labels, model.num_classes
        )
        return pos_flat.to(device), neg_flat.to(device)

    # Function factory: creates a function to get input for layer `i+1`
    # based on the output of layer `i` (using model.forward_upto)
    def create_layer_input_closure(model_ref: FF_MLP, prev_layer_idx: int):
        if not (0 <= prev_layer_idx < len(model_ref.hidden_dims)):
            raise ValueError(f"Invalid prev_layer_idx {prev_layer_idx}")

        @torch.no_grad()
        def get_layer_input(images, labels):
            # Generate base pos/neg pixel-embedded inputs first
            pos_flattened, neg_flattened = generate_ff_pos_neg_pixel_data(
                images, labels, model_ref.num_classes
            )
            pos_flattened, neg_flattened = pos_flattened.to(device), neg_flattened.to(device)

            # Ensure model is on the correct device for forward_upto
            model_ref.to(device)
            model_ref.eval() # Ensure model is in eval mode for this forward pass

            # Run forward pass up to the *output* of the previous layer
            # This output is normalized and serves as input for the current layer training
            pos_input_current = model_ref.forward_upto(pos_flattened, prev_layer_idx)
            neg_input_current = model_ref.forward_upto(neg_flattened, prev_layer_idx)

            # Detach to prevent gradients flowing back further than the current layer
            return pos_input_current.detach().to(device), neg_input_current.detach().to(device)
        return get_layer_input

    # --- Train Layer by Layer ---
    current_layer_input_fn = get_initial_input_data
    peak_mem_train = 0.0 # Track overall peak memory

    # 1. Train Input Adapter Layer (Effective Hidden Layer 0)
    logger.info(f"--- Training Input Adapter Layer (Layer 1/{num_hidden_layers}) ---")
    layer_module_0 = model.input_adapter_layer # The Linear layer
    params_0 = list(layer_module_0.parameters())
    layer_0_peak_mem = 0.0
    final_avg_loss_layer_0 = float('nan')

    if params_0: # Only train if the layer has parameters
        optimizer_0_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_params_extra}
        optimizer_0 = getattr(optim, optimizer_type)(params_0, **optimizer_0_kwargs)

        # Train the layer, get loss and peak memory for this layer
        final_avg_loss_layer_0, layer_0_peak_mem = train_ff_layer(
            model=model, # Pass the model reference for activation access
            layer_module=layer_module_0,
            is_input_adapter_layer=True,
            optimizer=optimizer_0,
            train_loader=train_loader,
            get_layer_input_fn=current_layer_input_fn,
            threshold=threshold,
            epochs=epochs_per_layer,
            device=device,
            layer_index=0, # 0-based index
            wandb_run=wandb_run,
            log_interval=log_interval,
            step_ref=step_ref,
            gpu_handle=gpu_handle,
            nvml_active=nvml_active
        )
        peak_mem_train = max(peak_mem_train, layer_0_peak_mem) # Update overall peak

        # Freeze layer after training
        for param in params_0: param.requires_grad = False
        layer_module_0.eval()
    else:
         logger.warning("Input adapter layer has no parameters. Skipping training.")

    # Log layer summary after training completes
    current_global_step = step_ref[0] # Get step after training this layer
    layer_summary_metrics = {
        "global_step": current_global_step,
        f"FF/Layer_1/Train_Loss_LayerAvg": final_avg_loss_layer_0,
        f"FF/Layer_1/Peak_GPU_Mem_Layer_MiB": layer_0_peak_mem,
    }
    log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=True)
    logger.debug(f"Logged FF Layer 1 summary at global_step {current_global_step}")

    if checkpoint_dir and params_0: # Only save if trained
        create_directory_if_not_exists(checkpoint_dir)
        save_checkpoint(
            state={"state_dict": model.state_dict(), "layer_trained": 0},
            is_best=False, filename=f"ff_layer_0_complete.pth", checkpoint_dir=checkpoint_dir,
        )

    # Prepare the input function for the next layer (Layer 1)
    current_layer_input_fn = create_layer_input_closure(model, prev_layer_idx=0)

    # 2. Train Subsequent Hidden FF_Layers
    # model.layers contains FF_Layer instances for layers 1, 2, ..., L-1
    for i in range(len(model.layers)):
        effective_layer_index = i + 1 # This is layer 1, 2, ... L-1
        layer_log_index = effective_layer_index + 1 # For logging (1-based) -> 2, 3, ..., L
        logger.info(f"--- Training Hidden FF_Layer {layer_log_index}/{num_hidden_layers} ---")

        ff_layer_module = model.layers[i] # This is an FF_Layer instance
        params_i = list(ff_layer_module.parameters())
        layer_i_peak_mem = 0.0
        final_avg_loss_layer_i = float('nan')

        if not params_i:
            logger.warning(f"Hidden FF_Layer {layer_log_index} has no parameters.")
        else:
            optimizer_i_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_params_extra}
            optimizer_i = getattr(optim, optimizer_type)(params_i, **optimizer_i_kwargs)

            # Train this FF_Layer instance
            final_avg_loss_layer_i, layer_i_peak_mem = train_ff_layer(
                model=model, # Pass model reference (needed for activation if is_input_adapter was True)
                layer_module=ff_layer_module,
                is_input_adapter_layer=False, # These are subsequent layers
                optimizer=optimizer_i,
                train_loader=train_loader,
                get_layer_input_fn=current_layer_input_fn, # Gets normalized output from layer i
                threshold=threshold,
                epochs=epochs_per_layer,
                device=device,
                layer_index=effective_layer_index, # 0-based index of this layer
                wandb_run=wandb_run,
                log_interval=log_interval,
                step_ref=step_ref,
                gpu_handle=gpu_handle,
                nvml_active=nvml_active
            )
            peak_mem_train = max(peak_mem_train, layer_i_peak_mem) # Update overall peak

            # Freeze layer after training
            for param in params_i: param.requires_grad = False
            ff_layer_module.eval()

        # Log layer summary after training completes
        current_global_step = step_ref[0] # Get step after training this layer
        layer_summary_metrics = {
            "global_step": current_global_step,
            f"FF/Layer_{layer_log_index}/Train_Loss_LayerAvg": final_avg_loss_layer_i,
            f"FF/Layer_{layer_log_index}/Peak_GPU_Mem_Layer_MiB": layer_i_peak_mem,
        }
        log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=True)
        logger.debug(f"Logged FF Layer {layer_log_index} summary at global_step {current_global_step}")

        if checkpoint_dir and params_i: # Only save if trained
            create_directory_if_not_exists(checkpoint_dir)
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained": effective_layer_index},
                is_best=False, filename=f"ff_layer_{effective_layer_index}_complete.pth", checkpoint_dir=checkpoint_dir,
            )

        # Prepare input function for the *next* layer's training, if there is one
        if effective_layer_index < num_hidden_layers - 1:
            current_layer_input_fn = create_layer_input_closure(model, prev_layer_idx=effective_layer_index)

    logger.info("Finished all layer-wise FF training.")
    return peak_mem_train # Return overall peak memory


# --- evaluate_ff_model (no changes) ---
def evaluate_ff_model(
    model: FF_MLP, # The trained FF_MLP model
    data_loader: DataLoader,
    device: torch.device,
    **kwargs, # Accept potential criterion/adapter from engine but ignore them
) -> Dict[str, float]:
    """
    Evaluates the trained FF_MLP model using multi-pass inference (Hinton Sec 3.3).
    For each test image, create inputs with each possible label embedded, run forward
    to get goodness per layer, sum goodness across layers, and choose the label
    with the highest total goodness.
    Ignores criterion and input_adapter if passed.
    """
    model.eval()
    model.to(device)
    num_classes = model.num_classes
    logger.info(
        f"Evaluating FF model using multi-pass inference ({num_classes} passes per image, pixel embedding)."
    )

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating FF Model", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]
            # Store total goodness for each sample for each candidate label
            batch_total_goodness = torch.zeros((batch_size, num_classes), device=device)

            # Iterate through each possible label candidate
            for label_candidate in range(num_classes):
                # Create input with this candidate label embedded
                candidate_labels = torch.full((batch_size,), label_candidate, dtype=torch.long, device=device)
                try:
                    ff_input_candidate = create_ff_pixel_label_input(
                        images, candidate_labels, num_classes
                    )
                except Exception as e:
                    logger.error(f"Error creating pixel input for candidate {label_candidate}: {e}", exc_info=True)
                    # Assign a very low goodness if input creation fails
                    batch_total_goodness[:, label_candidate] = -torch.inf
                    continue # Skip to next candidate label

                # Perform forward pass to get goodness per layer for this candidate input
                try:
                    # This returns a list of goodness tensors (one per layer)
                    layer_goodness_list = model.forward_goodness_per_layer(ff_input_candidate)
                except Exception as e:
                    logger.error(f"Error in FF goodness forward pass for candidate {label_candidate}: {e}", exc_info=True)
                    batch_total_goodness[:, label_candidate] = -torch.inf
                    continue # Skip to next candidate label

                # Sum goodness across all layers for this candidate
                if not layer_goodness_list:
                    # Handle case with no layers (shouldn't happen with valid FF_MLP)
                    total_goodness_candidate = torch.zeros((batch_size,), device=device)
                else:
                    try:
                        # Stack goodness tensors along a new dimension (dim=0) and sum across that dimension
                        total_goodness_candidate = torch.stack(layer_goodness_list, dim=0).sum(dim=0)
                        # Shape should now be [batch_size]
                        if total_goodness_candidate.shape != (batch_size,):
                            raise ValueError(f"Unexpected shape after summing goodness: {total_goodness_candidate.shape}")
                    except Exception as e:
                        logger.error(f"Error stacking/summing goodness for candidate {label_candidate}: {e}", exc_info=True)
                        batch_total_goodness[:, label_candidate] = -torch.inf
                        continue # Skip to next candidate label

                # Store the total goodness for this label candidate
                batch_total_goodness[:, label_candidate] = total_goodness_candidate

            # Choose the label with the highest total goodness for each sample
            try:
                # Check for samples where all candidates resulted in -inf (error cases)
                all_inf_mask = torch.all(torch.isinf(batch_total_goodness) & (batch_total_goodness < 0), dim=1)
                if torch.any(all_inf_mask):
                    num_all_inf = all_inf_mask.sum().item()
                    logger.warning(f"Batch {batch_idx+1}: {num_all_inf} samples had -inf goodness for all candidates. Predicting 0 for these.")
                    # Initialize predictions with a default (e.g., 0)
                    predicted_labels = torch.zeros_like(labels)
                    # Only compute argmax for samples that had at least one valid goodness score
                    valid_indices = ~all_inf_mask
                    if torch.any(valid_indices): # Check if there are any valid samples
                        predicted_labels[valid_indices] = torch.argmax(batch_total_goodness[valid_indices], dim=1)
                else:
                    # If no all-inf rows, compute argmax for all samples
                    predicted_labels = torch.argmax(batch_total_goodness, dim=1)
            except Exception as e:
                logger.error(f"Error during argmax prediction for batch {batch_idx+1}: {e}", exc_info=True)
                # Default prediction to 0 in case of error during argmax
                predicted_labels = torch.zeros_like(labels)

            # Calculate correct predictions for the batch
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

    # Calculate final accuracy
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"FF Evaluation Accuracy (Pixel Embedding, Multi-Pass): {accuracy:.2f}%")
    # FF doesn't have a standard loss concept during this multi-pass evaluation
    results = {"eval_accuracy": accuracy, "eval_loss": float("nan")}
    return results