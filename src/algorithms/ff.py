import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List, Callable
import functools
import os  # For checkpointing
import time # For calculation step timing

# Ensure FF_MLP is imported correctly
from src.architectures.ff_mlp import FF_MLP
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint, format_time # Import checkpoint helper

logger = logging.getLogger(__name__)


def ff_loss_fn(
    pos_goodness: torch.Tensor, neg_goodness: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Calculates the Forward-Forward loss for a layer using the logistic function (implemented via softplus).
    The objective is to make positive goodness high (> threshold) and negative goodness low (< threshold).
    Loss = mean( softplus(-(pos_goodness - threshold)) + softplus(neg_goodness - threshold) )
           (log(1+exp(-x)) is numerically more stable than log(sigmoid(x)))
    """
    # --- Input Validation ---
    if not isinstance(pos_goodness, torch.Tensor) or not isinstance(
        neg_goodness, torch.Tensor
    ):
        raise TypeError("Goodness inputs must be PyTorch Tensors.")
    if pos_goodness.shape != neg_goodness.shape:
        raise ValueError(
            f"Positive ({pos_goodness.shape}) and negative ({neg_goodness.shape}) goodness tensors must have the same shape."
        )

    # Handle potential higher dimensions gracefully, though typically goodness is [B]
    if pos_goodness.dim() > 1:
        logger.warning(
            f"Goodness tensors have dimension {pos_goodness.dim()}, expected 1 (batch size). Loss calculated element-wise then averaged."
        )
    # --- Loss Calculation ---
    # softplus(x) = log(1 + exp(x))
    loss_pos = F.softplus(
        -(pos_goodness - threshold)
    )  # Encourages pos_goodness > threshold
    loss_neg = F.softplus(
        neg_goodness - threshold
    )  # Encourages neg_goodness < threshold
    loss = torch.mean(
        loss_pos + loss_neg
    )  # Average over the batch (and potentially other dims if input > 1D)

    return loss


def create_ff_pixel_label_input(
    original_images: torch.Tensor,  # Shape [B, C, H, W]
    labels: torch.Tensor,  # Shape [B]
    num_classes: int,
    replace_value_on: float = 1.0,  # Value for pixels corresponding to the correct class
    replace_value_off: float = 0.0,  # Value for other pixels in the label area
) -> torch.Tensor:
    """
    Creates FF input by replacing the initial pixels of the image (flattened view)
    with a representation of the label. This implementation replaces the first
    `num_classes` pixels across all channels identically.

    Args:
        original_images: Batch of original images.
        labels: Batch of corresponding labels.
        num_classes: Total number of classes.
        replace_value_on: Pixel value for the 'on' state in the label representation.
        replace_value_off: Pixel value for the 'off' state in the label representation.

    Returns:
        A tensor containing the modified images, flattened. Shape [B, C*H*W].
    """
    batch_size, channels, height, width = original_images.shape
    device = original_images.device  # Use input device

    if num_classes > height * width:
        raise ValueError(
            f"num_classes ({num_classes}) is larger than the number of pixels per channel ({height*width}). Cannot embed label."
        )

    # Create one-hot labels [B, num_classes] on the correct device
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).to(
        device=device, dtype=torch.float
    )

    # Create the label patch [B, num_classes] using specified values
    label_patch = torch.where(one_hot_labels == 1, replace_value_on, replace_value_off)

    # Make a copy of the original images to modify
    modified_images = original_images.clone()

    # Calculate row and column indices for the first num_classes pixels (row-major)
    pixels_to_replace = label_patch.view(batch_size, num_classes)  # [B, num_classes]
    row_indices = torch.arange(num_classes, device=device) // width
    col_indices = torch.arange(num_classes, device=device) % width

    # Ensure indices are within bounds (should be due to check above)
    if torch.any(row_indices >= height) or torch.any(col_indices >= width):
        # This should not happen if the initial check passed
        raise RuntimeError("Calculated pixel indices are out of image bounds.")

    # Apply the replacement across all channels and the batch
    # This broadcasts the replacement across the channel dimension.
    try:
        modified_images[:, :, row_indices, col_indices] = pixels_to_replace.unsqueeze(
            1
        )  # Unsqueeze to add channel dim for broadcasting -> [B, 1, num_classes]
    except Exception as e:
        logger.error(f"Error during pixel replacement: {e}", exc_info=True)
        logger.error(
            f"Shapes - modified_images: {modified_images.shape}, pixels_to_replace unsqueezed: {pixels_to_replace.unsqueeze(1).shape}, row_indices: {row_indices.shape}, col_indices: {col_indices.shape}"
        )
        raise

    # Flatten the modified images
    flattened_modified_images = modified_images.view(batch_size, -1)  # [B, C*H*W]

    return flattened_modified_images


def generate_ff_pos_neg_pixel_data(
    base_images: torch.Tensor,  # [B, C, H, W]
    base_labels: torch.Tensor,  # [B]
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates positive and negative flattened image tensors for FF training
    using the pixel replacement method. Ensures negative label is different.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (pos_flattened_images, neg_flattened_images)
                                           Shape: [B, C*H*W] each.
    """
    batch_size = base_images.shape[0]
    device = base_images.device

    # 1. Create positive data (image + correct label)
    pos_flattened_images = create_ff_pixel_label_input(
        base_images, base_labels, num_classes
    )

    # 2. Create distinct negative labels
    # Generate random offsets (1 to num_classes-1)
    rand_offset = torch.randint(
        1, num_classes, (batch_size,), device=device, dtype=torch.long
    )
    neg_labels = (base_labels + rand_offset) % num_classes

    # Ensure negative label is different from positive label (handles edge case if offset makes it wrap around)
    collision = neg_labels == base_labels
    retries = 0
    max_retries = 5  # Prevent infinite loop in unlikely scenarios
    while torch.any(collision) and retries < max_retries:
        # For colliding indices, generate a new random offset and recalculate
        num_collisions = collision.sum().item()
        new_rand_offset = torch.randint(
            1, num_classes, (num_collisions,), device=device, dtype=torch.long
        )
        neg_labels[collision] = (base_labels[collision] + new_rand_offset) % num_classes
        # Recheck for collisions
        collision = neg_labels == base_labels
        retries += 1
    if retries == max_retries and torch.any(collision):
        logger.warning(
            f"Could not guarantee distinct negative labels after {max_retries} retries for {collision.sum().item()} samples."
        )
        # Handle remaining collisions, e.g., by incrementing by 1 again
        neg_labels[collision] = (neg_labels[collision] + 1) % num_classes

    # 3. Create negative data (image + incorrect label)
    neg_flattened_images = create_ff_pixel_label_input(
        base_images, neg_labels, num_classes
    )

    return pos_flattened_images, neg_flattened_images


def train_ff_layer(
    model: FF_MLP,
    layer_module: nn.Module,
    is_input_adapter_layer: bool,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    get_layer_input_fn: Callable[
        [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ],
    threshold: float,
    epochs: int,
    device: torch.device,
    layer_index: int, # 0-based index of the effective hidden layer
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    max_step_tracker: Dict[str, int] = {"value": 0}, # Use dict for mutable tracking
) -> None:
    """
    Trains a single effective layer (input adapter or subsequent FF_Layer) of the FF_MLP.
    Updates weights locally based on the FF loss calculated from positive/negative goodness.
    """
    layer_module.train()  # Set the specific module being trained to train mode
    layer_module.to(device)
    logger.info(
        f"Starting FF training for Layer {layer_index + 1}"
    )  # Log 1-based index

    total_steps_per_epoch = len(train_loader)
    # Global step offset calculation remains the same conceptually
    global_step_offset = layer_index * epochs * total_steps_per_epoch

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"FF Layer {layer_index+1} Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            # Ensure raw data is on device before passing to input fn
            images, labels = images.to(device), labels.to(device)

            # 1. Get detached positive/negative input activations for the current layer
            try:
                pos_activation_input, neg_activation_input = get_layer_input_fn(
                    images, labels
                )
            except Exception as e:
                logger.error(
                    f"Error getting layer input at Layer {layer_index+1}, Epoch {epoch+1}, Batch {batch_idx}: {e}",
                    exc_info=True,
                )
                continue

            pos_activation_input = pos_activation_input.to(device)
            neg_activation_input = neg_activation_input.to(device)

            # 2. Forward pass through current layer to get goodness (before normalization)
            pos_goodness: torch.Tensor
            neg_goodness: torch.Tensor

            try:
                if is_input_adapter_layer:
                    pos_lin = layer_module(pos_activation_input)
                    neg_lin = layer_module(neg_activation_input)
                    pos_act = model.first_layer_activation(pos_lin)
                    neg_act = model.first_layer_activation(neg_lin)
                    pos_goodness = torch.sum(pos_act.pow(2), dim=1)
                    neg_goodness = torch.sum(neg_act.pow(2), dim=1)
                else:
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
                continue

            # 3. Calculate loss based on goodness
            try:
                loss = ff_loss_fn(pos_goodness, neg_goodness, threshold)
            except Exception as e:
                logger.error(
                    f"Error calculating FF loss at Layer {layer_index+1}, Epoch {epoch+1}, Batch {batch_idx}: {e}",
                    exc_info=True,
                )
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"NaN or Inf loss detected at Layer {layer_index+1}, Epoch {epoch+1}, Batch {batch_idx}. Stopping layer training."
                )
                break

            # 4. Backpropagate gradients locally and optimize ONLY the current layer_module
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Logging and Progress ---
            epoch_loss += loss.item()
            # Calculate the *true* global step
            current_global_step = (
                global_step_offset + epoch * total_steps_per_epoch + batch_idx + 1 # Use 1-based batch index
            )
            max_step_tracker["value"] = max(max_step_tracker["value"], current_global_step) # Update tracker

            if (batch_idx + 1) % log_interval == 0 or batch_idx == total_steps_per_epoch - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.4f}")
                metrics = {
                    f"FF/Layer_{layer_index+1}/Train_Loss_Batch": avg_loss_batch,
                    f"FF/Layer_{layer_index+1}/Pos_Goodness_Mean": pos_goodness.mean().item(),
                    f"FF/Layer_{layer_index+1}/Neg_Goodness_Mean": neg_goodness.mean().item(),
                }
                log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

        # --- End of Epoch ---
        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)):
            logger.error(
                f"Terminating training for Layer {layer_index+1} due to invalid loss in epoch {epoch+1}."
            )
            break

        avg_epoch_loss = (
            epoch_loss / total_steps_per_epoch if total_steps_per_epoch > 0 else 0.0
        )
        # Log epoch loss AT the step corresponding to the end of the epoch
        epoch_end_step = global_step_offset + (epoch + 1) * total_steps_per_epoch
        max_step_tracker["value"] = max(max_step_tracker["value"], epoch_end_step) # Update tracker

        logger.info(
            f"FF Layer {layer_index+1} Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}"
        )
        log_metrics(
            {f"FF/Layer_{layer_index+1}/Train_Loss_Epoch": avg_epoch_loss},
            step=epoch_end_step, # Log at the correct end step
            wandb_run=wandb_run,
        )

    logger.info(f"Finished FF training for Layer {layer_index + 1}")
    layer_module.eval()


def train_ff_model(
    model_instance: FF_MLP,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    max_step_tracker: Dict[str, int] = {"value": 0}, # Pass down the tracker
    # input_adapter: Optional[Callable] = None, # REMOVED - Not needed here
):
    """
    Orchestrates the layer-wise training of an FF_MLP model using pixel label embedding.
    """
    model = model_instance
    model.to(device)
    num_hidden_layers = len(
        model.hidden_dims
    )
    logger.info(
        f"Starting layer-wise FF training for {num_hidden_layers} hidden layers (using pixel label embedding)."
    )

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

    # --- Define Input Generation Functions (Closures capturing 'model') ---
    @torch.no_grad()
    def get_initial_input_data(images, labels):
        pos_flat, neg_flat = generate_ff_pos_neg_pixel_data(
            images, labels, model.num_classes
        )
        return pos_flat.to(device), neg_flat.to(device)

    def create_layer_input_closure(model_ref: FF_MLP, prev_layer_idx: int):
        if not (0 <= prev_layer_idx < len(model_ref.hidden_dims)):
            raise ValueError(f"Invalid prev_layer_idx {prev_layer_idx}")

        @torch.no_grad()
        def get_layer_input(images, labels):
            pos_flattened, neg_flattened = generate_ff_pos_neg_pixel_data(
                images, labels, model_ref.num_classes
            )
            pos_flattened, neg_flattened = pos_flattened.to(device), neg_flattened.to(device)
            pos_input_current = model_ref.forward_upto(pos_flattened, prev_layer_idx)
            neg_input_current = model_ref.forward_upto(neg_flattened, prev_layer_idx)
            return pos_input_current.detach().to(device), neg_input_current.detach().to(device)
        return get_layer_input

    # --- Train Layer by Layer ---
    current_layer_input_fn = get_initial_input_data

    # 1. Train Input Adapter Layer (Effective Hidden Layer 0)
    logger.info(f"--- Training Input Adapter Layer (Layer 1/{num_hidden_layers}) ---")
    layer_module_0 = model.input_adapter_layer
    params_0 = list(layer_module_0.parameters())
    if params_0:
        optimizer_0_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_params_extra}
        optimizer_0 = getattr(optim, optimizer_type)(params_0, **optimizer_0_kwargs)

        train_ff_layer(
            model=model,
            layer_module=layer_module_0,
            is_input_adapter_layer=True,
            optimizer=optimizer_0,
            train_loader=train_loader,
            get_layer_input_fn=current_layer_input_fn,
            threshold=threshold,
            epochs=epochs_per_layer,
            device=device,
            layer_index=0,
            wandb_run=wandb_run,
            log_interval=log_interval,
            max_step_tracker=max_step_tracker, # Pass tracker
        )
        for param in params_0: param.requires_grad = False
        layer_module_0.eval()
        if checkpoint_dir:
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained": 0},
                is_best=False, filename=f"ff_layer_0_complete.pth", checkpoint_dir=checkpoint_dir,
            )
    else: logger.warning("Input adapter layer has no parameters.")

    current_layer_input_fn = create_layer_input_closure(model, prev_layer_idx=0)

    # 2. Train Subsequent Hidden FF_Layers
    for i in range(len(model.layers)):
        effective_layer_index = i + 1
        layer_log_index = effective_layer_index + 1
        logger.info(f"--- Training Hidden FF_Layer {layer_log_index}/{num_hidden_layers} ---")

        ff_layer_module = model.layers[i]
        params_i = list(ff_layer_module.parameters())
        if not params_i:
            logger.warning(f"Hidden FF_Layer {layer_log_index} has no parameters.")
        else:
            optimizer_i_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_params_extra}
            optimizer_i = getattr(optim, optimizer_type)(params_i, **optimizer_i_kwargs)

            train_ff_layer(
                model=model,
                layer_module=ff_layer_module,
                is_input_adapter_layer=False,
                optimizer=optimizer_i,
                train_loader=train_loader,
                get_layer_input_fn=current_layer_input_fn,
                threshold=threshold,
                epochs=epochs_per_layer,
                device=device,
                layer_index=effective_layer_index,
                wandb_run=wandb_run,
                log_interval=log_interval,
                max_step_tracker=max_step_tracker, # Pass tracker
            )
            for param in params_i: param.requires_grad = False
            ff_layer_module.eval()
            if checkpoint_dir:
                save_checkpoint(
                    state={"state_dict": model.state_dict(), "layer_trained": effective_layer_index},
                    is_best=False, filename=f"ff_layer_{effective_layer_index}_complete.pth", checkpoint_dir=checkpoint_dir,
                )

        if effective_layer_index < num_hidden_layers - 1:
            current_layer_input_fn = create_layer_input_closure(model, prev_layer_idx=effective_layer_index)

    logger.info("Finished all layer-wise FF training.")


def evaluate_ff_model(
    model_instance: FF_MLP,
    data_loader: DataLoader,
    device: torch.device,
    **kwargs,
) -> Dict[str, float]:
    """
    Evaluates the trained FF_MLP model using multi-pass inference.
    """
    model = model_instance
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
            batch_total_goodness = torch.zeros((batch_size, num_classes), device=device)

            for label_candidate in range(num_classes):
                candidate_labels = torch.full((batch_size,), label_candidate, dtype=torch.long, device=device)
                try:
                    ff_input_candidate = create_ff_pixel_label_input(
                        images, candidate_labels, num_classes
                    )
                except Exception as e:
                    logger.error(f"Error creating pixel input for candidate {label_candidate}: {e}", exc_info=True)
                    batch_total_goodness[:, label_candidate] = -torch.inf
                    continue
                try:
                    layer_goodness_list = model.forward_goodness_per_layer(ff_input_candidate)
                except Exception as e:
                    logger.error(f"Error in FF goodness forward pass for candidate {label_candidate}: {e}", exc_info=True)
                    batch_total_goodness[:, label_candidate] = -torch.inf
                    continue
                if not layer_goodness_list:
                    total_goodness_candidate = torch.zeros((batch_size,), device=device)
                else:
                    try:
                        total_goodness_candidate = torch.stack(layer_goodness_list, dim=0).sum(dim=0)
                        if total_goodness_candidate.shape != (batch_size,):
                            raise ValueError(f"Unexpected shape after summing goodness: {total_goodness_candidate.shape}")
                    except Exception as e:
                        logger.error(f"Error stacking/summing goodness for candidate {label_candidate}: {e}", exc_info=True)
                        batch_total_goodness[:, label_candidate] = -torch.inf
                        continue
                batch_total_goodness[:, label_candidate] = total_goodness_candidate

            try:
                all_inf_mask = torch.all(torch.isinf(batch_total_goodness), dim=1)
                if torch.any(all_inf_mask):
                    num_all_inf = all_inf_mask.sum().item()
                    logger.warning(f"Batch {batch_idx+1}: {num_all_inf} samples had -inf goodness for all candidates. Predicting 0 for these.")
                    predicted_labels = torch.zeros_like(labels)
                    if not torch.all(all_inf_mask):
                        predicted_labels[~all_inf_mask] = torch.argmax(batch_total_goodness[~all_inf_mask], dim=1)
                else:
                    predicted_labels = torch.argmax(batch_total_goodness, dim=1)
            except Exception as e:
                logger.error(f"Error during argmax prediction for batch {batch_idx+1}: {e}", exc_info=True)
                predicted_labels = torch.zeros_like(labels)

            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"FF Evaluation Accuracy (Pixel Embedding): {accuracy:.2f}%")
    results = {"eval_accuracy": accuracy, "eval_loss": float("nan")}
    return results