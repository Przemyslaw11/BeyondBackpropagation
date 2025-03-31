# File: src/algorithms/ff.py
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

# Ensure FF_MLP is imported correctly
from src.architectures.ff_mlp import FF_MLP
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint  # Import checkpoint helper

logger = logging.getLogger(__name__)


def ff_loss_fn(
    pos_goodness: torch.Tensor, neg_goodness: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Calculates the Forward-Forward loss for a layer using logistic loss (softplus).
    Loss = softplus(-(pos_goodness - threshold)) + softplus(neg_goodness - threshold)
    """
    # Ensure goodness tensors are valid before calculation
    if not isinstance(pos_goodness, torch.Tensor) or not isinstance(
        neg_goodness, torch.Tensor
    ):
        raise TypeError("Goodness inputs must be PyTorch Tensors.")
    if pos_goodness.shape != neg_goodness.shape:
        raise ValueError(
            "Positive and negative goodness tensors must have the same shape."
        )
    if pos_goodness.dim() != 1:
        logger.warning(
            f"Goodness tensors have dimension {pos_goodness.dim()}, expected 1 (batch size). Loss calculated per element."
        )
        # Handle potential higher dimensions by calculating loss element-wise then averaging
        loss_pos = F.softplus(-(pos_goodness - threshold))
        loss_neg = F.softplus(neg_goodness - threshold)
        return torch.mean(loss_pos + loss_neg)
    else:
        # Standard case: goodness is per-sample [B]
        loss_pos = F.softplus(-(pos_goodness - threshold))
        loss_neg = F.softplus(neg_goodness - threshold)
        loss = torch.mean(loss_pos + loss_neg)  # Mean over batch
        return loss


def create_ff_pixel_label_input(
    original_images: torch.Tensor,  # Shape [B, C, H, W]
    labels: torch.Tensor,  # Shape [B]
    num_classes: int,
    replace_value_on: float = 1.0,  # Value for pixels corresponding to the correct class
    replace_value_off: float = 0.0,  # Value for other pixels in the label area
) -> torch.Tensor:
    """
    Creates FF input by replacing the first 'num_classes' pixels of the image
    with a one-hot representation of the label.

    Args:
        original_images: Batch of original images.
        labels: Batch of corresponding labels.
        num_classes: Total number of classes.
        replace_value_on: Pixel value for the 'on' state in the one-hot representation.
        replace_value_off: Pixel value for the 'off' state in the one-hot representation.

    Returns:
        A tensor containing the modified images, flattened. Shape [B, C*H*W].
    """
    batch_size, channels, height, width = original_images.shape
    device = original_images.device

    if num_classes > height * width:
        raise ValueError(
            f"num_classes ({num_classes}) is larger than the number of pixels per channel ({height*width}). Cannot embed label."
        )

    # Create one-hot labels [B, num_classes]
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()  # Use float

    # Create the label patch [B, num_classes] using specified values
    # Where one_hot is 1, use replace_value_on, otherwise use replace_value_off
    label_patch = torch.where(one_hot_labels == 1, replace_value_on, replace_value_off)

    # Make a copy of the original images to modify
    modified_images = original_images.clone()

    # Replace the first 'num_classes' pixels in each channel
    # We need to reshape the label patch and assign it carefully
    pixels_to_replace = label_patch.view(batch_size, num_classes)  # [B, num_classes]

    # Calculate row and column indices for the first num_classes pixels
    # Example for row-major flattening:
    row_indices = torch.arange(num_classes, device=device) // width
    col_indices = torch.arange(num_classes, device=device) % width

    # Ensure indices are within bounds (should be due to check above)
    if torch.any(row_indices >= height) or torch.any(col_indices >= width):
        raise RuntimeError("Calculated pixel indices are out of image bounds.")

    # Apply the replacement across all channels and the batch
    # Expand pixels_to_replace to match the number of channels implicitly if needed,
    # or replicate it if replacement should be identical across channels.
    # Assuming replacement should be identical across channels:
    for c in range(channels):
        # This assignment is tricky with advanced indexing. A loop might be clearer/safer:
        for b in range(batch_size):
            modified_images[b, c, row_indices, col_indices] = pixels_to_replace[b]

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
    using the pixel replacement method.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (pos_flattened_images, neg_flattened_images)
                                           Shape: [B, C*H*W] each.
    """
    batch_size = base_images.shape[0]
    device = base_images.device

    # 1. Create positive data
    pos_flattened_images = create_ff_pixel_label_input(
        base_images, base_labels, num_classes
    )

    # 2. Create negative labels
    rand_offset = torch.randint(
        1, num_classes, (batch_size,), device=device, dtype=torch.long
    )
    neg_labels = (base_labels + rand_offset) % num_classes
    # Ensure negative label is different from positive label
    collision = neg_labels == base_labels
    if torch.any(collision):
        neg_labels[collision] = (neg_labels[collision] + 1) % num_classes

    # 3. Create negative data
    neg_flattened_images = create_ff_pixel_label_input(
        base_images, neg_labels, num_classes
    )

    return pos_flattened_images, neg_flattened_images


def train_ff_layer(
    layer_module: nn.Module,  # Can be nn.Linear or FF_Layer
    is_input_adapter_layer: bool,  # Flag to know which module we're training
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    get_layer_input_fn: Callable[
        [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ],  # Provides input activations
    threshold: float,
    epochs: int,
    device: torch.device,
    layer_index: int,  # Effective hidden layer index (0 for input adapter, 1 for first FF_Layer, etc.)
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
) -> None:
    """
    Trains a single effective layer (input adapter or FF_Layer) of the FF_MLP.

    Args:
        layer_module: The specific nn.Module instance to train (e.g., model.input_adapter_layer or model.layers[i]).
        is_input_adapter_layer: True if training the first layer (input adapter).
        optimizer: Optimizer configured for the layer_module's parameters.
        train_loader: DataLoader for the training data.
        get_layer_input_fn: Function that takes (images, labels) and returns the
                           (pos_activation_input, neg_activation_input) Tensors
                           *for this layer*, detached.
        threshold: Goodness threshold for the FF loss.
        epochs: Number of epochs to train this layer.
        device: The device to run training on.
        layer_index: The effective index of the hidden layer being trained (0-based).
        wandb_run: Optional W&B run object for logging.
        log_interval: Logging frequency.
    """
    layer_module.train()  # Set the specific module being trained to train mode
    layer_module.to(device)
    logger.info(
        f"Starting FF training for Layer {layer_index + 1}"
    )  # Log 1-based index

    total_steps_per_epoch = len(train_loader)
    # Calculate global step offset based on how many *full layer training runs* came before
    global_step_offset = layer_index * epochs * total_steps_per_epoch

    # Determine which functions to call based on the layer type
    if is_input_adapter_layer:
        # Need to apply activation and norm manually after the linear layer
        activation_fn = (
            model.first_layer_activation
        )  # Get from model instance (Closure needed?)
        norm_fn = model.first_layer_norm
        forward_module = layer_module  # This is the nn.Linear
    else:
        # FF_Layer handles activation and norm internally via forward_with_goodness
        activation_fn = None
        norm_fn = None
        forward_module = layer_module  # This is the FF_Layer instance

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"FF Layer {layer_index+1} Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # 1. Get detached positive/negative input activations for the current layer
            # This function should return the output of the *previous* layer (or the initial modified input)
            pos_activation_input, neg_activation_input = get_layer_input_fn(
                images, labels
            )

            # 2. Forward pass through current layer module to get activations before norm
            pos_lin = forward_module(pos_activation_input)
            neg_lin = forward_module(neg_activation_input)

            # Apply activation if needed (for input adapter layer)
            pos_act = activation_fn(pos_lin) if activation_fn else pos_lin
            neg_act = activation_fn(neg_lin) if activation_fn else neg_lin

            # 3. Calculate goodness (sum sq activations BEFORE norm)
            pos_goodness = torch.sum(pos_act.pow(2), dim=1)
            neg_goodness = torch.sum(neg_act.pow(2), dim=1)

            # 4. Calculate loss
            loss = ff_loss_fn(pos_goodness, neg_goodness, threshold)

            # 5. Backpropagate and optimize (only for the current layer_module)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            current_global_step = (
                global_step_offset + epoch * total_steps_per_epoch + batch_idx
            )

            # Logging
            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(
                train_loader
            ) - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.4f}")
                metrics = {
                    f"Layer_{layer_index+1}/Train_Loss_Batch": avg_loss_batch,
                    f"Layer_{layer_index+1}/Pos_Goodness_Mean": pos_goodness.mean().item(),
                    f"Layer_{layer_index+1}/Neg_Goodness_Mean": neg_goodness.mean().item(),
                }
                log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

        avg_epoch_loss = (
            epoch_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        )
        logger.info(
            f"FF Layer {layer_index+1} Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}"
        )
        # Log epoch loss against global step number representing end of this epoch for this layer
        log_metrics(
            {f"Layer_{layer_index+1}/Train_Loss_Epoch": avg_epoch_loss},
            step=global_step_offset + (epoch + 1) * total_steps_per_epoch,
            wandb_run=wandb_run,
        )

    logger.info(f"Finished FF training for Layer {layer_index + 1}")
    layer_module.eval()  # Set module back to eval mode


# --- Global reference to the model being trained (needed for closures) ---
# This is not ideal, consider passing model explicitly or using a class structure
model: Optional[FF_MLP] = None


def train_ff_model(
    model_instance: FF_MLP,  # Changed name to avoid conflict
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[
        Callable
    ] = None,  # input_adapter (flatten) is no longer needed here
):
    """
    Orchestrates the layer-wise training of an FF_MLP model using pixel embedding.
    """
    global model  # Use the global reference (needs careful handling)
    model = model_instance  # Assign the passed model instance to the global var

    model.to(device)
    num_hidden_layers = len(model.hidden_dims)  # Number of actual hidden layers
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

    # --- Define Input Generation Function ---
    # This closure uses the global `model` reference
    @torch.no_grad()
    def get_initial_input_data(images, labels):
        if model is None:
            raise RuntimeError("Global model not set in get_initial_input_data")
        # Generates the flattened, pixel-modified input for the *first* layer
        return generate_ff_pos_neg_pixel_data(images, labels, model.num_classes)

    # Function to get input for subsequent layers (needs output of previous layer)
    def create_layer_input_closure(prev_layer_idx: int):
        # prev_layer_idx = index of the layer *whose output is the input to the current layer*
        # If training layer 1 (FF_Layer[0]), prev_layer_idx is 0 (input_adapter).
        # If training layer k (FF_Layer[k-1]), prev_layer_idx is k-1.
        @torch.no_grad()
        def get_layer_input(images, labels):
            if model is None:
                raise RuntimeError("Global model not set in get_layer_input")
            # 1. Generate base pixel-embedded positive/negative data
            pos_flattened, neg_flattened = generate_ff_pos_neg_pixel_data(
                images, labels, model.num_classes
            )
            # 2. Pass through model up to the output of the previous layer
            pos_input_current = model.forward_upto(pos_flattened, prev_layer_idx)
            neg_input_current = model.forward_upto(neg_flattened, prev_layer_idx)
            return pos_input_current.detach(), neg_input_current.detach()

        return get_layer_input

    # --- Train Layer by Layer ---
    current_layer_input_fn = get_initial_input_data

    # 1. Train Input Adapter Layer (Effective Hidden Layer 0)
    logger.info(f"--- Training Input Adapter Layer (Layer 1/{num_hidden_layers}) ---")
    layer_module_0 = model.input_adapter_layer
    params_0 = list(layer_module_0.parameters())
    if params_0:
        optimizer_0_kwargs = {
            "lr": lr,
            "weight_decay": weight_decay,
            **optimizer_params_extra,
        }
        if optimizer_type.lower() == "adam":
            optimizer_0 = optim.Adam(params_0, **optimizer_0_kwargs)
        elif optimizer_type.lower() == "sgd":
            optimizer_0 = optim.SGD(params_0, **optimizer_0_kwargs)
        elif optimizer_type.lower() == "adamw":
            optimizer_0 = optim.AdamW(params_0, **optimizer_0_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        train_ff_layer(
            layer_module=layer_module_0,
            is_input_adapter_layer=True,
            optimizer=optimizer_0,
            train_loader=train_loader,
            get_layer_input_fn=current_layer_input_fn,  # Takes raw images/labels, returns modified flattened
            threshold=threshold,
            epochs=epochs_per_layer,
            device=device,
            layer_index=0,  # Log as Layer 1
            wandb_run=wandb_run,
            log_interval=log_interval,
        )
        for param in params_0:
            param.requires_grad = False
        layer_module_0.eval()

        # Update input function for the NEXT layer (layer 1 needs output of layer 0)
        current_layer_input_fn = create_layer_input_closure(prev_layer_idx=0)

        # --- Checkpointing after input adapter ---
        if checkpoint_dir:
            chkpt_filename = f"ff_layer_0_complete.pth"  # Index 0 for input adapter
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained": 0},
                is_best=False,
                filename=chkpt_filename,
                checkpoint_dir=checkpoint_dir,
            )
    else:
        logger.warning(
            "Input adapter layer has no trainable parameters. Skipping training, but updating input function."
        )
        # Still need to update the input function for the next layer
        current_layer_input_fn = create_layer_input_closure(prev_layer_idx=0)

    # 2. Train Subsequent Hidden FF_Layers
    for i in range(len(model.layers)):  # model.layers contains hidden 1 -> 2, etc.
        effective_layer_index = (
            i + 1
        )  # This is layer 1, 2, ... in terms of hidden layers
        layer_log_index = effective_layer_index + 1  # Log as Layer 2, 3, ...
        logger.info(
            f"--- Training Hidden FF_Layer {layer_log_index}/{num_hidden_layers} ---"
        )

        ff_layer_module = model.layers[i]
        params_i = list(ff_layer_module.parameters())
        if not params_i:
            logger.warning(
                f"Hidden FF_Layer {layer_log_index} has no trainable parameters. Skipping training, but updating input fn."
            )
        else:
            optimizer_i_kwargs = {
                "lr": lr,
                "weight_decay": weight_decay,
                **optimizer_params_extra,
            }
            if optimizer_type.lower() == "adam":
                optimizer_i = optim.Adam(params_i, **optimizer_i_kwargs)
            elif optimizer_type.lower() == "sgd":
                optimizer_i = optim.SGD(params_i, **optimizer_i_kwargs)
            elif optimizer_type.lower() == "adamw":
                optimizer_i = optim.AdamW(params_i, **optimizer_i_kwargs)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_type}")

            train_ff_layer(
                layer_module=ff_layer_module,
                is_input_adapter_layer=False,
                optimizer=optimizer_i,
                train_loader=train_loader,
                get_layer_input_fn=current_layer_input_fn,  # Provides input activations for this layer
                threshold=threshold,
                epochs=epochs_per_layer,
                device=device,
                layer_index=effective_layer_index,  # Pass effective layer index
                wandb_run=wandb_run,
                log_interval=log_interval,
            )
            for param in params_i:
                param.requires_grad = False
            ff_layer_module.eval()

            # --- Checkpointing after each hidden layer ---
            if checkpoint_dir:
                chkpt_filename = f"ff_layer_{effective_layer_index}_complete.pth"
                save_checkpoint(
                    state={
                        "state_dict": model.state_dict(),
                        "layer_trained": effective_layer_index,
                    },
                    is_best=False,
                    filename=chkpt_filename,
                    checkpoint_dir=checkpoint_dir,
                )

        # --- Update input function for the NEXT layer ---
        # Need output of layer `effective_layer_index`
        if (
            effective_layer_index < num_hidden_layers - 1
        ):  # Only if there is a next layer
            current_layer_input_fn = create_layer_input_closure(
                prev_layer_idx=effective_layer_index
            )

    logger.info("Finished all layer-wise FF training.")
    model = None  # Clear global reference


def evaluate_ff_model(
    model_instance: FF_MLP,  # Changed name
    data_loader: DataLoader,  # Validation or Test loader
    device: torch.device,
    input_adapter: Optional[Callable] = None,  # Not used with pixel embedding
) -> Dict[str, float]:
    """Evaluates the trained FF_MLP model using multi-pass inference with pixel embedding."""
    # Use the passed model instance directly
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
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(
                device
            )  # images are [B, C, H, W]
            batch_size = images.shape[0]

            batch_total_goodness = torch.zeros((batch_size, num_classes), device=device)

            for label_candidate in range(num_classes):
                candidate_labels = torch.full(
                    (batch_size,), label_candidate, dtype=torch.long, device=device
                )

                # Create the flattened, pixel-embedded input for this candidate label
                try:
                    ff_input_candidate = create_ff_pixel_label_input(
                        images, candidate_labels, num_classes
                    )  # Returns [B, C*H*W]
                except Exception as e:
                    logger.error(
                        f"Error creating pixel input for candidate {label_candidate}: {e}",
                        exc_info=True,
                    )
                    batch_total_goodness[:, label_candidate] = (
                        -torch.inf
                    )  # Mark as invalid
                    continue

                # Pass this modified input through the model to get goodness scores
                try:
                    layer_goodness_list = model.forward_goodness_per_layer(
                        ff_input_candidate  # Pass modified flattened input
                    )
                except Exception as e:
                    logger.error(
                        f"Error in FF goodness forward pass for candidate {label_candidate}: {e}",
                        exc_info=True,
                    )
                    batch_total_goodness[:, label_candidate] = -torch.inf
                    continue

                # Sum goodness across layers
                if not layer_goodness_list:
                    logger.warning(
                        f"Received empty goodness list for candidate {label_candidate}"
                    )
                    total_goodness_candidate = torch.zeros((batch_size,), device=device)
                else:
                    try:
                        # Sum goodness across all hidden layers (list index corresponds to layer index)
                        total_goodness_candidate = torch.stack(
                            layer_goodness_list, dim=0
                        ).sum(dim=0)
                        if total_goodness_candidate.shape != (batch_size,):
                            raise ValueError(
                                f"Unexpected shape after summing goodness: {total_goodness_candidate.shape}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error stacking/summing goodness: {e}", exc_info=True
                        )
                        batch_total_goodness[:, label_candidate] = -torch.inf
                        continue

                batch_total_goodness[:, label_candidate] = total_goodness_candidate

            # Predict based on highest total goodness
            try:
                predicted_labels = torch.argmax(batch_total_goodness, dim=1)
            except Exception as e:
                logger.error(f"Error during argmax prediction: {e}", exc_info=True)
                # Handle cases where all goodness might be -inf? Predict 0?
                predicted_labels = torch.zeros_like(labels)  # Fallback prediction

            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"FF Evaluation Accuracy (Pixel Embedding): {accuracy:.2f}%")

    results = {"eval_accuracy": accuracy}  # Use consistent naming
    return results
