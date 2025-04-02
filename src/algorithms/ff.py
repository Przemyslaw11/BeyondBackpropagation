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
import os
import time

from src.architectures.ff_mlp import FF_MLP
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint, format_time, create_directory_if_not_exists # Added create_dir

logger = logging.getLogger(__name__)

# --- ff_loss_fn (no changes needed) ---
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
    loss_pos = F.softplus(-(pos_goodness - threshold))
    loss_neg = F.softplus(neg_goodness - threshold)
    loss = torch.mean(loss_pos + loss_neg)
    return loss

# --- create_ff_pixel_label_input (no changes needed) ---
def create_ff_pixel_label_input(
    original_images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    replace_value_on: float = 1.0,
    replace_value_off: float = 0.0,
) -> torch.Tensor:
    """
    Creates FF input by replacing the initial pixels of the image (flattened view)
    with a representation of the label.
    """
    batch_size, channels, height, width = original_images.shape
    device = original_images.device

    if num_classes > height * width:
        raise ValueError(
            f"num_classes ({num_classes}) is larger than the number of pixels per channel ({height*width}). Cannot embed label."
        )

    one_hot_labels = F.one_hot(labels, num_classes=num_classes).to(
        device=device, dtype=torch.float
    )
    label_patch = torch.where(one_hot_labels == 1, replace_value_on, replace_value_off)
    modified_images = original_images.clone()
    pixels_to_replace = label_patch.view(batch_size, num_classes)
    row_indices = torch.arange(num_classes, device=device) // width
    col_indices = torch.arange(num_classes, device=device) % width

    if torch.any(row_indices >= height) or torch.any(col_indices >= width):
        raise RuntimeError("Calculated pixel indices are out of image bounds.")

    try:
        modified_images[:, :, row_indices, col_indices] = pixels_to_replace.unsqueeze(1)
    except Exception as e:
        logger.error(f"Error during pixel replacement: {e}", exc_info=True)
        raise

    flattened_modified_images = modified_images.view(batch_size, -1)
    return flattened_modified_images

# --- generate_ff_pos_neg_pixel_data (no changes needed) ---
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
    pos_flattened_images = create_ff_pixel_label_input(
        base_images, base_labels, num_classes
    )
    rand_offset = torch.randint(
        1, num_classes, (batch_size,), device=device, dtype=torch.long
    )
    neg_labels = (base_labels + rand_offset) % num_classes
    collision = neg_labels == base_labels
    retries = 0
    max_retries = 5
    while torch.any(collision) and retries < max_retries:
        num_collisions = collision.sum().item()
        new_rand_offset = torch.randint(
            1, num_classes, (num_collisions,), device=device, dtype=torch.long
        )
        neg_labels[collision] = (base_labels[collision] + new_rand_offset) % num_classes
        collision = neg_labels == base_labels
        retries += 1
    if retries == max_retries and torch.any(collision):
        logger.warning(
            f"Could not guarantee distinct negative labels after {max_retries} retries for {collision.sum().item()} samples."
        )
        neg_labels[collision] = (neg_labels[collision] + 1) % num_classes
    neg_flattened_images = create_ff_pixel_label_input(
        base_images, neg_labels, num_classes
    )
    return pos_flattened_images, neg_flattened_images


# --- train_ff_layer (MODIFIED) ---
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
    step_ref: List[int] = [-1], # MODIFIED: Use step_ref list
) -> None:
    """
    Trains a single effective layer (input adapter or subsequent FF_Layer) of the FF_MLP.
    Updates weights locally based on the FF loss calculated from positive/negative goodness.
    """
    layer_module.train()
    layer_module.to(device)
    logger.info(
        f"Starting FF training for Layer {layer_index + 1}"
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"FF Layer {layer_index+1} Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1 # MODIFIED: Increment global step reference
            current_global_step = step_ref[0]

            images, labels = images.to(device), labels.to(device)

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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.4f}")
                # MODIFIED: Add global_step to metrics dict
                metrics_to_log = {
                    "global_step": current_global_step,
                    f"FF/Layer_{layer_index+1}/Train_Loss_Batch": avg_loss_batch,
                    f"FF/Layer_{layer_index+1}/Pos_Goodness_Mean": pos_goodness.mean().item(),
                    f"FF/Layer_{layer_index+1}/Neg_Goodness_Mean": neg_goodness.mean().item(),
                }
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True) # Pass full dict

        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)):
            logger.error(
                f"Terminating training for Layer {layer_index+1} due to invalid loss in epoch {epoch+1}."
            )
            break

        # --- REMOVED epoch summary logging from here, will be logged in train_ff_model ---
        # avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        # logger.info(
        #     f"FF Layer {layer_index+1} Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}"
        # )
        # ... (removed log_metrics call for epoch loss) ...

    logger.info(f"Finished FF training for Layer {layer_index + 1}")
    layer_module.eval()
    # Return average loss for potential logging in the orchestrator
    return epoch_loss / len(train_loader) if len(train_loader) > 0 else 0.0


# --- train_ff_model (MODIFIED) ---
def train_ff_model(
    model_instance: FF_MLP,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None, # Added to match signature, though FF uses pixel embedding
    step_ref: List[int] = [-1], # MODIFIED: Accept step_ref
):
    """
    Orchestrates the layer-wise training of an FF_MLP model using pixel label embedding.
    MODIFIED: Logs epoch summary metrics.
    """
    model = model_instance
    model.to(device)
    num_hidden_layers = len(model.hidden_dims)
    logger.info(
        f"Starting layer-wise FF training for {num_hidden_layers} hidden layers (using pixel label embedding)."
    )
    if input_adapter is not None:
        logger.warning("FF Training: 'input_adapter' provided but FF typically uses internal pixel embedding. Adapter will not be used.")

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

    # --- Define Input Generation Functions ---
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

        # Train the layer
        final_avg_loss_layer_0 = train_ff_layer(
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
            step_ref=step_ref, # Pass step_ref
        )
        # MODIFIED: Log layer summary after training completes
        current_global_step = step_ref[0]
        layer_summary_metrics = {
            "global_step": current_global_step,
            f"FF/Layer_1/Train_Loss_LayerAvg": final_avg_loss_layer_0,
        }
        log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=True)
        logger.debug(f"Logged FF Layer 1 summary at global_step {current_global_step}")

        for param in params_0: param.requires_grad = False
        layer_module_0.eval()
        if checkpoint_dir:
            create_directory_if_not_exists(checkpoint_dir) # Ensure dir exists
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained": 0},
                is_best=False, filename=f"ff_layer_0_complete.pth", checkpoint_dir=checkpoint_dir,
            )
    else: logger.warning("Input adapter layer has no parameters.")

    current_layer_input_fn = create_layer_input_closure(model, prev_layer_idx=0)

    # 2. Train Subsequent Hidden FF_Layers
    for i in range(len(model.layers)):
        effective_layer_index = i + 1
        layer_log_index = effective_layer_index + 1 # For logging (1-based)
        logger.info(f"--- Training Hidden FF_Layer {layer_log_index}/{num_hidden_layers} ---")

        ff_layer_module = model.layers[i]
        params_i = list(ff_layer_module.parameters())
        if not params_i:
            logger.warning(f"Hidden FF_Layer {layer_log_index} has no parameters.")
        else:
            optimizer_i_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_params_extra}
            optimizer_i = getattr(optim, optimizer_type)(params_i, **optimizer_i_kwargs)

            final_avg_loss_layer_i = train_ff_layer(
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
                step_ref=step_ref, # Pass step_ref
            )
            # MODIFIED: Log layer summary after training completes
            current_global_step = step_ref[0]
            layer_summary_metrics = {
                "global_step": current_global_step,
                f"FF/Layer_{layer_log_index}/Train_Loss_LayerAvg": final_avg_loss_layer_i,
            }
            log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=True)
            logger.debug(f"Logged FF Layer {layer_log_index} summary at global_step {current_global_step}")

            for param in params_i: param.requires_grad = False
            ff_layer_module.eval()
            if checkpoint_dir:
                create_directory_if_not_exists(checkpoint_dir) # Ensure dir exists
                save_checkpoint(
                    state={"state_dict": model.state_dict(), "layer_trained": effective_layer_index},
                    is_best=False, filename=f"ff_layer_{effective_layer_index}_complete.pth", checkpoint_dir=checkpoint_dir,
                )

        if effective_layer_index < num_hidden_layers - 1:
            current_layer_input_fn = create_layer_input_closure(model, prev_layer_idx=effective_layer_index)

    logger.info("Finished all layer-wise FF training.")


# --- evaluate_ff_model (no changes needed) ---
def evaluate_ff_model(
    model_instance: FF_MLP,
    data_loader: DataLoader,
    device: torch.device,
    **kwargs, # Accept potential criterion/adapter but ignore them
) -> Dict[str, float]:
    """
    Evaluates the trained FF_MLP model using multi-pass inference.
    Ignores criterion and input_adapter if passed.
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
                        # Ensure stacking happens correctly even with single layer
                        if len(layer_goodness_list) == 1:
                           total_goodness_candidate = layer_goodness_list[0]
                        else:
                           total_goodness_candidate = torch.stack(layer_goodness_list, dim=0).sum(dim=0)

                        if total_goodness_candidate.shape != (batch_size,):
                            raise ValueError(f"Unexpected shape after summing goodness: {total_goodness_candidate.shape}")
                    except Exception as e:
                        logger.error(f"Error stacking/summing goodness for candidate {label_candidate}: {e}", exc_info=True)
                        batch_total_goodness[:, label_candidate] = -torch.inf
                        continue
                batch_total_goodness[:, label_candidate] = total_goodness_candidate

            try:
                all_inf_mask = torch.all(torch.isinf(batch_total_goodness) & (batch_total_goodness < 0), dim=1)
                if torch.any(all_inf_mask):
                    num_all_inf = all_inf_mask.sum().item()
                    logger.warning(f"Batch {batch_idx+1}: {num_all_inf} samples had -inf goodness for all candidates. Predicting 0 for these.")
                    # Initialize predictions with 0
                    predicted_labels = torch.zeros_like(labels)
                    # Only compute argmax for non-all-inf samples
                    if not torch.all(all_inf_mask):
                        valid_indices = ~all_inf_mask
                        predicted_labels[valid_indices] = torch.argmax(batch_total_goodness[valid_indices], dim=1)
                else:
                    predicted_labels = torch.argmax(batch_total_goodness, dim=1)
            except Exception as e:
                logger.error(f"Error during argmax prediction for batch {batch_idx+1}: {e}", exc_info=True)
                # Default prediction to 0 in case of error during argmax
                predicted_labels = torch.zeros_like(labels)


            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"FF Evaluation Accuracy (Pixel Embedding): {accuracy:.2f}%")
    # FF doesn't have a standard loss concept during this eval type
    results = {"eval_accuracy": accuracy, "eval_loss": float("nan")}
    return results