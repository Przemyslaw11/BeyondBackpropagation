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

from src.architectures.ff_mlp import FF_MLP, FF_Layer

from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint, format_time, create_directory_if_not_exists
from src.utils.monitoring import get_gpu_memory_usage # Import memory usage function

logger = logging.getLogger(__name__)

# --- ff_loss_fn ---
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

# --- create_ff_pixel_label_input ---
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

    if num_classes > height * width:
        raise ValueError(
            f"num_classes ({num_classes}) is larger than the number of pixels per channel ({height*width}). Cannot embed label."
        )

    one_hot_labels = F.one_hot(labels, num_classes=num_classes).to(
        device=device, dtype=torch.float
    )
    label_patch = torch.where(one_hot_labels == 1, replace_value_on, replace_value_off)
    modified_images = original_images.clone()
    first_channel_flat = modified_images[:, 0, :, :].view(batch_size, -1)
    first_channel_flat[:, :num_classes] = label_patch
    modified_images[:, 0, :, :] = first_channel_flat.view(batch_size, height, width)
    flattened_modified_images = modified_images.view(batch_size, -1)
    return flattened_modified_images

# --- generate_ff_pos_neg_pixel_data ---
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
            f"Could not guarantee distinct negative labels after {max_retries} retries for {collision.sum().item()} samples. Forcing difference."
        )
        neg_labels[collision] = (neg_labels[collision] + 1) % num_classes
    neg_flattened_images = create_ff_pixel_label_input(
        base_images, neg_labels, num_classes
    )
    return pos_flattened_images, neg_flattened_images


# --- train_ff_layer ---
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
    layer_index: int,
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> Tuple[float, float, float, float]: # Added return for grad norms
    """
    Trains a single effective layer (input adapter or subsequent FF_Layer) of the FF_MLP.
    Updates weights locally based on the FF loss calculated from positive/negative goodness.
    Returns the average loss, peak memory, avg weight grad norm, avg bias grad norm of the last epoch.
    """
    layer_module.train()
    layer_module.to(device)
    log_prefix = f"FF/Layer {layer_index + 1} ({'Input Adapter' if is_input_adapter_layer else 'Hidden'})"
    logger.info(f"Starting FF training for {log_prefix}")
    peak_mem_layer_train = 0.0
    last_epoch_avg_loss = float('nan')
    last_epoch_grad_norm_w_mean = float('nan')
    last_epoch_grad_norm_b_mean = float('nan')

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_pos_goodness_acc = 0.0
        epoch_neg_goodness_acc = 0.0
        epoch_loss_pos_acc = 0.0
        epoch_loss_neg_acc = 0.0
        epoch_grad_norm_w_acc = 0.0
        epoch_grad_norm_b_acc = 0.0
        num_grad_batches = 0
        peak_mem_layer_epoch = 0.0
        has_bias_ever = False # Track if bias exists for logging

        pbar = tqdm(
            train_loader,
            desc=f"{log_prefix} Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1
            current_global_step = step_ref[0]

            images, labels = images.to(device), labels.to(device)

            try:
                pos_activation_input, neg_activation_input = get_layer_input_fn(
                    images, labels
                )
            except Exception as e:
                logger.error(
                    f"Error getting layer input at {log_prefix}, Epoch {epoch+1}, Batch {batch_idx}: {e}",
                    exc_info=True,
                )
                continue

            pos_activation_input = pos_activation_input.to(device)
            neg_activation_input = neg_activation_input.to(device)

            pos_goodness: torch.Tensor
            neg_goodness: torch.Tensor
            loss_pos: torch.Tensor
            loss_neg: torch.Tensor

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

                loss_pos = F.softplus(-(pos_goodness - threshold))
                loss_neg = F.softplus(neg_goodness - threshold)
                loss = torch.mean(loss_pos + loss_neg)

            except Exception as e:
                logger.error(
                    f"Error during forward/goodness/loss calculation at {log_prefix}, Epoch {epoch+1}, Batch {batch_idx}: {e}",
                    exc_info=True,
                )
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"NaN or Inf loss detected at {log_prefix}, Epoch {epoch+1}, Batch {batch_idx}. Stopping layer training."
                )
                epoch_loss = float('nan'); epoch_samples = 1
                epoch_pos_goodness_acc=float('nan'); epoch_neg_goodness_acc=float('nan')
                epoch_loss_pos_acc=float('nan'); epoch_loss_neg_acc=float('nan')
                epoch_grad_norm_w_acc=float('nan'); epoch_grad_norm_b_acc=float('nan')
                num_grad_batches=1
                break

            optimizer.zero_grad()
            loss.backward()

            grad_norm_w = 0.0
            grad_norm_b = 0.0
            has_bias_batch = False
            if isinstance(layer_module, (nn.Linear, FF_Layer)):
                 lin_layer = layer_module if isinstance(layer_module, nn.Linear) else layer_module.linear
                 if lin_layer.weight.grad is not None:
                     grad_norm_w = torch.linalg.norm(lin_layer.weight.grad).item()
                 if lin_layer.bias is not None and lin_layer.bias.grad is not None:
                     has_bias_batch = True
                     has_bias_ever = True # Keep track if bias exists at all for this layer
                     grad_norm_b = torch.linalg.norm(lin_layer.bias.grad).item()
                 epoch_grad_norm_w_acc += grad_norm_w
                 epoch_grad_norm_b_acc += grad_norm_b
                 num_grad_batches += 1

            optimizer.step()

            batch_size = images.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            epoch_pos_goodness_acc += pos_goodness.mean().item() * batch_size
            epoch_neg_goodness_acc += neg_goodness.mean().item() * batch_size
            epoch_loss_pos_acc += loss_pos.mean().item() * batch_size
            epoch_loss_neg_acc += loss_neg.mean().item() * batch_size

            current_mem_used = float('nan')
            if nvml_active and gpu_handle and ((batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1):
                 mem_info = get_gpu_memory_usage(gpu_handle)
                 if mem_info:
                     current_mem_used = mem_info[0]
                     peak_mem_layer_epoch = max(peak_mem_layer_epoch, current_mem_used)

            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.4f}")
                metrics_to_log = {
                    "global_step": current_global_step,
                    f"{log_prefix}/Train_Loss_Batch": avg_loss_batch,
                    f"{log_prefix}/Pos_Goodness_Mean": pos_goodness.mean().item(),
                    f"{log_prefix}/Neg_Goodness_Mean": neg_goodness.mean().item(),
                    f"{log_prefix}/Loss_Pos_Mean": loss_pos.mean().item(),
                    f"{log_prefix}/Loss_Neg_Mean": loss_neg.mean().item(),
                    f"{log_prefix}/Grad_Norm_W_Mean": grad_norm_w,
                    f"{log_prefix}/Grad_Norm_B_Mean": grad_norm_b if has_bias_batch else float('nan'),
                }
                if not torch.isnan(torch.tensor(current_mem_used)):
                    metrics_to_log[f"{log_prefix}/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)

        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)):
            logger.error(
                f"Terminating training for {log_prefix} due to invalid loss in epoch {epoch+1}."
            )
            last_epoch_avg_loss = float('nan')
            last_epoch_grad_norm_w_mean = float('nan')
            last_epoch_grad_norm_b_mean = float('nan')
            break

        last_epoch_avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float('nan')
        avg_pos_goodness = epoch_pos_goodness_acc / epoch_samples if epoch_samples > 0 else float('nan')
        avg_neg_goodness = epoch_neg_goodness_acc / epoch_samples if epoch_samples > 0 else float('nan')
        avg_loss_pos = epoch_loss_pos_acc / epoch_samples if epoch_samples > 0 else float('nan')
        avg_loss_neg = epoch_loss_neg_acc / epoch_samples if epoch_samples > 0 else float('nan')
        last_epoch_grad_norm_w_mean = epoch_grad_norm_w_acc / num_grad_batches if num_grad_batches > 0 else float('nan')
        last_epoch_grad_norm_b_mean = epoch_grad_norm_b_acc / num_grad_batches if num_grad_batches > 0 and has_bias_ever else float('nan')
        peak_mem_layer_train = max(peak_mem_layer_train, peak_mem_layer_epoch)

        logger.info(
            f"{log_prefix} Epoch {epoch+1}/{epochs} - Avg Loss: {last_epoch_avg_loss:.4f}, Peak Mem Epoch: {peak_mem_layer_epoch:.1f} MiB"
        )
        epoch_summary_metrics = {
            "global_step": current_global_step,
            f"{log_prefix}/Train_Loss_EpochAvg": last_epoch_avg_loss,
            f"{log_prefix}/Pos_Goodness_EpochAvg": avg_pos_goodness,
            f"{log_prefix}/Neg_Goodness_EpochAvg": avg_neg_goodness,
            f"{log_prefix}/Loss_Pos_EpochAvg": avg_loss_pos,
            f"{log_prefix}/Loss_Neg_EpochAvg": avg_loss_neg,
            f"{log_prefix}/Grad_Norm_W_EpochAvg": last_epoch_grad_norm_w_mean,
            f"{log_prefix}/Grad_Norm_B_EpochAvg": last_epoch_grad_norm_b_mean,
            f"{log_prefix}/Peak_GPU_Mem_Epoch_MiB": peak_mem_layer_epoch,
        }
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)

    if nvml_active and gpu_handle:
        mem_info = get_gpu_memory_usage(gpu_handle)
        if mem_info:
            peak_mem_layer_train = max(peak_mem_layer_train, mem_info[0])

    logger.info(f"Finished FF training for {log_prefix}. Overall Peak Mem for Layer: {peak_mem_layer_train:.1f} MiB")
    layer_module.eval()
    return last_epoch_avg_loss, peak_mem_layer_train, last_epoch_grad_norm_w_mean, last_epoch_grad_norm_b_mean


# --- train_ff_model ---
def train_ff_model(
    model: FF_MLP,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float:
    """
    Orchestrates the layer-wise training of an FF_MLP model using pixel label embedding.
    Logs layer summary metrics including peak memory.
    Returns the overall peak GPU memory observed across all layer training phases.
    """
    model.to(device)
    num_hidden_layers = len(model.hidden_dims)
    logger.info(
        f"Starting layer-wise FF training for {num_hidden_layers} hidden layers (using pixel label embedding)."
    )
    if input_adapter is not None:
        logger.warning("FF Training: 'input_adapter' provided but FF uses internal pixel embedding. Adapter will not be used by FF training logic.")

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
            model_ref.to(device)
            model_ref.eval()
            pos_input_current = model_ref.forward_upto(pos_flattened, prev_layer_idx)
            neg_input_current = model_ref.forward_upto(neg_flattened, prev_layer_idx)
            return pos_input_current.detach().to(device), neg_input_current.detach().to(device)
        return get_layer_input

    current_layer_input_fn = get_initial_input_data
    peak_mem_train = 0.0

    # 1. Train Input Adapter Layer (Effective Hidden Layer 0)
    layer_log_idx_0 = 1
    log_prefix_0 = f"FF/Layer {layer_log_idx_0} (Input Adapter)"
    logger.info(f"--- Training {log_prefix_0} ---")
    layer_module_0 = model.input_adapter_layer
    params_0 = list(layer_module_0.parameters())
    layer_0_peak_mem = 0.0
    final_avg_loss_layer_0 = float('nan')
    final_grad_w_0 = float('nan')
    final_grad_b_0 = float('nan')

    if params_0:
        optimizer_0_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_params_extra}
        optimizer_0 = getattr(optim, optimizer_type)(params_0, **optimizer_0_kwargs)

        final_avg_loss_layer_0, layer_0_peak_mem, final_grad_w_0, final_grad_b_0 = train_ff_layer(
            model=model, layer_module=layer_module_0, is_input_adapter_layer=True,
            optimizer=optimizer_0, train_loader=train_loader,
            get_layer_input_fn=current_layer_input_fn, threshold=threshold,
            epochs=epochs_per_layer, device=device, layer_index=0,
            wandb_run=wandb_run, log_interval=log_interval, step_ref=step_ref,
            gpu_handle=gpu_handle, nvml_active=nvml_active
        )
        peak_mem_train = max(peak_mem_train, layer_0_peak_mem)

        for param in params_0: param.requires_grad = False
        layer_module_0.eval()
    else:
         logger.warning(f"{log_prefix_0} has no parameters. Skipping training.")

    current_global_step = step_ref[0]
    layer_summary_metrics = {
        "global_step": current_global_step,
        f"{log_prefix_0}/Train_Loss_LayerAvg": final_avg_loss_layer_0,
        f"{log_prefix_0}/Peak_GPU_Mem_Layer_MiB": layer_0_peak_mem,
        f"{log_prefix_0}/Final_GradNormW_LayerAvg": final_grad_w_0,
        f"{log_prefix_0}/Final_GradNormB_LayerAvg": final_grad_b_0,
    }
    log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=True)
    logger.debug(f"Logged {log_prefix_0} summary at global_step {current_global_step}")

    if checkpoint_dir and params_0:
        create_directory_if_not_exists(checkpoint_dir)
        save_checkpoint(
            state={"state_dict": model.state_dict(), "layer_trained": 0},
            is_best=False, filename=f"ff_layer_0_complete.pth", checkpoint_dir=checkpoint_dir,
        )

    current_layer_input_fn = create_layer_input_closure(model, prev_layer_idx=0)

    # 2. Train Subsequent Hidden FF_Layers
    for i in range(len(model.layers)):
        effective_layer_index = i + 1
        layer_log_idx = effective_layer_index + 1
        log_prefix_i = f"FF/Layer {layer_log_idx} (Hidden)"
        logger.info(f"--- Training {log_prefix_i} ---")

        ff_layer_module = model.layers[i]
        params_i = list(ff_layer_module.parameters())
        layer_i_peak_mem = 0.0
        final_avg_loss_layer_i = float('nan')
        final_grad_w_i = float('nan')
        final_grad_b_i = float('nan')

        if not params_i:
            logger.warning(f"{log_prefix_i} has no parameters.")
        else:
            optimizer_i_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_params_extra}
            optimizer_i = getattr(optim, optimizer_type)(params_i, **optimizer_i_kwargs)

            final_avg_loss_layer_i, layer_i_peak_mem, final_grad_w_i, final_grad_b_i = train_ff_layer(
                model=model, layer_module=ff_layer_module, is_input_adapter_layer=False,
                optimizer=optimizer_i, train_loader=train_loader,
                get_layer_input_fn=current_layer_input_fn, threshold=threshold,
                epochs=epochs_per_layer, device=device, layer_index=effective_layer_index,
                wandb_run=wandb_run, log_interval=log_interval, step_ref=step_ref,
                gpu_handle=gpu_handle, nvml_active=nvml_active
            )
            peak_mem_train = max(peak_mem_train, layer_i_peak_mem)

            for param in params_i: param.requires_grad = False
            ff_layer_module.eval()

        current_global_step = step_ref[0]
        layer_summary_metrics = {
            "global_step": current_global_step,
            f"{log_prefix_i}/Train_Loss_LayerAvg": final_avg_loss_layer_i,
            f"{log_prefix_i}/Peak_GPU_Mem_Layer_MiB": layer_i_peak_mem,
            f"{log_prefix_i}/Final_GradNormW_LayerAvg": final_grad_w_i,
            f"{log_prefix_i}/Final_GradNormB_LayerAvg": final_grad_b_i,
        }
        log_metrics(layer_summary_metrics, wandb_run=wandb_run, commit=True)
        logger.debug(f"Logged {log_prefix_i} summary at global_step {current_global_step}")

        if checkpoint_dir and params_i:
            create_directory_if_not_exists(checkpoint_dir)
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained": effective_layer_index},
                is_best=False, filename=f"ff_layer_{effective_layer_index}_complete.pth", checkpoint_dir=checkpoint_dir,
            )

        if effective_layer_index < num_hidden_layers - 1:
             # Prepare input for the *next* layer's training
             current_layer_input_fn = create_layer_input_closure(model, prev_layer_idx=effective_layer_index)


    logger.info("Finished all layer-wise FF training.")
    return peak_mem_train


# --- evaluate_ff_model ---
def evaluate_ff_model(
    model: FF_MLP,
    data_loader: DataLoader,
    device: torch.device,
    **kwargs, # Accept extra args for signature consistency
) -> Dict[str, float]:
    """
    Evaluates the trained FF_MLP model using multi-pass inference (Hinton Sec 3.3).
    MODIFIED: Sums goodness excluding the first layer.
    """
    model.eval()
    model.to(device)
    num_classes = model.num_classes
    logger.info(
        f"Evaluating FF model using multi-pass inference ({num_classes} passes per image, pixel embedding, excluding first layer goodness)."
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
                    # Should not happen if model has layers, but handle defensively
                    logger.warning(f"Candidate {label_candidate}: No goodness scores returned.")
                    total_goodness_candidate = torch.zeros((batch_size,), device=device)
                else:
                    try:
                        # <<< MODIFICATION START >>>
                        # Sum goodness from the second layer onwards (index 1)
                        if len(layer_goodness_list) > 1:
                             # Stack goodness from layers 1 to L (inclusive)
                             total_goodness_candidate = torch.stack(layer_goodness_list[1:], dim=0).sum(dim=0)
                             logger.debug(f"Candidate {label_candidate}: Summed goodness from layers 1 to {len(layer_goodness_list)-1}")
                        else:
                             # If only one layer (Input Adapter only), goodness for eval is 0
                             total_goodness_candidate = torch.zeros((batch_size,), device=device)
                             logger.warning(f"Candidate {label_candidate}: Only first layer's goodness available. Using 0 goodness for eval as per Hinton Sec 3.3 method.")
                        # <<< MODIFICATION END >>>

                        if total_goodness_candidate.shape != (batch_size,):
                            raise ValueError(f"Unexpected shape after summing goodness: {total_goodness_candidate.shape}")
                    except Exception as e:
                        logger.error(f"Error stacking/summing goodness for candidate {label_candidate}: {e}", exc_info=True)
                        batch_total_goodness[:, label_candidate] = -torch.inf
                        continue

                batch_total_goodness[:, label_candidate] = total_goodness_candidate

            try:
                # Handle cases where all candidates might have resulted in -inf
                all_inf_mask = torch.all(torch.isinf(batch_total_goodness) & (batch_total_goodness < 0), dim=1)
                if torch.any(all_inf_mask):
                    num_all_inf = all_inf_mask.sum().item()
                    logger.warning(f"Batch {batch_idx+1}: {num_all_inf} samples had -inf goodness for all candidates. Predicting 0 for these.")
                    # Initialize predictions to 0
                    predicted_labels = torch.zeros_like(labels)
                    # Find indices that are *not* all -inf
                    valid_indices = ~all_inf_mask
                    # Predict only for valid indices
                    if torch.any(valid_indices):
                        predicted_labels[valid_indices] = torch.argmax(batch_total_goodness[valid_indices], dim=1)
                else:
                    # Standard prediction if no all-inf rows
                    predicted_labels = torch.argmax(batch_total_goodness, dim=1)

            except Exception as e:
                logger.error(f"Error during argmax prediction for batch {batch_idx+1}: {e}", exc_info=True)
                # Fallback prediction if argmax fails
                predicted_labels = torch.zeros_like(labels)

            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"FF Evaluation Accuracy (Pixel Embedding, Multi-Pass, Excl. L1 Goodness): {accuracy:.2f}%")
    # FF evaluation doesn't typically calculate a conventional loss
    results = {"eval_accuracy": accuracy, "eval_loss": float("nan")}
    return results