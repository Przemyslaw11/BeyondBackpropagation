# File: src/algorithms/ff.py (MODIFIED - Replaces previous content)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm
import pynvml # Import for type hint
from typing import Dict, Any, Optional, Tuple, List, Callable
import os
import time

# Import the MODIFIED FF_MLP
from src.architectures.ff_mlp import FF_MLP # Ensure this path is correct
from src.utils.logging_utils import log_metrics
from src.utils.helpers import format_time, save_checkpoint, create_directory_if_not_exists
from src.utils.monitoring import get_gpu_memory_usage

logger = logging.getLogger(__name__)

# --- Helper for Pixel Label Embedding (Copied & Verified from reference logic) ---
# <<< CORRECTION: Added helper function from reference code >>>
def generate_ff_hinton_inputs(
    base_images: torch.Tensor,
    base_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    replace_value_on: float = 1.0, # Reference uses 1.0
    replace_value_off: float = 0.0, # Reference uses 0.0
    neutral_value: float = 0.1, # Reference uses 0.1 for neutral
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates positive, negative, and neutral flattened image tensors for FF training
    using Hinton's pixel replacement method. Ensures negative label is different.
    Matches the logic in the reference `ff_mnist.py`.

    Returns:
        Tuple (pos_flattened, neg_flattened, neutral_flattened)
    """
    batch_size = base_images.shape[0]
    # Reference embeds into first `num_classes` pixels of the flattened image.
    # We assume the input `base_images` are (B, C, H, W) initially.

    # --- Positive Samples ---
    one_hot_pos = F.one_hot(base_labels, num_classes=num_classes).to(device=device, dtype=torch.float)
    # Create the patch (1.0 where label is 1, 0.0 otherwise)
    label_patch_pos = torch.where(one_hot_pos == 1, replace_value_on, replace_value_off)
    pos_images = base_images.clone()
    # Embed into the first C=0 channel, first H=0 row, first num_classes pixels
    # Reference code implies embedding into first N pixels of flattened image.
    pos_images_flat_view = pos_images.view(batch_size, -1)
    pos_images_flat_view[:, :num_classes] = label_patch_pos
    pos_flattened = pos_images_flat_view # Already flattened

    # --- Negative Samples ---
    # Ensure negative label is different from positive label
    rand_offset = torch.randint(1, num_classes, (batch_size,), device=device, dtype=torch.long)
    neg_labels = (base_labels + rand_offset) % num_classes
    # Simple check for collisions (highly unlikely for num_classes=10, but good practice)
    collision = neg_labels == base_labels
    retries = 0; max_retries = 5 # Limit retries
    while torch.any(collision) and retries < max_retries:
        num_collisions = collision.sum().item()
        # Only generate new offsets for collisions
        new_rand_offset = torch.randint(1, num_classes, (num_collisions,), device=device, dtype=torch.long)
        # Apply new offset only where collision occurred
        neg_labels[collision] = (base_labels[collision] + new_rand_offset) % num_classes
        collision = neg_labels == base_labels # Recheck collision
        retries += 1
    # If still colliding after retries (very rare), force a change
    if retries == max_retries and torch.any(collision):
        logger.warning(f"Could not guarantee distinct negative labels after {max_retries} retries. Forcing increment.")
        neg_labels[collision] = (neg_labels[collision] + 1) % num_classes

    one_hot_neg = F.one_hot(neg_labels, num_classes=num_classes).to(device=device, dtype=torch.float)
    label_patch_neg = torch.where(one_hot_neg == 1, replace_value_on, replace_value_off)
    neg_images = base_images.clone()
    neg_images_flat_view = neg_images.view(batch_size, -1)
    neg_images_flat_view[:, :num_classes] = label_patch_neg
    neg_flattened = neg_images_flat_view

    # --- Neutral Samples ---
    # Reference uses a uniform label vector (e.g., [0.1]*num_classes)
    neutral_patch = torch.full((batch_size, num_classes), neutral_value, device=device, dtype=torch.float)
    neutral_images = base_images.clone()
    neutral_images_flat_view = neutral_images.view(batch_size, -1)
    neutral_images_flat_view[:, :num_classes] = neutral_patch
    neutral_flattened = neutral_images_flat_view

    # Detach all outputs as they are inputs for the model
    return pos_flattened.detach(), neg_flattened.detach(), neutral_flattened.detach()


# --- train_ff_model (Implement Optimizer Correction) ---
def train_ff_model(
    model: FF_MLP, # Use the modified FF_MLP
    train_loader: DataLoader,
    val_loader: Optional[DataLoader], # FF Eval uses validation set
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None, # Ignored by FF
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float: # Returns overall peak memory
    """
    Trains the MODIFIED FF_MLP model (Hinton style).
    Correctly sets up SGD optimizer based on reference.
    """
    model.to(device)
    logger.info(f"Starting Forward-Forward (Hinton style) training using modified FF_MLP.")
    if input_adapter is not None: logger.warning("FF Training: 'input_adapter' provided but FF_MLP uses internal logic. Adapter ignored.")

    train_config = config.get("training", {})
    algo_config = config.get("algorithm_params", {})
    loader_config = config.get("data_loader", {})
    data_config = config.get("data", {})
    checkpoint_config = config.get("checkpointing", {})

    optimizer_type = algo_config.get("optimizer_type", "AdamW")
    try:
        ff_lr = float(algo_config.get("ff_learning_rate", 1e-3))
        ff_wd = float(algo_config.get("ff_weight_decay", 3e-4))
        ds_lr = float(algo_config.get("downstream_learning_rate", 1e-2))
        ds_wd = float(algo_config.get("downstream_weight_decay", 3e-3))
        # <<< CORRECTION: Read momentum params for SGD >>>
        ff_momentum = float(algo_config.get("ff_momentum", 0.9)) # Default to 0.9 like ref
        ds_momentum = float(algo_config.get("downstream_momentum", 0.9)) # Default to 0.9 like ref
    except (ValueError, TypeError) as e: raise ValueError(f"Invalid LR/WD/Momentum format: {e}") from e

    epochs = train_config.get("epochs", 100)
    log_interval = train_config.get("log_interval", 100)
    num_classes = data_config.get("num_classes", 10)
    batch_size = loader_config.get("batch_size", 100) # Use 100 like ref
    checkpoint_dir = checkpoint_config.get("checkpoint_dir", None)
    save_best_metric = "ff_val_accuracy"; save_best_metric_mode = "max"

    # --- Optimizer Setup (Corrected for SGD) ---
    ff_layer_params = [p for layer in model.layers for p in layer.parameters() if p.requires_grad]
    classifier_params = [p for p in model.linear_classifier.parameters() if p.requires_grad]
    optimizer_groups = []
    # Group for FF layers (W1, W2, ...)
    if ff_layer_params:
        group_ff = {"params": ff_layer_params, "lr": ff_lr, "weight_decay": ff_wd}
        # <<< CORRECTION: Add momentum if optimizer is SGD >>>
        if optimizer_type.lower() == 'sgd': group_ff['momentum'] = ff_momentum
        optimizer_groups.append(group_ff)
        logger.info(f"Opt Group 0 (FF): LR={ff_lr}, WD={ff_wd}" + (f", Mom={ff_momentum}" if 'momentum' in group_ff else ""))
    # Group for Downstream Classifier
    if classifier_params:
        group_ds = {"params": classifier_params, "lr": ds_lr, "weight_decay": ds_wd}
        # <<< CORRECTION: Add momentum if optimizer is SGD >>>
        if optimizer_type.lower() == 'sgd': group_ds['momentum'] = ds_momentum
        optimizer_groups.append(group_ds)
        logger.info(f"Opt Group 1 (DS): LR={ds_lr}, WD={ds_wd}" + (f", Mom={ds_momentum}" if 'momentum' in group_ds else ""))

    if not optimizer_groups: logger.error("FF_MLP: No trainable parameters found."); return 0.0

    try:
        # <<< CORRECTION: Select optimizer based on config type >>>
        if optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(optimizer_groups)
        elif optimizer_type.lower() == "adamw":
             optimizer = optim.AdamW(optimizer_groups)
        elif optimizer_type.lower() == "adam":
             optimizer = optim.Adam(optimizer_groups)
        else: raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        logger.info(f"Using optimizer: {optimizer_type}")
    except Exception as e: logger.error(f"Failed to create optimizer: {e}", exc_info=True); return 0.0

    # --- Training Loop ---
    best_metric_value = -float("inf"); peak_mem_train = 0.0; run_start_time = time.time()

    for epoch in range(epochs):
        model.train(); epoch_start_time = time.time()
        epoch_total_loss, epoch_samples, epoch_ff_loss_total, epoch_peer_loss_total = 0.0, 0, 0.0, 0.0
        epoch_cls_loss_total, epoch_cls_acc_total = 0.0, 0.0
        epoch_layer_ff_acc_avg = {f"Layer_{i+1}": 0.0 for i in range(model.num_layers)}
        num_batches_processed = 0; peak_mem_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"FF Epoch {epoch+1}/{epochs}", leave=False)

        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1; current_global_step = step_ref[0]; current_batch_size = images.size(0)
            images, labels = images.to(device), labels.to(device)
            # <<< CORRECTION: Use exact data generation function >>>
            try: pos_images_flat, neg_images_flat, neutral_images_flat = generate_ff_hinton_inputs(images, labels, num_classes, device)
            except Exception as e_gen: logger.error(f"Input gen error: {e_gen}", exc_info=True); continue

            stacked_z = torch.cat([pos_images_flat, neg_images_flat], dim=0)
            posneg_labels = torch.zeros(stacked_z.shape[0], device=device); posneg_labels[:current_batch_size] = 1
            try:
                # Main FF forward pass (updates internal state for downstream)
                ff_combined_loss, ff_metrics_dict = model.forward_ff_train(stacked_z, posneg_labels, current_batch_size)
                # Downstream forward pass (uses internal state)
                # <<< CORRECTION: Use neutral samples for downstream training pass >>>
                # Need to run ff_train first to populate internal state, then ds_only
                # But we need the grads from ds_only too.
                # Let's keep the structure where ff_train sets up _current_downstream_input
                # and ds_only consumes it. The total loss includes both.
                # The `neutral_images_flat` are only used for EVALUATION in the reference.
                # The `forward_downstream_only` uses the *detached* activations from the
                # *positive* samples run during `forward_ff_train`. This seems correct.
                cls_loss, cls_accuracy = model.forward_downstream_only(labels)

                # Ensure cls_loss requires grad if classifier params do
                if not torch.isnan(cls_loss) and not cls_loss.requires_grad and any(p.requires_grad for p in model.linear_classifier.parameters()):
                     cls_loss = cls_loss.clone().requires_grad_(True)

                # Combine losses for backward pass
                # Check for NaN before combining
                if torch.isnan(ff_combined_loss) or torch.isnan(cls_loss):
                     logger.warning(f"NaN loss detected (FF: {ff_combined_loss.item()}, Cls: {cls_loss.item()}). Skipping batch {batch_idx}.")
                     continue
                total_batch_loss = ff_combined_loss + cls_loss

            except Exception as e_fwd: logger.error(f"Forward/loss error: {e_fwd}", exc_info=True); continue
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss): logger.error(f"NaN/Inf total loss. Skipping."); continue

            optimizer.zero_grad(); total_batch_loss.backward(); optimizer.step()

            # --- Accumulate Metrics ---
            epoch_total_loss += total_batch_loss.item() * current_batch_size
            # Use .get with default for safety
            epoch_ff_loss_total += ff_metrics_dict.get("FF_Loss_Total", torch.tensor(0.0)).item() * current_batch_size
            epoch_peer_loss_total += ff_metrics_dict.get("Peer_Normalization_Loss_Total", torch.tensor(0.0)).item() * current_batch_size
            epoch_cls_loss_total += cls_loss.item() * current_batch_size
            epoch_cls_acc_total += cls_accuracy * current_batch_size # cls_accuracy is already scaled by batch size
            epoch_samples += current_batch_size; num_batches_processed += 1
            for i in range(model.num_layers): epoch_layer_ff_acc_avg[f"Layer_{i+1}"] += ff_metrics_dict.get(f"Layer_{i+1}/FF_Accuracy", 0.0) # Accumulate batch accuracy

            # --- Memory & Logging ---
            current_mem_used = float('nan')
            if nvml_active and gpu_handle: mem_info = get_gpu_memory_usage(gpu_handle); current_mem_used = mem_info[0] if mem_info else float('nan'); peak_mem_epoch = max(peak_mem_epoch, current_mem_used if not torch.isnan(torch.tensor(current_mem_used)) else 0.0)
            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                metrics_to_log = {
                    "global_step": current_global_step, "FF_Hinton/Train_Loss_Batch": total_batch_loss.item(),
                    "FF_Hinton/FF_Loss_Batch": ff_combined_loss.item(),
                    "FF_Hinton/PeerNorm_Loss_Batch": ff_metrics_dict.get("Peer_Normalization_Loss_Total", torch.tensor(0.0)).item(),
                    "FF_Hinton/Cls_Loss_Batch": cls_loss.item(), "FF_Hinton/Cls_Acc_Batch": cls_accuracy}
                for i in range(model.num_layers): metrics_to_log[f"Layer_{i+1}/FF_Acc_Batch"] = ff_metrics_dict.get(f"Layer_{i+1}/FF_Accuracy", 0.0)
                if not torch.isnan(torch.tensor(current_mem_used)): metrics_to_log["FF_Hinton/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)
                pbar.set_postfix(loss=f"{total_batch_loss.item():.4f}", cls_acc=f"{cls_accuracy:.2f}%")

        # --- End of Epoch ---
        peak_mem_train = max(peak_mem_train, peak_mem_epoch)
        avg_epoch_loss = epoch_total_loss / epoch_samples if epoch_samples > 0 else float('nan')
        avg_ff_loss = epoch_ff_loss_total / epoch_samples if epoch_samples > 0 else float('nan')
        avg_peer_loss = epoch_peer_loss_total / epoch_samples if epoch_samples > 0 else float('nan')
        avg_cls_loss = epoch_cls_loss_total / epoch_samples if epoch_samples > 0 else float('nan')
        avg_cls_acc = epoch_cls_acc_total / epoch_samples if epoch_samples > 0 else float('nan') # Correct: accuracy already weighted by batch size
        for key in epoch_layer_ff_acc_avg: epoch_layer_ff_acc_avg[key] = epoch_layer_ff_acc_avg[key] / epoch_samples if epoch_samples > 0 else 0.0 # Normalize FF Acc
        epoch_duration = time.time() - epoch_start_time
        # Get current LR from optimizer group 0 (FF) and 1 (DS) if they exist
        current_lr_ff = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups)>0 else float('nan')
        current_lr_ds = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups)>1 else float('nan')

        val_results = {}
        if val_loader:
            # <<< CORRECTION: Pass neutral_value if needed by eval (unlikely needed here) >>>
            val_results = evaluate_ff_model(model, val_loader, device) # Call correct eval
            logger.info(f"FF Validation Epoch {epoch+1}/{epochs} - Accuracy: {val_results.get('eval_accuracy', 'N/A'):.2f}%")
        else: val_results = {"eval_accuracy": float("nan"), "eval_loss": float("nan")}

        epoch_summary_metrics = {
            "global_step": current_global_step, "FF_Hinton/Train_Loss_Epoch": avg_epoch_loss, "FF_Hinton/FF_Loss_Epoch": avg_ff_loss,
            "FF_Hinton/PeerNorm_Loss_Epoch": avg_peer_loss, "FF_Hinton/Cls_Loss_Epoch": avg_cls_loss, "FF_Hinton/Cls_Acc_Epoch": avg_cls_acc,
            "FF_Hinton/Val_Acc_Epoch": val_results["eval_accuracy"], "FF_Hinton/Epoch_Duration_Sec": epoch_duration,
            "FF_Hinton/LR_FF_Layers": current_lr_ff, "FF_Hinton/LR_Downstream": current_lr_ds, "FF_Hinton/Epoch": epoch + 1,
            "FF_Hinton/Peak_GPU_Mem_Epoch_MiB": peak_mem_epoch}
        for i in range(model.num_layers): epoch_summary_metrics[f"Layer_{i+1}/FF_Acc_EpochAvg"] = epoch_layer_ff_acc_avg[f"Layer_{i+1}"]
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)
        logger.info(f"FF Epoch {epoch+1}/{epochs} | Train Loss: {avg_epoch_loss:.4f}, Cls Acc: {avg_cls_acc:.2f}% | Val Acc: {val_results['eval_accuracy']:.2f}% | Peak Mem: {peak_mem_epoch:.1f} MiB | Duration: {format_time(epoch_duration)}")

        current_metric_value = val_results["eval_accuracy"]; is_best = False
        if not torch.isnan(torch.tensor(current_metric_value)) and current_metric_value > best_metric_value:
            best_metric_value = current_metric_value; is_best = True; logger.info(f"Epoch {epoch+1}: New best val acc: {best_metric_value:.2f}%")

        if checkpoint_dir:
            create_directory_if_not_exists(checkpoint_dir)
            save_checkpoint(
                state={"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "best_metric_value": best_metric_value, "val_accuracy": current_metric_value},
                is_best=is_best, checkpoint_dir=checkpoint_dir, filename=f"ff_checkpoint_epoch_{epoch+1}.pth",
                best_filename=f"ff_{config.get('experiment_name', 'model')}_best.pth")

    total_training_time = time.time() - run_start_time
    logger.info(f"Finished Forward-Forward (Hinton) training. Total time: {format_time(total_training_time)}")
    return peak_mem_train

# --- evaluate_ff_model (Use Corrected Data Generation) ---
def evaluate_ff_model(
    model: FF_MLP, # Use the modified FF_MLP
    data_loader: DataLoader,
    device: torch.device,
    **kwargs, # Accept extra args for consistent signature
) -> Dict[str, float]:
    """Evaluates the modified FF_MLP model using multi-pass goodness summation."""
    model.eval(); model.to(device)
    num_classes = model.num_classes
    logger.info(f"Evaluating FF (Hinton style) model using multi-pass inference.")
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating FF (Hinton) Model", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device); batch_size = images.shape[0]
            batch_total_goodness = torch.zeros((batch_size, num_classes), device=device)
            for label_candidate in range(num_classes):
                candidate_labels = torch.full((batch_size,), label_candidate, dtype=torch.long, device=device)
                # <<< CORRECTION: Use exact data generation function >>>
                try: ff_input_candidate, _, _ = generate_ff_hinton_inputs(images, candidate_labels, num_classes, device)
                except Exception as e_gen: logger.error(f"Eval input gen err {label_candidate}: {e_gen}"); batch_total_goodness[:, label_candidate] = -torch.inf; continue
                try:
                    layer_goodness_list = model.forward_goodness_per_layer(ff_input_candidate)
                    if not layer_goodness_list: total_goodness_candidate = torch.zeros((batch_size,), device=device); logger.warning(f"Eval no goodness {label_candidate}.")
                    # <<< CORRECTION: Reference sums goodness over layers 1..N-1 >>>
                    # Paper Fig 2: "Hidden activities in all but the first hidden layer are then used..."
                    # Let's match the reference repo's logic which might sum all layers. If not, adjust here.
                    # Reference ff_model.py evaluation accumulates goodness from ALL layers (0 to N-1).
                    else: total_goodness_candidate = torch.stack(layer_goodness_list, dim=0).sum(dim=0)
                    if total_goodness_candidate.shape != (batch_size,): raise ValueError(f"Bad goodness shape: {total_goodness_candidate.shape}")
                except Exception as e_fwd: logger.error(f"Eval fwd err {label_candidate}: {e_fwd}"); batch_total_goodness[:, label_candidate] = -torch.inf; continue
                batch_total_goodness[:, label_candidate] = total_goodness_candidate
            try:
                all_inf_mask = torch.all(torch.isinf(batch_total_goodness) & (batch_total_goodness < 0), dim=1)
                predicted_labels = torch.zeros_like(labels)
                if torch.any(~all_inf_mask): valid_indices = ~all_inf_mask; predicted_labels[valid_indices] = torch.argmax(batch_total_goodness[valid_indices], dim=1)
                if torch.any(all_inf_mask): logger.warning(f"Eval Batch {batch_idx+1}: {all_inf_mask.sum().item()}/{batch_size} samples failed all candidates.")
            except Exception as e_pred: logger.error(f"Eval pred err Batch {batch_idx+1}: {e_pred}"); predicted_labels = torch.zeros_like(labels)
            total_correct += (predicted_labels == labels).sum().item(); total_samples += batch_size
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"FF Evaluation Accuracy (Hinton Multi-Pass): {accuracy:.2f}%")
    return {"eval_accuracy": accuracy, "eval_loss": float("nan")} # FF eval doesn't compute loss this way