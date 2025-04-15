# File: ./src/algorithms/ff.py # <<< MODIFIED - Refined Negative Sampling >>>
# --------------------------------------------------------------------------------
# File: ./src/algorithms/ff.py (Refined Negative Sampling)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np # Keep numpy for potential temporary use if torch choice is complex
import logging
from tqdm import tqdm
import pynvml # Import for type hint
from typing import Dict, Any, Optional, Tuple, List, Callable
import os
import time

from src.architectures.ff_mlp import FF_MLP # Ensure this path is correct
from src.utils.logging_utils import log_metrics
from src.utils.helpers import format_time, save_checkpoint, create_directory_if_not_exists
from src.utils.monitoring import get_gpu_memory_usage

logger = logging.getLogger(__name__)

# --- Helper for Pixel Label Embedding (Refined Negative Sampling) ---
def generate_ff_hinton_inputs(
    base_images: torch.Tensor,
    base_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    replace_value_on: float = 1.0,
    replace_value_off: float = 0.0,
    neutral_value: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates positive, negative, and neutral flattened image tensors for FF training
    using Hinton's pixel replacement method.
    <<< CORRECTION: Uses remove-then-choose logic for negative labels matching reference. >>>
    """
    batch_size = base_images.shape[0]
    # Ensure base_images is float before flattening and modifying
    base_flat_view = base_images.view(batch_size, -1).float()
    image_pixels = base_flat_view.shape[1]
    if num_classes > image_pixels: raise ValueError(f"num_classes ({num_classes}) > total pixels ({image_pixels}). Cannot embed label.")

    # Positive Sample
    one_hot_pos = F.one_hot(base_labels, num_classes=num_classes).to(device=device, dtype=torch.float)
    label_patch_pos = torch.where(one_hot_pos == 1, replace_value_on, replace_value_off)
    pos_flattened = base_flat_view.clone(); pos_flattened[:, :num_classes] = label_patch_pos

    # Negative Sample (Remove-then-choose logic)
    neg_labels = torch.zeros_like(base_labels)
    possible_labels = torch.arange(num_classes, device=device)
    for i in range(batch_size):
        true_label = base_labels[i].item()
        # Create mask to select labels *not* equal to the true label
        mask = possible_labels != true_label
        # Select from the valid choices
        valid_choices = possible_labels[mask]
        if len(valid_choices) == 0:
            # This should only happen if num_classes is 1
            logger.warning(f"Only one possible class ({num_classes}), cannot choose a different negative label for sample {i}. Using original label.")
            neg_labels[i] = true_label
        else:
            # Randomly choose one index from the valid choices
            # Use torch.randint for reproducibility with torch seed
            choice_index = torch.randint(len(valid_choices), (1,), device=device).item()
            neg_labels[i] = valid_choices[choice_index]

    one_hot_neg = F.one_hot(neg_labels, num_classes=num_classes).to(device=device, dtype=torch.float)
    label_patch_neg = torch.where(one_hot_neg == 1, replace_value_on, replace_value_off)
    neg_flattened = base_flat_view.clone(); neg_flattened[:, :num_classes] = label_patch_neg

    # Neutral Sample
    neutral_patch = torch.full((batch_size, num_classes), neutral_value, device=device, dtype=torch.float)
    neutral_flattened = base_flat_view.clone(); neutral_flattened[:, :num_classes] = neutral_patch

    return pos_flattened.detach(), neg_flattened.detach(), neutral_flattened.detach()


# --- LR Cooldown function from reference utils.py (Unchanged) ---
def get_linear_cooldown_lr(initial_lr: float, epoch: int, total_epochs: int):
    current_epoch_num = epoch + 1
    if current_epoch_num > (total_epochs // 2):
        # Linear cooldown from initial_lr to ~0 over the second half
        lr_factor = 2.0 * (1.0 + total_epochs - current_epoch_num) / float(total_epochs); new_lr = initial_lr * lr_factor
        # Ensure LR doesn't go below a minimum threshold (e.g., 1e-9)
        return max(new_lr, 1e-9)
    else:
        # Keep initial LR for the first half
        return initial_lr

# --- train_ff_model (Unchanged from previous correction) ---
def train_ff_model(
    model: FF_MLP, train_loader: DataLoader, val_loader: Optional[DataLoader], config: Dict[str, Any], device: torch.device,
    wandb_run: Optional[Any] = None, input_adapter: Optional[Callable] = None, step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, nvml_active: bool = False,
) -> float: # Returns overall peak memory
    model.to(device); logger.info(f"Starting Forward-Forward (Hinton style) training using modified FF_MLP.")
    if input_adapter is not None: logger.warning("FF Training: 'input_adapter' provided but FF_MLP uses internal logic. Adapter ignored.")
    train_config = config.get("training", {}); algo_config = config.get("algorithm_params", {}); loader_config = config.get("data_loader", {}); data_config = config.get("data", {}); checkpoint_config = config.get("checkpointing", {})
    optimizer_type = algo_config.get("optimizer_type", "SGD")
    try:
        initial_ff_lr = float(algo_config.get("ff_learning_rate", 1e-3)); initial_ds_lr = float(algo_config.get("downstream_learning_rate", 1e-2))
        ff_wd = float(algo_config.get("ff_weight_decay", 3e-4)); ds_wd = float(algo_config.get("downstream_weight_decay", 3e-3))
        ff_momentum = float(algo_config.get("ff_momentum", 0.9)); ds_momentum = float(algo_config.get("downstream_momentum", 0.9))
    except (ValueError, TypeError) as e: raise ValueError(f"Invalid LR/WD/Momentum format in config: {e}") from e
    epochs = train_config.get("epochs", 100); log_interval = train_config.get("log_interval", 100); num_classes = data_config.get("num_classes", 10)
    checkpoint_dir = checkpoint_config.get("checkpoint_dir", None); save_best_metric = checkpoint_config.get("save_best_metric", "ff_val_accuracy").lower(); save_best_metric_mode = "max"
    ff_layer_params = [p for layer in model.layers for p in layer.parameters() if p.requires_grad]
    classifier_params = [p for p in model.linear_classifier.parameters() if p.requires_grad]
    optimizer_groups = []
    if ff_layer_params:
        group_ff = {"params": ff_layer_params, "lr": initial_ff_lr, "weight_decay": ff_wd};
        if optimizer_type.lower() == 'sgd': group_ff['momentum'] = ff_momentum
        optimizer_groups.append(group_ff); logger.info(f"Opt Group 0 (FF): Initial LR={initial_ff_lr}, WD={ff_wd}" + (f", Mom={ff_momentum}" if 'momentum' in group_ff else ""))
    else: logger.warning("FF_MLP: No trainable parameters found in FF layers.")
    if classifier_params:
        group_ds = {"params": classifier_params, "lr": initial_ds_lr, "weight_decay": ds_wd};
        if optimizer_type.lower() == 'sgd': group_ds['momentum'] = ds_momentum
        optimizer_groups.append(group_ds); logger.info(f"Opt Group 1 (DS): Initial LR={initial_ds_lr}, WD={ds_wd}" + (f", Mom={ds_momentum}" if 'momentum' in group_ds else ""))
    else: logger.warning("FF_MLP: No trainable parameters found in downstream classifier.")
    if not optimizer_groups: logger.error("FF_MLP: No trainable parameters found."); return 0.0
    try:
        if optimizer_type.lower() == "sgd": optimizer = optim.SGD(optimizer_groups)
        elif optimizer_type.lower() == "adamw": logger.warning("Reference uses SGD, but AdamW specified. Applying group settings if applicable."); optimizer = optim.AdamW(optimizer_groups)
        elif optimizer_type.lower() == "adam": logger.warning("Reference uses SGD, but Adam specified. Applying group settings if applicable."); optimizer = optim.Adam(optimizer_groups)
        else: raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        logger.info(f"Using optimizer: {optimizer_type}")
    except Exception as e: logger.error(f"Failed to create optimizer: {e}", exc_info=True); return 0.0
    best_metric_value = -float("inf"); peak_mem_train = 0.0; run_start_time = time.time()
    for epoch in range(epochs):
        current_lr_ff = get_linear_cooldown_lr(initial_ff_lr, epoch, epochs); current_lr_ds = get_linear_cooldown_lr(initial_ds_lr, epoch, epochs)
        if len(optimizer.param_groups) > 0: optimizer.param_groups[0]['lr'] = current_lr_ff
        if len(optimizer.param_groups) > 1: optimizer.param_groups[1]['lr'] = current_lr_ds
        logger.debug(f"Epoch {epoch+1}/{epochs}: LR Update - FF={current_lr_ff:.6f}, DS={current_lr_ds:.6f}")
        model.train(); epoch_start_time = time.time()
        epoch_total_combined_loss, epoch_samples, epoch_ff_loss_total, epoch_peer_loss_total = 0.0, 0, 0.0, 0.0
        epoch_cls_loss_total, epoch_cls_acc_total = 0.0, 0.0; epoch_layer_ff_acc_sum = {f"Layer_{i+1}": 0.0 for i in range(model.num_layers)}
        num_batches_processed = 0; peak_mem_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"FF Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1; current_global_step = step_ref[0]; current_batch_size = images.size(0)
            images, labels = images.to(device), labels.to(device)
            try: pos_images_flat, neg_images_flat, neutral_images_flat = generate_ff_hinton_inputs(images, labels, num_classes, device)
            except Exception as e_gen: logger.error(f"Input gen error: {e_gen}", exc_info=True); continue
            stacked_z = torch.cat([pos_images_flat, neg_images_flat], dim=0)
            posneg_labels = torch.zeros(stacked_z.shape[0], device=device); posneg_labels[:current_batch_size] = 1
            try:
                ff_combined_loss, ff_metrics_dict = model.forward_ff_train(stacked_z, posneg_labels, current_batch_size)
                cls_loss, cls_accuracy = model.forward_downstream_only(labels)
                if torch.isnan(ff_combined_loss) or torch.isinf(ff_combined_loss): logger.warning(f"NaN/Inf FF loss ({ff_combined_loss.item()}). Skip."); continue
                if torch.isnan(cls_loss) or torch.isinf(cls_loss): logger.warning(f"NaN/Inf Cls loss ({cls_loss.item()}). Skip."); continue
                if classifier_params and not cls_loss.requires_grad: cls_loss = cls_loss.clone().requires_grad_(True)
                total_batch_loss = ff_combined_loss + cls_loss
            except Exception as e_fwd: logger.error(f"Forward/loss error: {e_fwd}", exc_info=True); continue
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss): logger.error(f"NaN/Inf total loss before backward ({total_batch_loss.item()}). Skip."); continue
            optimizer.zero_grad()
            try: total_batch_loss.backward()
            except Exception as e_bwd: logger.error(f"Backward pass error: {e_bwd}", exc_info=True); continue
            optimizer.step()
            epoch_total_combined_loss += total_batch_loss.item() * current_batch_size
            epoch_ff_loss_total += ff_metrics_dict.get("FF_Loss_Total", torch.tensor(0.0)).item() * current_batch_size
            epoch_peer_loss_total += ff_metrics_dict.get("Peer_Normalization_Loss_Total", torch.tensor(0.0)).item() * current_batch_size
            epoch_cls_loss_total += cls_loss.item() * current_batch_size; epoch_cls_acc_total += cls_accuracy * current_batch_size
            epoch_samples += current_batch_size; num_batches_processed += 1
            for i in range(model.num_layers): layer_key = f"Layer_{i+1}"; acc_key = f"{layer_key}/FF_Accuracy"; epoch_layer_ff_acc_sum[layer_key] += ff_metrics_dict.get(acc_key, 0.0) * current_batch_size
            current_mem_used = float('nan')
            if nvml_active and gpu_handle: mem_info = get_gpu_memory_usage(gpu_handle); current_mem_used = mem_info[0] if mem_info else float('nan'); peak_mem_epoch = max(peak_mem_epoch, current_mem_used if not torch.isnan(torch.tensor(current_mem_used)) else 0.0)
            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                metrics_to_log = {"global_step": current_global_step, "FF_Hinton/Train_Loss_Batch": total_batch_loss.item(), "FF_Hinton/FF_Loss_Batch": ff_metrics_dict.get("FF_Loss_Total", torch.tensor(0.0)).item(), "FF_Hinton/PeerNorm_Loss_Batch": ff_metrics_dict.get("Peer_Normalization_Loss_Total", torch.tensor(0.0)).item(), "FF_Hinton/Cls_Loss_Batch": cls_loss.item(), "FF_Hinton/Cls_Acc_Batch": cls_accuracy}
                for i in range(model.num_layers): metrics_to_log[f"Layer_{i+1}/FF_Acc_Batch"] = ff_metrics_dict.get(f"Layer_{i+1}/FF_Accuracy", 0.0)
                if not torch.isnan(torch.tensor(current_mem_used)): metrics_to_log["FF_Hinton/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True); pbar.set_postfix(loss=f"{total_batch_loss.item():.4f}", cls_acc=f"{cls_accuracy:.2f}%")
        peak_mem_train = max(peak_mem_train, peak_mem_epoch)
        if epoch_samples == 0: logger.warning(f"Epoch {epoch+1} completed with 0 samples. Skip eval."); continue
        avg_epoch_loss = epoch_total_combined_loss / epoch_samples; avg_ff_loss = epoch_ff_loss_total / epoch_samples; avg_peer_loss = epoch_peer_loss_total / epoch_samples; avg_cls_loss = epoch_cls_loss_total / epoch_samples; avg_cls_acc = epoch_cls_acc_total / epoch_samples
        epoch_layer_ff_acc_avg = {};
        for i in range(model.num_layers): layer_key = f"Layer_{i+1}"; epoch_layer_ff_acc_avg[layer_key] = epoch_layer_ff_acc_sum[layer_key] / epoch_samples
        epoch_duration = time.time() - epoch_start_time
        val_results = {}
        if val_loader: val_results = evaluate_ff_model(model, val_loader, device); logger.info(f"FF Validation Epoch {epoch+1}/{epochs} - Accuracy: {val_results.get('eval_accuracy', 'N/A'):.2f}%")
        else: val_results = {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
        current_global_step = step_ref[0]; epoch_summary_metrics = {"global_step": current_global_step, "FF_Hinton/Train_Loss_Epoch": avg_epoch_loss, "FF_Hinton/FF_Loss_Epoch": avg_ff_loss, "FF_Hinton/PeerNorm_Loss_Epoch": avg_peer_loss, "FF_Hinton/Cls_Loss_Epoch": avg_cls_loss, "FF_Hinton/Cls_Acc_Epoch": avg_cls_acc, "FF_Hinton/Val_Acc_Epoch": val_results.get("eval_accuracy", float("nan")), "FF_Hinton/Epoch_Duration_Sec": epoch_duration, "FF_Hinton/LR_FF_Layers": current_lr_ff, "FF_Hinton/LR_Downstream": current_lr_ds, "FF_Hinton/Epoch": epoch + 1, "FF_Hinton/Peak_GPU_Mem_Epoch_MiB": peak_mem_epoch}
        for i in range(model.num_layers): epoch_summary_metrics[f"Layer_{i+1}/FF_Acc_EpochAvg"] = epoch_layer_ff_acc_avg[f"Layer_{i+1}"]
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True); logger.info(f"FF Epoch {epoch+1}/{epochs} | Train Loss: {avg_epoch_loss:.4f}, Cls Acc: {avg_cls_acc:.2f}% | Val Acc: {val_results.get('eval_accuracy', 'N/A'):.2f}% | Peak Mem: {peak_mem_epoch:.1f} MiB | Duration: {format_time(epoch_duration)}")
        current_metric_value = val_results.get("eval_accuracy", float("-inf")); is_best = False
        if not torch.isnan(torch.tensor(current_metric_value)) and current_metric_value > best_metric_value: best_metric_value = current_metric_value; is_best = True; logger.info(f"Epoch {epoch+1}: New best val acc: {best_metric_value:.2f}%")
        if checkpoint_dir:
            create_directory_if_not_exists(checkpoint_dir); save_checkpoint(state={"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "best_metric_value": best_metric_value, "val_accuracy": current_metric_value}, is_best=is_best, checkpoint_dir=checkpoint_dir, filename=f"ff_checkpoint_epoch_{epoch+1}.pth", best_filename=f"ff_{config.get('experiment_name', 'model')}_best.pth")
    total_training_time = time.time() - run_start_time; logger.info(f"Finished Forward-Forward (Hinton) training. Total time: {format_time(total_training_time)}")
    logger.info(f"NOTE: Reference implementation used PyTorch 1.11. Your environment uses {torch.__version__}. Small differences in final accuracy might arise from library versions or hardware.")
    return peak_mem_train

# --- evaluate_ff_model (Unchanged from previous correction) ---
def evaluate_ff_model(
    model: FF_MLP, data_loader: DataLoader, device: torch.device, **kwargs,
) -> Dict[str, float]:
    model.eval(); model.to(device); num_classes = model.num_classes
    logger.info(f"Evaluating FF (Hinton style) model using multi-pass inference.")
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating FF (Hinton) Model", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device); batch_size = images.shape[0]
            batch_total_goodness = torch.zeros((batch_size, num_classes), device=device)
            for label_candidate in range(num_classes):
                candidate_labels = torch.full((batch_size,), label_candidate, dtype=torch.long, device=device)
                try: ff_input_candidate, _, _ = generate_ff_hinton_inputs(images, candidate_labels, num_classes, device)
                except Exception as e_gen: logger.error(f"Eval input gen err for label {label_candidate}: {e_gen}"); batch_total_goodness[:, label_candidate] = -torch.inf; continue
                try:
                    layer_goodness_list = model.forward_goodness_per_layer(ff_input_candidate)
                    if not layer_goodness_list: total_goodness_candidate = torch.zeros((batch_size,), device=device); logger.warning(f"Eval no goodness {label_candidate}.")
                    elif len(layer_goodness_list) == 1: logger.debug(f"Eval: Using goodness from layer 0 for 1-layer net (label {label_candidate})."); total_goodness_candidate = layer_goodness_list[0]
                    else: logger.debug(f"Eval: Summing goodness from layers 1..{len(layer_goodness_list)-1} (label {label_candidate})."); total_goodness_candidate = torch.stack(layer_goodness_list[1:], dim=0).sum(dim=0)
                    if total_goodness_candidate.shape != (batch_size,): raise ValueError(f"Bad goodness shape: {total_goodness_candidate.shape}")
                except Exception as e_fwd: logger.error(f"Eval forward/goodness err for label {label_candidate}: {e_fwd}", exc_info=True); batch_total_goodness[:, label_candidate] = -torch.inf; continue
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
    return {"eval_accuracy": accuracy, "eval_loss": float("nan")}
# --------------------------------------------------------------------------------