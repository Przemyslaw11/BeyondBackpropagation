# File: src/algorithms/cafo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
from tqdm import tqdm
import pynvml # Import for type hint
from typing import Dict, Any, Optional, Callable, List, Type, Tuple
import os
import time
import math

from src.architectures.cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint, create_directory_if_not_exists, format_time
from src.utils.monitoring import get_gpu_memory_usage

logger = logging.getLogger(__name__)

# --- train_cafo_dfa_blocks (Remains Unchanged - Not relevant to predictor ES) ---
def train_cafo_dfa_blocks(
    model: CaFo_CNN,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float: # Return peak memory
    """
    Trains the blocks of the CaFo_CNN model using Direct Feedback Alignment (DFA).
    (Function content remains the same as provided previously)
    """
    logger.info("--- Starting CaFo Block Training (DFA) Phase ---")
    model.to(device)
    model.train()
    algo_config = config.get("algorithm_params", {}); data_config = config.get("data", {})
    num_classes = data_config.get("num_classes", 10); epochs = algo_config.get("block_training_epochs", 10)
    optimizer_name = algo_config.get("block_optimizer_type", "Adam"); lr = algo_config.get("block_lr", 0.0001)
    weight_decay = algo_config.get("block_weight_decay", 0.0); log_interval = algo_config.get("log_interval", 100)
    dfa_feedback_matrix_type = algo_config.get("dfa_feedback_matrix_type", "gaussian")
    try:
        last_block_idx = len(model.blocks) - 1; last_block_flat_dim = model.get_predictor_input_dim(last_block_idx)
        aux_layer = nn.Linear(last_block_flat_dim, num_classes).to(device); nn.init.kaiming_uniform_(aux_layer.weight, a=math.sqrt(5))
        if aux_layer.bias is not None: nn.init.zeros_(aux_layer.bias); logger.info(f"Created auxiliary layer for DFA block training: Linear({last_block_flat_dim}, {num_classes})")
    except Exception as e: logger.error(f"Failed to create auxiliary layer for DFA: {e}", exc_info=True); return float('nan')
    feedback_matrices = []
    error_dim = num_classes
    for i in range(len(model.blocks) - 1):
        block_output_flat_dim = model.get_predictor_input_dim(i); b_matrix = torch.randn(error_dim, block_output_flat_dim, device=device)
        if dfa_feedback_matrix_type == "uniform": b_matrix.uniform_(-1.0, 1.0)
        with torch.no_grad(): b_matrix /= torch.norm(b_matrix, dim=1, keepdim=True) + 1e-8
        b_matrix.requires_grad_(False); feedback_matrices.append(b_matrix); logger.debug(f"Created fixed feedback matrix B_{i} with shape {b_matrix.shape}")
    block_params = list(model.blocks.parameters()); aux_params = list(aux_layer.parameters()); all_params_to_train = block_params + aux_params
    if not all_params_to_train: logger.error("DFA Block Training: No parameters found to train."); return 0.0
    optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay}; optimizer = getattr(optim, optimizer_name)(all_params_to_train, **optimizer_kwargs); criterion = nn.CrossEntropyLoss()
    peak_mem_block_train = 0.0; block_train_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time(); model.train(); aux_layer.train(); total_loss = 0.0; total_correct = 0; total_samples = 0; peak_mem_block_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"DFA Block Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1; current_global_step = step_ref[0]; images, labels = images.to(device), labels.to(device); batch_size = images.size(0); labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
            activations = [images]; current_h = images
            for i, block in enumerate(model.blocks): current_h = block(current_h); activations.append(current_h)
            last_block_output = activations[-1]; last_block_output_flat = last_block_output.view(batch_size, -1); aux_output_logits = aux_layer(last_block_output_flat); loss = criterion(aux_output_logits, labels)
            optimizer.zero_grad(); loss.backward()
            with torch.no_grad(): global_error = (F.softmax(aux_output_logits, dim=1) - labels_one_hot).detach()
            for i in range(len(model.blocks) - 1):
                block_index = i; h_i = activations[block_index + 1]; h_prev = activations[block_index]; delta_h_i_flat = torch.matmul(global_error, feedback_matrices[block_index]); delta_h_i_spatial = delta_h_i_flat.view_as(h_i); target_block = model.blocks[block_index]
                try:
                   original_mode = target_block.training; target_block.train(); h_i_recompute = target_block(h_prev.detach())
                   grads = torch.autograd.grad(outputs=h_i_recompute, inputs=target_block.parameters(), grad_outputs=delta_h_i_spatial, allow_unused=True, retain_graph=False)
                   for param, grad in zip(target_block.parameters(), grads):
                       if grad is not None: param.grad = grad
                   target_block.train(original_mode)
                except Exception as e_dfa_grad: logger.error(f"Error computing/assigning DFA gradient for block {block_index}: {e_dfa_grad}", exc_info=True)
            optimizer.step()
            with torch.no_grad(): total_loss += loss.item() * batch_size; predicted_labels = torch.argmax(aux_output_logits, dim=1); total_correct += (predicted_labels == labels).sum().item(); total_samples += batch_size
            current_mem_used = float('nan')
            if nvml_active and gpu_handle: mem_info = get_gpu_memory_usage(gpu_handle);
            if mem_info: current_mem_used = mem_info[0]; peak_mem_block_epoch = max(peak_mem_block_epoch, current_mem_used)
            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                batch_accuracy = calculate_accuracy(aux_output_logits, labels); pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_accuracy:.2f}%")
                metrics_to_log = {"global_step": current_global_step, "CaFo_DFA/BlockTrain_Loss_Batch": loss.item(), "CaFo_DFA/BlockTrain_Acc_Batch": batch_accuracy}
                if not torch.isnan(torch.tensor(current_mem_used)): metrics_to_log["CaFo_DFA/BlockTrain_GPU_Mem_MiB_Batch"] = current_mem_used
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan'); avg_acc = (total_correct / total_samples) * 100.0 if total_samples > 0 else float('nan'); peak_mem_block_train = max(peak_mem_block_train, peak_mem_block_epoch); epoch_duration = time.time() - epoch_start_time
        logger.info(f"DFA Block Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}% | Peak Mem Epoch: {peak_mem_block_epoch:.1f} MiB | Duration: {format_time(epoch_duration)}")
        epoch_summary_metrics = {"global_step": current_global_step, "CaFo_DFA/BlockTrain_Loss_EpochAvg": avg_loss, "CaFo_DFA/BlockTrain_Acc_EpochAvg": avg_acc, "CaFo_DFA/BlockTrain_Peak_GPU_Mem_Epoch_MiB": peak_mem_block_epoch}
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)
    block_train_duration = time.time() - block_train_start_time; logger.info(f"--- Finished CaFo Block Training (DFA) Phase. Duration: {format_time(block_train_duration)} ---")
    if nvml_active and gpu_handle: mem_info = get_gpu_memory_usage(gpu_handle);
    if mem_info: peak_mem_block_train = max(peak_mem_block_train, mem_info[0])
    model.eval(); del aux_layer, feedback_matrices;
    for param in model.parameters(): param.requires_grad_(False)
    return peak_mem_block_train


# --- train_cafo_predictor_only (Remains Unchanged) ---
def train_cafo_predictor_only(
    block: CaFoBlock, predictor: CaFoPredictor, optimizer: optim.Optimizer, criterion: nn.Module, train_loader: DataLoader,
    epochs: int, device: torch.device, get_block_input_fn: Callable[[torch.Tensor], torch.Tensor],
    wandb_run: Optional[Any] = None, log_interval: int = 100, block_index: int = 0,
    step_ref: List[int] = [-1], gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, nvml_active: bool = False,
) -> Tuple[float, float, float]: # Return avg loss, avg accuracy, peak memory
    """
    Trains a single CaFoPredictor layer-wise, keeping the corresponding CaFoBlock frozen.
    (Function content remains the same as provided previously)
    """
    block.eval(); predictor.train(); block.to(device); predictor.to(device)
    log_prefix = f"Predictor_{block_index+1}"; logger.info(f"Starting CaFo training for {log_prefix} (Block {block_index+1} frozen)")
    peak_mem_predictor_train = 0.0; final_avg_epoch_loss = float('nan'); final_avg_epoch_accuracy = float('nan')
    for epoch in range(epochs):
        epoch_loss = 0.0; epoch_correct = 0; epoch_samples = 0; peak_mem_predictor_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"{log_prefix} Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1; current_global_step = step_ref[0]; images, labels = images.to(device), labels.to(device)
            with torch.no_grad(): block_input = get_block_input_fn(images); block_output = block(block_input)
            predictions = predictor(block_output.detach()); loss = criterion(predictions, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            with torch.no_grad(): pred_labels = torch.argmax(predictions, dim=1); batch_correct = (pred_labels == labels).sum().item()
            batch_size = labels.size(0); batch_accuracy = (batch_correct / batch_size) * 100.0 if batch_size > 0 else 0.0
            epoch_loss += loss.item() * batch_size; epoch_correct += batch_correct; epoch_samples += batch_size
            current_mem_used = float('nan')
            if nvml_active and gpu_handle and ((batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1):
                 mem_info = get_gpu_memory_usage(gpu_handle)
                 if mem_info: current_mem_used = mem_info[0]; peak_mem_predictor_epoch = max(peak_mem_predictor_epoch, current_mem_used)
            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_accuracy:.2f}%")
                metrics_to_log = {"global_step": current_global_step, f"{log_prefix}/Train_Loss_Batch": loss.item(), f"{log_prefix}/Train_Acc_Batch": batch_accuracy}
                if not torch.isnan(torch.tensor(current_mem_used)): metrics_to_log[f"{log_prefix}/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)
        final_avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float('nan')
        final_avg_epoch_accuracy = (epoch_correct / epoch_samples) * 100.0 if epoch_samples > 0 else float('nan')
        peak_mem_predictor_train = max(peak_mem_predictor_train, peak_mem_predictor_epoch)
        logger.info(f"{log_prefix} Epoch {epoch+1}/{epochs} - Avg Loss: {final_avg_epoch_loss:.4f}, Avg Acc: {final_avg_epoch_accuracy:.2f}%, Peak Mem Epoch: {peak_mem_predictor_epoch:.1f} MiB")
        epoch_summary_metrics = {"global_step": current_global_step, f"{log_prefix}/Train_Loss_EpochAvg": final_avg_epoch_loss, f"{log_prefix}/Train_Acc_EpochAvg": final_avg_epoch_accuracy, f"{log_prefix}/Peak_GPU_Mem_Epoch_MiB": peak_mem_predictor_epoch}
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)
    if nvml_active and gpu_handle: mem_info = get_gpu_memory_usage(gpu_handle);
    if mem_info: peak_mem_predictor_train = max(peak_mem_predictor_train, mem_info[0])
    logger.info(f"Finished CaFo training for {log_prefix}. Overall Peak Mem Predictor: {peak_mem_predictor_train:.1f} MiB")
    predictor.eval()
    return final_avg_epoch_loss, final_avg_epoch_accuracy, peak_mem_predictor_train


# --- train_cafo_model (MODIFIED - Corrected Early Stopping Logic) ---
def train_cafo_model(
    model: CaFo_CNN,  # Contains only blocks
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
    Orchestrates the training of CaFo_CNN.
    Optionally trains blocks using DFA first, then trains predictors layer-wise.
    <<< MODIFIED: Corrected Early Stopping logic based on aggregated validation performance. >>>
    Returns the overall peak GPU memory observed across all training phases.
    """
    model.to(device)
    algo_config = config.get("algorithm_params", {})
    train_config = config.get("training", {})
    train_blocks_flag = algo_config.get("train_blocks", False)
    num_blocks = len(model.blocks)
    peak_mem_train = 0.0

    # --- Early Stopping Configuration ---
    es_enabled = train_config.get("early_stopping_enabled", True)
    es_metric_key = train_config.get("early_stopping_metric", "cafo_val_accuracy").lower()
    es_patience = train_config.get("early_stopping_patience", 10)
    es_mode_config = train_config.get("early_stopping_mode", "max").lower()
    es_min_delta = train_config.get("early_stopping_min_delta", 0.0)
    epochs_no_improve = 0
    # Determine the actual ES mode based on metric (accuracy=max, loss=min)
    # Determine the metric name to extract from evaluate_cafo_model output
    if "accuracy" in es_metric_key:
        es_mode = "max"
        metric_name_for_eval = "eval_accuracy"
        if es_mode_config != "max": logger.warning(f"CaFo ES metric '{es_metric_key}' implies 'max' mode, but config has '{es_mode_config}'. Using 'max'.")
    elif "loss" in es_metric_key:
        es_mode = "min"
        metric_name_for_eval = "eval_loss"
        if es_mode_config != "min": logger.warning(f"CaFo ES metric '{es_metric_key}' implies 'min' mode, but config has '{es_mode_config}'. Using 'min'.")
    else:
        logger.warning(f"Cannot determine ES mode (max/min) for metric '{es_metric_key}'. Defaulting to accuracy/max.")
        es_mode = "max"
        metric_name_for_eval = "eval_accuracy"
        es_metric_key = "cafo_val_accuracy" # Use a default key

    best_es_metric_value = float('-inf') if es_mode == 'max' else float('inf')

    if es_enabled:
        if val_loader is None: logger.warning("CaFo Early stopping enabled but no validation loader provided. Disabling."); es_enabled = False
        else: logger.info(f"CaFo Early stopping enabled: Metric Key='{es_metric_key}' -> Eval Metric='{metric_name_for_eval}', Mode='{es_mode}', Patience={es_patience}, MinDelta={es_min_delta}")
    else: logger.info("CaFo Early stopping disabled.")
    # --- End Early Stopping Config ---


    # Optional Block Training (DFA)
    if train_blocks_flag:
        peak_mem_block_phase = train_cafo_dfa_blocks(model, train_loader, config, device, wandb_run, step_ref, gpu_handle, nvml_active)
        peak_mem_train = max(peak_mem_train, peak_mem_block_phase)
    else: logger.info("Skipping block training phase."); model.eval();
    for param in model.blocks.parameters(): param.requires_grad_(False)

    # Predictor Training Phase Setup
    logger.info(f"--- Starting CaFo Predictor Training Phase for {num_blocks} blocks ---")
    if input_adapter is not None: logger.warning("CaFo Training: 'input_adapter' ignored.")
    predictor_optimizer_name = algo_config.get("predictor_optimizer_type", "Adam")
    predictor_lr = algo_config.get("predictor_lr", 0.001)
    predictor_weight_decay = algo_config.get("predictor_weight_decay", 0.0)
    criterion_name = algo_config.get("loss_type", "CrossEntropyLoss"); criterion = nn.CrossEntropyLoss() if criterion_name.lower() == "crossentropyloss" else None
    if criterion is None: raise ValueError(f"Unsupported criterion: {criterion_name}")
    epochs_per_block = algo_config.get("num_epochs_per_block", 10)
    log_interval = algo_config.get("log_interval", 100); optimizer_params_extra = algo_config.get("optimizer_params", {})
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None); aggregation_method = algo_config.get("aggregation_method", "sum")

    # Create Predictors
    predictors = nn.ModuleList()
    for i in range(num_blocks):
        try: in_features = model.get_predictor_input_dim(i); predictor = CaFoPredictor(in_features, model.num_classes).to(device); predictors.append(predictor); logger.info(f"Created predictor {i+1} with input dim {in_features}")
        except Exception as e: logger.error(f"Failed to create predictor for block {i}: {e}", exc_info=True); raise

    # Train Predictors Sequentially with Early Stopping Check
    current_block_input_fn = lambda img: img.to(device)
    peak_mem_predictor_phase_overall = 0.0

    for i in range(num_blocks):
        predictor_log_idx = i + 1; log_prefix = f"Predictor_{predictor_log_idx}"
        logger.info(f"--- Training {log_prefix} ---")
        block = model.blocks[i]; block.eval()
        predictor = predictors[i]
        for param in predictor.parameters(): param.requires_grad_(True)
        params_to_optimize = list(predictor.parameters())
        predictor_peak_mem_this = 0.0; final_avg_loss = float('nan'); final_avg_acc = float('nan')

        if not params_to_optimize: logger.warning(f"{log_prefix} has no parameters. Skipping training.")
        else:
            optimizer_kwargs = {"lr": predictor_lr, "weight_decay": predictor_weight_decay, **optimizer_params_extra}
            optimizer = getattr(optim, predictor_optimizer_name)(params_to_optimize, **optimizer_kwargs)
            final_avg_loss, final_avg_acc, predictor_peak_mem_this = train_cafo_predictor_only(
                block, predictor, optimizer, criterion, train_loader, epochs_per_block, device,
                current_block_input_fn, wandb_run, log_interval, i, step_ref, gpu_handle, nvml_active
            )
            peak_mem_predictor_phase_overall = max(peak_mem_predictor_phase_overall, predictor_peak_mem_this)
            for param in predictor.parameters(): param.requires_grad = False # Freeze after training
            predictor.eval()

        # Log predictor summary
        current_global_step = step_ref[0]
        predictor_summary_metrics = {"global_step": current_global_step, f"{log_prefix}/Train_Loss_LayerAvg": final_avg_loss, f"{log_prefix}/Train_Acc_LayerAvg": final_avg_acc, f"{log_prefix}/Peak_GPU_Mem_Layer_MiB": predictor_peak_mem_this}
        log_metrics(predictor_summary_metrics, wandb_run=wandb_run, commit=False) # Commit after ES check

        if checkpoint_dir and params_to_optimize: create_directory_if_not_exists(checkpoint_dir); save_checkpoint(state={"state_dict": predictor.state_dict(), "predictor_index": i}, is_best=False, filename=f"cafo_predictor_{i}_complete.pth", checkpoint_dir=checkpoint_dir)

        # <<< Corrected Early Stopping Check >>>
        if es_enabled:
            logger.info(f"--- Evaluating Aggregated Performance after Training Predictor {i+1} for Early Stopping ---")
            current_predictors_trained = predictors[:i+1]
            val_eval_criterion = nn.CrossEntropyLoss() if "loss" in metric_name_for_eval else None
            val_results = evaluate_cafo_model(
                model=model, data_loader=val_loader, device=device,
                criterion=val_eval_criterion, predictors=current_predictors_trained,
                aggregation_method=aggregation_method
            )
            # Extract the specific metric needed for comparison
            current_metric_value = val_results.get(metric_name_for_eval, float('nan'))

            es_val_log = {"global_step": current_global_step, f"CaFo_ES/Val_{metric_name_for_eval.replace('eval_', '')}": current_metric_value, f"CaFo_ES/Predictors_Used": i + 1}
            log_metrics(es_val_log, wandb_run=wandb_run, commit=True) # Commit validation log

            logger.info(f"Predictor {i+1} ES Check: Current Val Metric ({metric_name_for_eval}) = {current_metric_value:.4f}, Best = {best_es_metric_value:.4f}, Patience = {epochs_no_improve}/{es_patience}")

            if torch.isnan(torch.tensor(current_metric_value)):
                logger.warning(f"Predictor {i+1}: Early stopping metric '{metric_name_for_eval}' is NaN. Treating as no improvement.")
                epochs_no_improve += 1
            else:
                improved = False
                if es_mode == "max": improved = current_metric_value > best_es_metric_value + es_min_delta
                else: improved = current_metric_value < best_es_metric_value - es_min_delta

                if improved:
                    best_es_metric_value = current_metric_value
                    epochs_no_improve = 0
                    logger.info(f"Predictor {i+1}: Early stopping metric improved. Reset patience.")
                else:
                    epochs_no_improve += 1
                    logger.info(f"Predictor {i+1}: Early stopping metric did not improve.")

            if epochs_no_improve >= es_patience:
                logger.info(f"--- Early Stopping Triggered after Predictor {i+1} ---")
                logger.info(f"Metric '{es_metric_key}' ({metric_name_for_eval}) did not improve for {es_patience} stages (Best: {best_es_metric_value:.4f}).")
                break # Exit the predictor training loop
        else:
             log_metrics({}, wandb_run=wandb_run, commit=True) # Ensure commit if ES disabled

        # Prepare input function for the next predictor
        def create_next_input_fn(trained_block_idx: int, previous_input_fn: Callable):
            block_k = model.blocks[trained_block_idx]; block_k.eval(); block_k.to(device)
            @torch.no_grad()
            def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                block_input = previous_input_fn(img_batch).to(device); block_output = block_k(block_input)
                return block_output.detach()
            return next_input_fn
        current_block_input_fn = create_next_input_fn(i, current_block_input_fn)
    # --- End Predictor Loop ---

    logger.info("Finished all layer-wise CaFo predictor training.")
    model.trained_predictors = predictors
    peak_mem_train = max(peak_mem_train, peak_mem_predictor_phase_overall)
    return peak_mem_train


# --- evaluate_cafo_model (Remains Unchanged) ---
def evaluate_cafo_model(
    model: CaFo_CNN, data_loader: DataLoader, device: torch.device, criterion: Optional[nn.Module] = None,
    predictors: Optional[nn.ModuleList] = None, aggregation_method: str = "sum", input_adapter: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Evaluates the CaFo model by aggregating predictor outputs.
    (Function content remains the same as provided previously)
    """
    model.eval(); model.to(device)
    if predictors is None: predictors = getattr(model, "trained_predictors", None)
    if not predictors: raise ValueError("Trained predictors are required for CaFo evaluation.")
    predictors.to(device); predictors.eval()
    num_predictors = len(predictors); num_blocks = len(model.blocks)
    if num_predictors > num_blocks: logger.warning(f"More predictors ({num_predictors}) than blocks ({num_blocks}). Using first {num_blocks}.")
    predictors_to_use = predictors[:num_blocks]
    if len(predictors_to_use) < num_blocks: logger.warning(f"Fewer predictors ({len(predictors_to_use)}) than blocks ({num_blocks}). Evaluation based on partial training?")
    total_loss, total_correct, total_samples = 0.0, 0, 0
    aggregation_method = aggregation_method.lower()
    logger.info(f"Evaluating CaFo model using {len(predictors_to_use)} predictors with '{aggregation_method}' aggregation.")
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Evaluating CaFo ({aggregation_method})", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            block_outputs = model.forward(images)
            predictor_outputs_list = []
            if aggregation_method == "last":
                 if len(block_outputs) > 0 and len(predictors_to_use) > 0:
                     last_idx = min(len(block_outputs), len(predictors_to_use)) - 1
                     try: pred_out = predictors_to_use[last_idx](block_outputs[last_idx]); predictor_outputs_list.append(pred_out)
                     except Exception as e: logger.error(f"Eval error (last): {e}"); return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
                 else: logger.error("Cannot eval 'last': no blocks/predictors."); return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
            else:
                for i, block_out in enumerate(block_outputs):
                    if i < len(predictors_to_use):
                        try: pred_out = predictors_to_use[i](block_out); predictor_outputs_list.append(pred_out)
                        except Exception as e: logger.error(f"Eval error (pred {i}): {e}"); return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
            if not predictor_outputs_list: logger.error("No predictor outputs generated."); return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
            try:
                if aggregation_method in ["sum", "last"]: final_prediction_logits = torch.stack(predictor_outputs_list, dim=0).sum(dim=0)
                elif aggregation_method == "average": final_prediction_logits = torch.stack(predictor_outputs_list, dim=0).mean(dim=0)
                else: raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
                if criterion: loss = criterion(final_prediction_logits, labels); total_loss += loss.item() * images.size(0)
                predicted_labels = torch.argmax(final_prediction_logits, dim=1); total_correct += (predicted_labels == labels).sum().item(); total_samples += labels.size(0)
            except Exception as e_agg: logger.error(f"Error during aggregation: {e_agg}"); return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
    avg_loss = total_loss / total_samples if criterion and total_samples > 0 else float('nan')
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"Evaluation Results (Aggregation: {aggregation_method}): Accuracy: {accuracy:.2f}%" + (f", Loss: {avg_loss:.4f}" if criterion and not torch.isnan(torch.tensor(avg_loss)) else ""))
    results = {"eval_accuracy": accuracy, "eval_loss": avg_loss}
    return results