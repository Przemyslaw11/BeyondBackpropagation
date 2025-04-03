# File: src/algorithms/cafo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from tqdm import tqdm
import pynvml # Import for type hint
from typing import Dict, Any, Optional, Callable, List, Type, Tuple
import os
import time

from src.architectures.cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint, create_directory_if_not_exists
from src.utils.monitoring import get_gpu_memory_usage # Import memory usage function

logger = logging.getLogger(__name__)

# --- train_cafo_predictor_only (MODIFIED - Logging prefixes) ---
def train_cafo_predictor_only(
    block: CaFoBlock,
    predictor: CaFoPredictor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    get_block_input_fn: Callable[[torch.Tensor], torch.Tensor],
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    block_index: int = 0,  # For logging
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, # ADDED
    nvml_active: bool = False, # ADDED
) -> Tuple[float, float, float]: # Return avg loss, avg accuracy, peak memory
    """
    Trains a single CaFoPredictor layer-wise, keeping the corresponding CaFoBlock frozen.
    MODIFIED: Accepts handle/active, samples memory, returns avg loss/acc and peak memory.
    MODIFIED: Updated logging prefixes for consistency.
    """
    block.eval()
    predictor.train()
    block.to(device)
    predictor.to(device)
    # <<< MODIFIED: Use consistent log prefix >>>
    log_prefix = f"Predictor_{block_index+1}"
    logger.info(
        f"Starting CaFo training for {log_prefix} (Block {block_index+1} frozen)"
    )

    peak_mem_predictor_train = 0.0 # Track peak memory for this predictor's training
    final_avg_epoch_loss = float('nan')
    final_avg_epoch_accuracy = float('nan')

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        peak_mem_predictor_epoch = 0.0 # Track peak memory for this specific epoch
        pbar = tqdm(
            train_loader,
            desc=f"{log_prefix} Epoch {epoch+1}/{epochs}", # Use prefix in tqdm
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1
            current_global_step = step_ref[0]

            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                block_input = get_block_input_fn(images)
                block_output = block(block_input)

            predictions = predictor(block_output.detach())
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy for the batch
            batch_size = labels.size(0)
            with torch.no_grad():
                 pred_labels = torch.argmax(predictions, dim=1)
                 batch_correct = (pred_labels == labels).sum().item()
            batch_accuracy = (batch_correct / batch_size) * 100.0 if batch_size > 0 else 0.0

            epoch_loss += loss.item() * batch_size # Weighted by batch size for avg calc
            epoch_correct += batch_correct
            epoch_samples += batch_size

            # --- Sample memory usage periodically or at end of batch ---
            current_mem_used = float('nan')
            if nvml_active and gpu_handle and ((batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1):
                 mem_info = get_gpu_memory_usage(gpu_handle)
                 if mem_info:
                     current_mem_used = mem_info[0]
                     peak_mem_predictor_epoch = max(peak_mem_predictor_epoch, current_mem_used)
            # --- End memory sampling ---

            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(
                    loss=f"{avg_loss_batch:.4f}", acc=f"{batch_accuracy:.2f}%"
                )
                # <<< MODIFIED: Use consistent log prefix >>>
                metrics_to_log = {
                    "global_step": current_global_step,
                    f"{log_prefix}/Train_Loss_Batch": avg_loss_batch,
                    f"{log_prefix}/Train_Acc_Batch": batch_accuracy,
                }
                if not torch.isnan(torch.tensor(current_mem_used)):
                     metrics_to_log[f"{log_prefix}/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)

        # Calculate averages for the epoch
        final_avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        final_avg_epoch_accuracy = (epoch_correct / epoch_samples) * 100.0 if epoch_samples > 0 else 0.0
        peak_mem_predictor_train = max(peak_mem_predictor_train, peak_mem_predictor_epoch) # Update peak for the predictor

        logger.info(
            # <<< MODIFIED: Use consistent log prefix >>>
            f"{log_prefix} Epoch {epoch+1}/{epochs} - Avg Loss: {final_avg_epoch_loss:.4f}, Avg Acc: {final_avg_epoch_accuracy:.2f}%, Peak Mem Epoch: {peak_mem_predictor_epoch:.1f} MiB"
        )
        # Log epoch summary metrics (optional)
        epoch_summary_metrics = {
            "global_step": current_global_step, # Use last step
            f"{log_prefix}/Train_Loss_EpochAvg": final_avg_epoch_loss,
            f"{log_prefix}/Train_Acc_EpochAvg": final_avg_epoch_accuracy,
            f"{log_prefix}/Peak_GPU_Mem_Epoch_MiB": peak_mem_predictor_epoch,
        }
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)

    # Sample memory one last time
    if nvml_active and gpu_handle:
        mem_info = get_gpu_memory_usage(gpu_handle)
        if mem_info:
            peak_mem_predictor_train = max(peak_mem_predictor_train, mem_info[0])

    # <<< MODIFIED: Use consistent log prefix >>>
    logger.info(f"Finished CaFo training for {log_prefix}. Overall Peak Mem: {peak_mem_predictor_train:.1f} MiB")
    predictor.eval()
    return final_avg_epoch_loss, final_avg_epoch_accuracy, peak_mem_predictor_train


# --- train_cafo_model (MODIFIED - Logging prefixes) ---
def train_cafo_model(
    model: CaFo_CNN,  # Contains only blocks
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None, # Added for signature consistency, not used
    step_ref: List[int] = [-1],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None, # ADDED
    nvml_active: bool = False, # ADDED
) -> float: # Returns overall peak memory
    """
    Orchestrates the layer-wise training of CaFo_CNN predictors (CaFo-Rand-CE variant).
    Logs predictor summary metrics including peak memory.
    MODIFIED: Updated logging prefixes for consistency.
    Returns the overall peak GPU memory observed across all predictor training phases.
    """
    model.to(device)
    model.eval()  # Blocks are frozen
    for param in model.blocks.parameters():
        param.requires_grad = False

    num_blocks = len(model.blocks)
    logger.info(
        f"Starting layer-wise CaFo predictor training for {num_blocks} blocks (CaFo-Rand-CE)."
    )
    if input_adapter is not None:
        logger.warning("CaFo Training: 'input_adapter' provided but typically unused for CNNs.")

    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    lr = algo_config.get("lr", 0.001)
    weight_decay = algo_config.get("weight_decay", 0.0)
    criterion_name = algo_config.get("loss_type", "CrossEntropyLoss")
    epochs_per_block = algo_config.get("num_epochs_per_block", 10)
    log_interval = algo_config.get("log_interval", 100)
    optimizer_params_extra = algo_config.get("optimizer_params", {})
    checkpoint_dir = config.get("checkpointing", {}).get(
        "checkpoint_dir", None
    )

    if criterion_name.lower() == "crossentropyloss":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion for CaFo: {criterion_name}")

    # --- Create Predictors Externally ---
    predictors = nn.ModuleList()
    for i in range(num_blocks):
        try:
            in_features = model.get_predictor_input_dim(i)
            predictor = CaFoPredictor(in_features, model.num_classes).to(device)
            predictors.append(predictor)
            logger.info(f"Created predictor {i+1} with input dim {in_features}")
        except Exception as e:
            logger.error(f"Failed to create predictor for block {i}: {e}", exc_info=True)
            raise RuntimeError("Predictor creation failed.") from e

    # --- Train Predictor by Predictor ---
    # Initial input is the raw image batch
    current_block_input_fn = lambda img: img.to(device)
    peak_mem_train = 0.0 # Track overall peak memory

    for i in range(num_blocks):
        predictor_log_idx = i + 1
        # <<< MODIFIED: Use consistent log prefix >>>
        log_prefix = f"Predictor_{predictor_log_idx}"
        logger.info(f"--- Training {log_prefix} ---")
        block = model.blocks[i]
        predictor = predictors[i]
        params_to_optimize = list(predictor.parameters())
        predictor_peak_mem = 0.0
        final_avg_loss = float('nan')
        final_avg_acc = float('nan')

        if not any(p.requires_grad for p in params_to_optimize):
            logger.warning(f"{log_prefix} has no parameters requiring gradients. Skipping training.")
        else:
            optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_params_extra}
            optimizer = getattr(optim, optimizer_name)(params_to_optimize, **optimizer_kwargs)

            # Train the predictor, get loss, acc, and peak mem for this predictor
            final_avg_loss, final_avg_acc, predictor_peak_mem = train_cafo_predictor_only(
                block=block,
                predictor=predictor,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                epochs=epochs_per_block,
                device=device,
                get_block_input_fn=current_block_input_fn,
                wandb_run=wandb_run,
                log_interval=log_interval,
                block_index=i,
                step_ref=step_ref,
                gpu_handle=gpu_handle, # Pass handle
                nvml_active=nvml_active # Pass status
            )
            peak_mem_train = max(peak_mem_train, predictor_peak_mem) # Update overall peak

            # Freeze predictor after training
            for param in predictor.parameters(): param.requires_grad = False
            predictor.eval()

        # Log predictor summary after training completes
        current_global_step = step_ref[0]
        # <<< MODIFIED: Use consistent log prefix >>>
        predictor_summary_metrics = {
            "global_step": current_global_step,
            f"{log_prefix}/Train_Loss_LayerAvg": final_avg_loss,
            f"{log_prefix}/Train_Acc_LayerAvg": final_avg_acc,
            f"{log_prefix}/Peak_GPU_Mem_Layer_MiB": predictor_peak_mem, # Log predictor peak
        }
        log_metrics(predictor_summary_metrics, wandb_run=wandb_run, commit=True)
        logger.debug(f"Logged CaFo {log_prefix} summary at global_step {current_global_step}")

        if checkpoint_dir and any(p.requires_grad for p in params_to_optimize): # Only save if trained
            create_directory_if_not_exists(checkpoint_dir)
            save_checkpoint(
                state={"state_dict": predictor.state_dict(), "predictor_index": i},
                is_best=False, filename=f"cafo_predictor_{i}_complete.pth", checkpoint_dir=checkpoint_dir,
            )

        # Prepare the input function for the *next* predictor training stage
        def create_next_input_fn(trained_block_idx: int, previous_input_fn: Callable):
            block_k = model.blocks[trained_block_idx]
            block_k.eval() # Ensure block is in eval mode
            @torch.no_grad()
            def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                block_input = previous_input_fn(img_batch)
                block_output = block_k(block_input)
                return block_output.detach() # Detach output
            return next_input_fn

        # Update the input function for the next iteration
        current_block_input_fn = create_next_input_fn(i, current_block_input_fn)

    logger.info("Finished all layer-wise CaFo predictor training.")
    # Store trained predictors on the model instance for evaluation
    model.trained_predictors = predictors
    return peak_mem_train # Return overall peak memory


# --- evaluate_cafo_model (MODIFIED - Optimized "last" aggregation, added comment) ---
def evaluate_cafo_model(
    model: CaFo_CNN,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    predictors: Optional[nn.ModuleList] = None,
    aggregation_method: str = "sum",
    input_adapter: Optional[Callable] = None, # Keep signature consistent
) -> Dict[str, float]:
    """
    Evaluates the CaFo model by aggregating predictor outputs.
    MODIFIED: Optimize forward pass if aggregation_method is 'last'.
    """
    model.eval()
    model.to(device)

    if predictors is None:
        if hasattr(model, "trained_predictors") and isinstance(
            model.trained_predictors, nn.ModuleList
        ):
            predictors = model.trained_predictors
        else:
            logger.warning("Predictors not explicitly provided and not found attached to model. Trying to load from checkpoint logic might be needed.")
            raise ValueError("Trained predictors are required for CaFo evaluation.")

    if not predictors:
        raise ValueError("Predictor list is empty or None.")

    predictors.to(device)
    predictors.eval()

    num_predictors = len(predictors)
    num_blocks = len(model.blocks)
    if num_predictors != num_blocks:
        logger.warning(f"Number of predictors ({num_predictors}) does not match number of blocks ({num_blocks}). Evaluation might be incomplete.")

    total_loss, total_correct, total_samples = 0.0, 0, 0
    logger.info(
        f"Evaluating CaFo model using {num_predictors} predictors with '{aggregation_method}' aggregation."
    )

    with torch.no_grad():
        pbar = tqdm(
            data_loader, desc=f"Evaluating CaFo ({aggregation_method})", leave=False
        )
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            adapted_images = images # input_adapter is ignored

            # Get intermediate block outputs from the main model
            block_outputs = model.forward(adapted_images)
            predictor_outputs = []

            # Apply predictors to corresponding block outputs
            # <<< MODIFIED: Optimization for "last" aggregation >>>
            if aggregation_method.lower() == "last":
                if num_blocks > 0 and len(block_outputs) == num_blocks and num_predictors == num_blocks:
                    last_block_idx = num_blocks - 1
                    try:
                        pred_out = predictors[last_block_idx](block_outputs[last_block_idx])
                        predictor_outputs.append(pred_out) # List will contain only the last predictor's output
                    except Exception as e_pred:
                        logger.error(f"Error during LAST predictor ({last_block_idx}) forward pass: {e_pred}", exc_info=True)
                        # Returning NaN if the single required predictor fails.
                        return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
                else:
                     logger.error(f"Cannot evaluate with 'last' method: mismatch in block/predictor counts or block outputs.")
                     return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
            else: # Original logic for "sum" or other methods needing all outputs
                for i, block_out in enumerate(block_outputs):
                    if i < len(predictors):
                        try:
                            pred_out = predictors[i](block_out)
                            predictor_outputs.append(pred_out)
                        except Exception as e_pred:
                            logger.error(f"Error during predictor {i} forward pass: {e_pred}", exc_info=True)
                            # Returning NaN if any predictor fails (as before).
                            # Note: Downstream analysis should handle NaN appropriately.
                            return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
            # <<< END MODIFICATION >>>

            if not predictor_outputs:
                logger.error("No predictor outputs generated for aggregation.")
                return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}

            try:
                # Aggregate predictions based on the method
                if aggregation_method.lower() == "sum":
                    final_prediction_logits = torch.stack(predictor_outputs, dim=0).sum(dim=0)
                elif aggregation_method.lower() == "last":
                    # predictor_outputs should contain only the last one due to optimization above
                    final_prediction_logits = predictor_outputs[0]
                elif aggregation_method.lower() == "average": # Added average as an option
                     final_prediction_logits = torch.stack(predictor_outputs, dim=0).mean(dim=0)
                else:
                    raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

                # Calculate loss if criterion is provided
                if criterion:
                    loss = criterion(final_prediction_logits, labels)
                    total_loss += loss.item() * adapted_images.size(0)

                # Calculate accuracy
                predicted_labels = torch.argmax(final_prediction_logits, dim=1)
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)

            except Exception as e_agg:
                 logger.error(f"Error during prediction aggregation or loss calculation: {e_agg}", exc_info=True)
                 # Returning NaN if aggregation fails
                 return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}


    avg_loss = total_loss / total_samples if criterion and total_samples > 0 else float("nan")
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(
        f"Evaluation Results (Aggregation: {aggregation_method}): Accuracy: {accuracy:.2f}%"
        + (f", Loss: {avg_loss:.4f}" if criterion and not torch.isnan(torch.tensor(avg_loss)) else "")
    )

    results = {"eval_accuracy": accuracy}
    # Ensure eval_loss is NaN if criterion is None or if loss calculation failed
    results["eval_loss"] = avg_loss # Will be NaN if criterion was None or samples=0

    return results
