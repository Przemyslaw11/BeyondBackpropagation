# File: src/algorithms/cafo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from tqdm import tqdm
from typing import Dict, Any, Optional, Callable, List, Type, Tuple # Added List, Tuple
import os
import time

from src.architectures.cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint, create_directory_if_not_exists # Added create_dir

logger = logging.getLogger(__name__)

# --- train_cafo_predictor_only (MODIFIED) ---
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
    step_ref: List[int] = [-1], # MODIFIED: Use step_ref list
) -> Tuple[float, float]: # Return avg loss, avg accuracy
    """
    Trains a single CaFoPredictor layer-wise, keeping the corresponding CaFoBlock frozen.
    MODIFIED: Accepts step_ref, logs batch metrics, returns avg loss/acc.
    """
    block.eval()
    predictor.train()
    block.to(device)
    predictor.to(device)
    logger.info(
        f"Starting CaFo training for Predictor {block_index + 1} (Block {block_index+1} frozen)"
    )

    final_avg_epoch_loss = float('nan')
    final_avg_epoch_accuracy = float('nan')

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        pbar = tqdm(
            train_loader,
            desc=f"Predictor {block_index+1} Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1 # MODIFIED: Increment global step
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

            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(train_loader) - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(
                    loss=f"{avg_loss_batch:.4f}", acc=f"{batch_accuracy:.2f}%"
                )
                # MODIFIED: Add global_step to metrics dict
                metrics_to_log = {
                    "global_step": current_global_step,
                    f"Predictor_{block_index+1}/Train_Loss_Batch": avg_loss_batch,
                    f"Predictor_{block_index+1}/Train_Acc_Batch": batch_accuracy,
                }
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True) # Pass full dict

        # Calculate averages for the epoch
        final_avg_epoch_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        final_avg_epoch_accuracy = (epoch_correct / epoch_samples) * 100.0 if epoch_samples > 0 else 0.0

        logger.info(
            f"Predictor {block_index+1} Epoch {epoch+1}/{epochs} - Avg Loss: {final_avg_epoch_loss:.4f}, Avg Acc: {final_avg_epoch_accuracy:.2f}%"
        )
        # --- REMOVED epoch summary logging from here ---

    logger.info(f"Finished CaFo training for Predictor {block_index + 1}")
    predictor.eval()
    return final_avg_epoch_loss, final_avg_epoch_accuracy


# --- train_cafo_model (MODIFIED) ---
def train_cafo_model(
    model: CaFo_CNN,  # Contains only blocks
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None, # Added for signature consistency, not used
    step_ref: List[int] = [-1], # MODIFIED: Accept step_ref
):
    """
    Orchestrates the layer-wise training of CaFo_CNN predictors (CaFo-Rand-CE variant).
    MODIFIED: Logs predictor summary metrics.
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

    for i in range(num_blocks):
        logger.info(f"--- Training Predictor {i+1}/{num_blocks} ---")
        block = model.blocks[i]
        predictor = predictors[i]
        params_to_optimize = list(predictor.parameters())

        if not any(p.requires_grad for p in params_to_optimize):
            logger.warning(f"Predictor {i+1} has no parameters requiring gradients. Skipping training.")
             # Log placeholder loss/acc
            current_global_step = step_ref[0] # Step doesn't advance
            predictor_summary_metrics = {
                "global_step": current_global_step,
                f"Predictor_{i+1}/Train_Loss_LayerAvg": float('nan'),
                f"Predictor_{i+1}/Train_Acc_LayerAvg": float('nan'),
            }
            log_metrics(predictor_summary_metrics, wandb_run=wandb_run, commit=True)
        else:
            optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_params_extra}
            optimizer = getattr(optim, optimizer_name)(params_to_optimize, **optimizer_kwargs)

            # Train the predictor
            final_avg_loss, final_avg_acc = train_cafo_predictor_only(
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
                step_ref=step_ref, # Pass step_ref
            )

            # MODIFIED: Log predictor summary after training completes
            current_global_step = step_ref[0]
            predictor_summary_metrics = {
                "global_step": current_global_step,
                f"Predictor_{i+1}/Train_Loss_LayerAvg": final_avg_loss,
                f"Predictor_{i+1}/Train_Acc_LayerAvg": final_avg_acc,
            }
            log_metrics(predictor_summary_metrics, wandb_run=wandb_run, commit=True)
            logger.debug(f"Logged CaFo Predictor {i+1} summary at global_step {current_global_step}")

            # Freeze predictor after training
            for param in predictor.parameters(): param.requires_grad = False
            predictor.eval()

            if checkpoint_dir:
                create_directory_if_not_exists(checkpoint_dir) # Ensure dir exists
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


# --- evaluate_cafo_model (no changes needed) ---
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
            # Attempt to load predictors based on config checkpoint_dir? Or raise error.
            # For now, raise error if not found.
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
            # input_adapter is ignored for standard CNN evaluation
            adapted_images = images

            block_outputs = model.forward(adapted_images)
            predictor_outputs = []
            # Only use predictors up to the number available
            for i, block_out in enumerate(block_outputs):
                if i < len(predictors):
                    try:
                        pred_out = predictors[i](block_out)
                        predictor_outputs.append(pred_out)
                    except Exception as e_pred:
                        logger.error(f"Error during predictor {i} forward pass: {e_pred}", exc_info=True)
                        # Decide how to handle: skip predictor, return NaN, etc.
                        # Returning NaN for now if any predictor fails.
                        return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}

            if not predictor_outputs:
                logger.error("No predictor outputs generated for aggregation.")
                return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}

            try:
                if aggregation_method == "sum":
                    final_prediction_logits = torch.stack(predictor_outputs, dim=0).sum(dim=0)
                elif aggregation_method == "last":
                    final_prediction_logits = predictor_outputs[-1]
                # Add other aggregation methods like 'average' if needed
                # elif aggregation_method == "average":
                #     final_prediction_logits = torch.stack(predictor_outputs, dim=0).mean(dim=0)
                else:
                    raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

                if criterion:
                    loss = criterion(final_prediction_logits, labels)
                    total_loss += loss.item() * adapted_images.size(0)

                predicted_labels = torch.argmax(final_prediction_logits, dim=1)
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)
            except Exception as e_agg:
                 logger.error(f"Error during prediction aggregation or loss calculation: {e_agg}", exc_info=True)
                 return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}


    avg_loss = total_loss / total_samples if criterion and total_samples > 0 else float("nan")
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(
        f"Evaluation Results (Aggregation: {aggregation_method}): Accuracy: {accuracy:.2f}%"
        + (f", Loss: {avg_loss:.4f}" if criterion and not torch.isnan(torch.tensor(avg_loss)) else "")
    )

    results = {"eval_accuracy": accuracy}
    if criterion and not torch.isnan(torch.tensor(avg_loss)):
        results["eval_loss"] = avg_loss
    # Ensure eval_loss is NaN if criterion is None
    elif not criterion:
        results["eval_loss"] = float("nan")

    return results