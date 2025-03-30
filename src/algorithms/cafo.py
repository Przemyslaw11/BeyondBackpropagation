# File: src/algorithms/cafo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from tqdm import tqdm
from typing import Dict, Any, Optional, Callable, List, Type
import os  # For checkpointing

from src.architectures.cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint  # Import checkpoint helper

logger = logging.getLogger(__name__)


# --- Modified for CaFo-Rand-CE variant ---
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
    block_index: int = 0,  # For logging and W&B step offset
) -> None:
    """
    Trains a single CaFoPredictor layer-wise, keeping the corresponding CaFoBlock frozen.
    """
    block.eval()
    predictor.train()
    block.to(device)
    predictor.to(device)
    logger.info(
        f"Starting CaFo training for Predictor {block_index + 1} (Block {block_index+1} frozen)"
    )

    total_steps_per_epoch = len(train_loader)
    global_step_offset = block_index * epochs * total_steps_per_epoch

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Predictor {block_index+1} Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                block_input = get_block_input_fn(images)
                block_output = block(block_input)

            predictions = predictor(block_output.detach())
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_accuracy = calculate_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_accuracy += batch_accuracy
            current_global_step = (
                global_step_offset + epoch * total_steps_per_epoch + batch_idx
            )

            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(
                train_loader
            ) - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(
                    loss=f"{avg_loss_batch:.4f}", acc=f"{batch_accuracy:.2f}%"
                )
                # Use consistent naming for W&B
                metrics = {
                    f"Predictor_{block_index+1}/Train_Loss_Batch": avg_loss_batch,
                    f"Predictor_{block_index+1}/Train_Acc_Batch": batch_accuracy,
                }
                log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_accuracy = epoch_accuracy / len(train_loader)
        logger.info(
            f"Predictor {block_index+1} Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_accuracy:.2f}%"
        )
        log_metrics(
            {
                f"Predictor_{block_index+1}/Train_Loss_Epoch": avg_epoch_loss,
                f"Predictor_{block_index+1}/Train_Acc_Epoch": avg_epoch_accuracy,
            },
            step=global_step_offset + (epoch + 1) * total_steps_per_epoch,
            wandb_run=wandb_run,
        )

    logger.info(f"Finished CaFo training for Predictor {block_index + 1}")
    predictor.eval()


def train_cafo_model(
    model: CaFo_CNN,  # Contains only blocks
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
):
    """
    Orchestrates the layer-wise training of CaFo_CNN predictors (CaFo-Rand-CE variant).
    """
    model.to(device)
    model.eval()  # Blocks are frozen
    for param in model.blocks.parameters():
        param.requires_grad = False

    num_blocks = len(model.blocks)
    logger.info(
        f"Starting layer-wise CaFo predictor training for {num_blocks} blocks (CaFo-Rand-CE)."
    )

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
    )  # For saving predictors

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
            logger.error(
                f"Failed to create predictor for block {i}: {e}", exc_info=True
            )
            raise RuntimeError("Predictor creation failed.") from e

    # --- Train Predictor by Predictor ---
    current_block_input_fn = lambda img: img.to(
        device
    )  # Ensure input is on correct device

    for i in range(num_blocks):
        logger.info(f"--- Training Predictor {i+1}/{num_blocks} ---")
        block = model.blocks[i]
        predictor = predictors[i]
        params_to_optimize = list(predictor.parameters())

        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(
                params_to_optimize,
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_params_extra,
            )
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(
                params_to_optimize,
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_params_extra,
            )
        elif optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(
                params_to_optimize,
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_params_extra,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        train_cafo_predictor_only(
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
        )

        # Freeze predictor after training (optional but good practice)
        for param in predictor.parameters():
            param.requires_grad = False
        predictor.eval()

        # --- Checkpointing after each predictor ---
        if checkpoint_dir:
            chkpt_filename = f"cafo_predictor_{i}.pth"  # Index 0 to N-1
            save_checkpoint(
                state={"state_dict": predictor.state_dict(), "predictor_index": i},
                is_best=False,  # Just save sequentially
                filename=chkpt_filename,
                checkpoint_dir=checkpoint_dir,
            )

        # Update the input function for the *next* predictor
        def create_next_input_fn(trained_block_idx: int, previous_input_fn: Callable):
            block_k = model.blocks[trained_block_idx]

            @torch.no_grad()
            def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                block_input = previous_input_fn(img_batch)
                block_output = block_k(block_input)
                return block_output

            return next_input_fn

        current_block_input_fn = create_next_input_fn(i, current_block_input_fn)

    logger.info("Finished all layer-wise CaFo predictor training.")
    model.trained_predictors = predictors  # Attach trained predictors for evaluation


def evaluate_cafo_model(
    model: CaFo_CNN,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    predictors: Optional[nn.ModuleList] = None,
    aggregation_method: str = "sum",  # Default to sum
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
            raise ValueError(
                "Trained predictors not provided and not found attached to the model."
            )
    if not predictors:
        raise ValueError("Predictor list is empty.")

    predictors.to(device)
    predictors.eval()

    num_predictors = len(predictors)
    num_blocks = len(model.blocks)
    if num_predictors != num_blocks:
        logger.warning(
            f"Number of predictors ({num_predictors}) != number of blocks ({num_blocks})."
        )

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
            block_outputs = model.forward(images)  # List of [B, Ck, Hk, Wk]
            predictor_outputs = []
            for i, block_out in enumerate(block_outputs):
                if i < len(predictors):
                    pred_out = predictors[i](block_out)  # Logits [B, num_classes]
                    predictor_outputs.append(pred_out)
                # else: logger.warning(f"Missing predictor for block {i+1}, skipping.") # Logged above

            if not predictor_outputs:
                logger.error("No predictor outputs generated.")
                return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}

            if aggregation_method == "sum":
                final_prediction_logits = torch.stack(predictor_outputs, dim=0).sum(
                    dim=0
                )
            elif aggregation_method == "last":
                final_prediction_logits = predictor_outputs[-1]
            else:
                raise ValueError(
                    f"Unsupported aggregation method: {aggregation_method}"
                )

            if criterion:
                loss = criterion(final_prediction_logits, labels)
                total_loss += loss.item() * images.size(0)

            predicted_labels = torch.argmax(final_prediction_logits, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = (
        total_loss / total_samples if criterion and total_samples > 0 else float("nan")
    )
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(
        f"Evaluation Results (Aggregation: {aggregation_method}): Accuracy: {accuracy:.2f}%"
        + (
            f", Loss: {avg_loss:.4f}"
            if criterion and not torch.isnan(torch.tensor(avg_loss))
            else ""
        )
    )

    results = {"eval_accuracy": accuracy}
    if criterion and not torch.isnan(torch.tensor(avg_loss)):
        results["eval_loss"] = avg_loss
    return results
