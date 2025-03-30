# File: src/algorithms/cafo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from tqdm import tqdm
from typing import Dict, Any, Optional, Callable, List, Type  # Added Type

from src.architectures.cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics

logger = logging.getLogger(__name__)


# --- Modified for CaFo-Rand-CE variant ---
def train_cafo_predictor_only(
    block: CaFoBlock,  # Block is FROZEN
    predictor: CaFoPredictor,  # Predictor is TRAINED
    optimizer: optim.Optimizer,  # Optimizer for PREDICTOR ONLY
    criterion: nn.Module,  # Loss function (e.g., CrossEntropyLoss)
    train_loader: DataLoader,  # Loader providing original (images, labels)
    epochs: int,
    device: torch.device,
    get_block_input_fn: Callable[
        [torch.Tensor], torch.Tensor
    ],  # Function to get input for the current block
    wandb_run: Optional[Any] = None,  # For logging
    log_interval: int = 100,
    block_index: int = 0,  # For logging purposes
) -> None:
    """
    Trains a single CaFoPredictor layer-wise, keeping the corresponding CaFoBlock frozen.
    This aligns with the CaFo-Rand-CE strategy.

    Args:
        block: The CaFoBlock module (used in eval mode to generate features).
        predictor: The CaFoPredictor module to train.
        optimizer: PyTorch optimizer for the parameters of the PREDICTOR only.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        train_loader: DataLoader providing batches of original (images, labels).
        epochs: Number of training epochs for this predictor.
        device: Device to perform training on ('cuda' or 'cpu').
        get_block_input_fn: A function that takes a batch of original images
                            and returns the input tensor for the *current* block
                            (output of previous frozen blocks, or original image).
        wandb_run: Optional Weights & Biases run object for logging.
        log_interval: How often to log batch metrics.
        block_index: Index of the current block (for clearer logging).
    """
    # --- Set modes: Block is EVAL, Predictor is TRAIN ---
    block.eval()
    predictor.train()
    block.to(device)
    predictor.to(device)
    logger.info(
        f"Starting CaFo training for Predictor {block_index + 1} (Block {block_index+1} frozen)"
    )

    global_step_offset = (
        block_index * epochs * len(train_loader)
    )  # Global step offset for W&B

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

            # 1. Get input for the current block (detached via no_grad)
            with torch.no_grad():
                block_input = get_block_input_fn(images)

            # 2. Forward pass through FROZEN block
            with torch.no_grad():
                block_output = block(block_input)  # No gradients needed for block

            # 3. Forward pass through predictor (gradients ARE needed)
            predictions = predictor(
                block_output.detach()
            )  # Detach input just in case, though block is no_grad

            # 4. Calculate loss
            loss = criterion(predictions, labels)

            # 5. Backpropagate and optimize (only for the predictor)
            optimizer.zero_grad()
            loss.backward()  # Computes gradients ONLY for predictor parameters
            optimizer.step()  # Updates ONLY predictor parameters

            # Calculate batch accuracy
            batch_accuracy = calculate_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_accuracy += batch_accuracy
            current_global_step = (
                global_step_offset + epoch * len(train_loader) + batch_idx
            )

            # Logging
            if (batch_idx + 1) % log_interval == 0 or batch_idx == len(
                train_loader
            ) - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(
                    loss=f"{avg_loss_batch:.4f}", acc=f"{batch_accuracy:.2f}%"
                )
                metrics = {
                    f"predictor_{block_index+1}_train_loss_batch": avg_loss_batch,
                    f"predictor_{block_index+1}_train_acc_batch": batch_accuracy,
                }
                log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_accuracy = epoch_accuracy / len(train_loader)
        logger.info(
            f"Predictor {block_index+1} Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_accuracy:.2f}%"
        )
        log_metrics(
            {
                f"predictor_{block_index+1}_train_loss_epoch": avg_epoch_loss,
                f"predictor_{block_index+1}_train_acc_epoch": avg_epoch_accuracy,
            },
            step=global_step_offset + (epoch + 1) * len(train_loader),
            wandb_run=wandb_run,
        )  # Log at end of epoch step

    logger.info(f"Finished CaFo training for Predictor {block_index + 1}")
    predictor.eval()  # Set predictor to eval mode after training


def train_cafo_model(
    model: CaFo_CNN,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
):
    """
    Orchestrates the layer-wise training of CaFo_CNN predictors (CaFo-Rand-CE variant).
    Assumes blocks are randomly initialized and frozen.
    """
    model.to(device)
    # --- Ensure blocks are frozen and in eval mode ---
    model.eval()
    for param in model.blocks.parameters():
        param.requires_grad = False

    num_blocks = len(model.blocks)
    logger.info(
        f"Starting layer-wise CaFo predictor training for {num_blocks} blocks (CaFo-Rand-CE)."
    )

    # --- Get Training Configuration ---
    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_name = algo_config.get(
        "optimizer_type", "Adam"
    )  # Changed key to match FF/MF
    lr = algo_config.get("lr", 0.001)
    weight_decay = algo_config.get("weight_decay", 0.0)
    criterion_name = algo_config.get("loss_type", "CrossEntropyLoss")  # Changed key
    epochs_per_block = algo_config.get("num_epochs_per_block", 10)  # Changed key
    log_interval = algo_config.get("log_interval", 100)
    optimizer_params_extra = algo_config.get("optimizer_params", {})

    # Select loss function
    if criterion_name.lower() == "crossentropyloss":
        criterion = nn.CrossEntropyLoss()
    # Add MSE/SL if implementing those CaFo variants
    else:
        raise ValueError(f"Unsupported criterion for CaFo: {criterion_name}")

    # --- Create Predictors Externally ---
    # Predictors are not part of CaFo_CNN model structure now
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
    current_block_input_fn = lambda img: img  # Input for the first block

    for i in range(num_blocks):
        logger.info(f"--- Training Predictor {i+1}/{num_blocks} ---")
        block = model.blocks[i]  # This block is frozen
        predictor = predictors[i]  # This predictor will be trained

        # Parameters to optimize: current predictor ONLY
        params_to_optimize = list(predictor.parameters())

        # Select optimizer for the predictor
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

        # Train the current predictor only
        train_cafo_predictor_only(  # Call the corrected function
            block=block,  # Frozen block
            predictor=predictor,  # Trainable predictor
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            epochs=epochs_per_block,
            device=device,
            get_block_input_fn=current_block_input_fn,  # Provides input for the block
            wandb_run=wandb_run,
            log_interval=log_interval,
            block_index=i,
        )

        # Freeze the trained predictor (optional, but good practice if not retraining later)
        # predictor.requires_grad_(False) # Freeze whole module
        for param in predictor.parameters():
            param.requires_grad = False
        predictor.eval()

        # Update the input function for the *next* block/predictor training stage
        def create_next_input_fn(trained_block_idx: int):
            # trained_block_idx is the index of the block whose features were just used (0-based)
            block_k = model.blocks[trained_block_idx]  # Get the block instance
            # Need the input function that yielded input for *this* block
            prev_input_fn = current_block_input_fn

            @torch.no_grad()  # Ensure computation is gradient-free
            def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                # Get input for block k
                block_input = prev_input_fn(img_batch)
                # Compute output of block k
                block_output = block_k(block_input)
                return block_output

            return next_input_fn

        current_block_input_fn = create_next_input_fn(i)

    logger.info("Finished all layer-wise CaFo predictor training.")
    # Store trained predictors with the model if needed for evaluation
    # This depends on how evaluate_cafo_model is called by the engine
    model.trained_predictors = predictors  # Attach trained predictors


def evaluate_cafo_model(
    model: CaFo_CNN,  # Base model containing blocks
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,  # Optional: if loss calculation is needed
    # Predictors must be passed in or attached to model
    predictors: Optional[nn.ModuleList] = None,
    aggregation_method: str = "sum",  # 'sum' or 'last'
) -> Dict[str, float]:
    """
    Evaluates the CaFo model by aggregating predictor outputs.

    Args:
        model: The CaFo_CNN model instance containing the blocks.
        data_loader: DataLoader for the evaluation dataset.
        device: Device to run evaluation on.
        criterion: Optional loss function for calculating evaluation loss (on final prediction).
        predictors: The trained ModuleList of CaFoPredictors. If None, tries `model.trained_predictors`.
        aggregation_method: How to combine predictor outputs ('sum' or 'last').

    Returns:
        Dictionary containing evaluation metrics (e.g., 'eval_loss', 'eval_accuracy').
    """
    model.eval()  # Ensure blocks are in eval mode
    model.to(device)

    # Get predictors
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
    predictors.eval()  # Ensure predictors are in eval mode

    num_predictors = len(predictors)
    num_blocks = len(model.blocks)
    if num_predictors != num_blocks:
        logger.warning(
            f"Number of predictors ({num_predictors}) does not match number of blocks ({num_blocks}). Aggregation might be incorrect."
        )

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    logger.info(
        f"Evaluating CaFo model using {num_predictors} predictors with '{aggregation_method}' aggregation."
    )

    with torch.no_grad():
        pbar = tqdm(
            data_loader, desc=f"Evaluating CaFo ({aggregation_method})", leave=False
        )
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Get all block outputs
            block_outputs = model.forward(images)  # List of [B, Ck, Hk, Wk]

            # Get predictions from each predictor
            predictor_outputs = []
            for i, block_out in enumerate(block_outputs):
                if i < len(predictors):
                    pred_out = predictors[i](block_out)  # Logits [B, num_classes]
                    predictor_outputs.append(pred_out)
                else:
                    logger.warning(f"Missing predictor for block {i+1}, skipping.")

            if not predictor_outputs:
                logger.error("No predictor outputs generated. Cannot evaluate.")
                # Handle error case, maybe return NaN or raise
                return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}

            # Aggregate predictions
            if aggregation_method == "sum":
                # Sum logits (or could sum probabilities after softmax)
                final_prediction_logits = torch.stack(predictor_outputs, dim=0).sum(
                    dim=0
                )
            elif aggregation_method == "last":
                final_prediction_logits = predictor_outputs[-1]
            else:
                raise ValueError(
                    f"Unsupported aggregation method: {aggregation_method}"
                )

            # Calculate loss on final aggregated prediction if criterion is provided
            if criterion:
                loss = criterion(final_prediction_logits, labels)
                total_loss += loss.item() * images.size(0)

            # Calculate accuracy based on final aggregated prediction
            predicted_labels = torch.argmax(final_prediction_logits, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            # Update progress bar if desired
            # current_acc = (total_correct / total_samples * 100.0 if total_samples > 0 else 0.0)
            # pbar.set_postfix(acc=f"{current_acc:.2f}%")

    avg_loss = (
        total_loss / total_samples if criterion and total_samples > 0 else float("nan")
    )
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

    logger.info(
        f"Evaluation Results (Aggregation: {aggregation_method}): Accuracy: {accuracy:.2f}%"
        + (f", Loss: {avg_loss:.4f}" if criterion else "")
    )

    results = {"eval_accuracy": accuracy}
    if criterion:
        results["eval_loss"] = avg_loss

    return results


# Removed the __main__ block for cleaner algorithm file
