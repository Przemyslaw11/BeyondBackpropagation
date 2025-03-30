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

from src.architectures.ff_mlp import FF_MLP, FF_Layer
from src.utils.logging_utils import log_metrics

logger = logging.getLogger(__name__)


def ff_loss_fn(
    pos_goodness: torch.Tensor, neg_goodness: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Calculates the Forward-Forward loss for a layer using logistic loss (softplus).
    Loss = softplus(-(pos_goodness - threshold)) + softplus(neg_goodness - threshold)
    """
    # log(1 + exp(-x)) -> softplus(-x)
    loss_pos = F.softplus(-(pos_goodness - threshold))
    # log(1 + exp(x)) -> softplus(x)
    loss_neg = F.softplus(neg_goodness - threshold)
    loss = torch.mean(loss_pos + loss_neg)
    return loss


def train_ff_layer(
    layer: FF_Layer,
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
) -> None:
    """Trains a single FF_Layer."""
    layer.train()
    layer.to(device)
    logger.info(f"Starting FF training for Layer {layer_index + 1}")

    total_steps_per_epoch = len(train_loader)
    global_step_offset = (
        layer_index * epochs * total_steps_per_epoch
    )  # Offset for global W&B step

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"FF Layer {layer_index+1} Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # 1. Get detached positive/negative input for the current layer
            pos_input, neg_input = get_layer_input_fn(images, labels)

            # 2. Forward pass through current layer to get goodness
            _, pos_goodness = layer.forward_with_goodness(pos_input)
            _, neg_goodness = layer.forward_with_goodness(neg_input)

            # 3. Calculate loss
            loss = ff_loss_fn(pos_goodness, neg_goodness, threshold)

            # 4. Backpropagate and optimize (only for the current layer)
            optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping if needed
            # torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_step = global_step_offset + epoch * total_steps_per_epoch + batch_idx

            # Logging
            if (
                batch_idx + 1
            ) % log_interval == 0 or batch_idx == total_steps_per_epoch - 1:
                avg_loss_batch = loss.item()  # Loss for this specific batch
                pbar.set_postfix(loss=f"{avg_loss_batch:.4f}")
                metrics = {
                    f"layer_{layer_index+1}_train_loss_batch": avg_loss_batch,
                    f"layer_{layer_index+1}_pos_goodness_mean": pos_goodness.mean().item(),
                    f"layer_{layer_index+1}_neg_goodness_mean": neg_goodness.mean().item(),
                }
                # Log against global step across all layers
                log_metrics(metrics, step=batch_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / total_steps_per_epoch
        logger.info(
            f"FF Layer {layer_index+1} Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}"
        )
        # Log epoch loss against global epoch number *within this layer's training*
        log_metrics(
            {f"layer_{layer_index+1}_train_loss_epoch": avg_epoch_loss},
            step=global_step_offset
            + (epoch + 1) * total_steps_per_epoch,  # Log at end of epoch step
            wandb_run=wandb_run,
        )

    logger.info(f"Finished FF training for Layer {layer_index + 1}")
    layer.eval()  # Set layer to eval mode after training


def train_ff_model(
    model: FF_MLP,
    train_loader: DataLoader,
    config: Dict[
        str, Any
    ],  # Contains optimizer params, threshold, epochs per layer etc.
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None,  # e.g., flatten
):
    """Orchestrates the layer-wise training of an FF_MLP model."""
    model.to(device)
    num_layers = len(model.layers)
    logger.info(f"Starting layer-wise FF training for {num_layers} layers.")

    # --- Get Training Configuration ---
    # Use 'algorithm_params' section from config if available, else 'training'
    algo_config = config.get("algorithm_params", config.get("training", {}))

    optimizer_type = algo_config.get("optimizer_type", "Adam")
    lr = algo_config.get("lr", 0.001)
    weight_decay = algo_config.get("weight_decay", 0.0)
    threshold = algo_config.get("threshold", 1.0)
    epochs_per_layer = algo_config.get("epochs_per_layer", 10)
    log_interval = algo_config.get("log_interval", 100)
    optimizer_params_extra = algo_config.get(
        "optimizer_params", {}
    )  # E.g. momentum for SGD

    # --- Helper function to generate pos/neg data ---
    @torch.no_grad()  # Ensure no gradients within this helper
    def generate_pos_neg_for_ff(base_images, base_labels):
        batch_size = base_images.shape[0]
        num_classes = model.num_classes
        device = base_images.device

        # Positive Data (apply adapter if needed before embedding)
        adapted_images = input_adapter(base_images) if input_adapter else base_images
        pos_data = model.prepare_ff_input(adapted_images, base_labels)

        # Negative Data
        rand_offset = torch.randint(
            1, num_classes, (batch_size,), device=device, dtype=torch.long
        )
        neg_labels = (base_labels + rand_offset) % num_classes
        # Simple check for collisions (might need retry loop for small num_classes)
        collision = neg_labels == base_labels
        if torch.any(collision):
            neg_labels[collision] = (neg_labels[collision] + 1) % num_classes
        neg_data = model.prepare_ff_input(adapted_images, neg_labels)

        return pos_data, neg_data

    # --- Train Layer by Layer ---
    for i, layer in enumerate(model.layers):  # layer is FF_Layer instance
        logger.info(f"--- Training Layer {i+1}/{num_layers} ---")

        # Define the function to get input for the *current* layer 'i'
        def get_layer_i_input_closure(current_layer_idx: int):
            def get_layer_i_input(images, labels):
                # Generate base pos/neg data (adapter applied inside)
                base_pos_data, base_neg_data = generate_pos_neg_for_ff(images, labels)

                if current_layer_idx == 0:
                    # Input to first layer is the prepared data
                    return base_pos_data.detach(), base_neg_data.detach()
                else:
                    # For subsequent layers, pass base data through frozen layers 0 to i-1
                    with torch.no_grad():
                        pos_input_i = model.forward_upto(
                            base_pos_data, current_layer_idx
                        )
                        neg_input_i = model.forward_upto(
                            base_neg_data, current_layer_idx
                        )
                    return pos_input_i.detach(), neg_input_i.detach()

            return get_layer_i_input

        get_layer_input_fn = get_layer_i_input_closure(i)

        # Create optimizer for the current layer *only*
        params_to_optimize = list(layer.parameters())
        if not params_to_optimize:
            logger.warning(
                f"Layer {i+1} has no trainable parameters. Skipping training."
            )
            continue

        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(
                params_to_optimize,
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_params_extra,
            )
        elif optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(
                params_to_optimize,
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_params_extra,
            )
        elif optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(
                params_to_optimize,
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_params_extra,
            )
        else:
            logger.error(f"Unsupported optimizer: {optimizer_type}")
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        # Train the current layer
        train_ff_layer(
            layer=layer,
            optimizer=optimizer,
            train_loader=train_loader,
            get_layer_input_fn=get_layer_input_fn,
            threshold=threshold,
            epochs=epochs_per_layer,
            device=device,
            layer_index=i,
            wandb_run=wandb_run,
            log_interval=log_interval,
        )

        # Freeze the trained layer parameters
        # layer.requires_grad_(False) # Freezes whole module, might be too broad
        for param in layer.parameters():
            param.requires_grad = False
        # Ensure layer stays in eval mode
        layer.eval()

    logger.info("Finished all layer-wise FF training.")


def evaluate_ff_model(
    model: FF_MLP,
    data_loader: DataLoader,  # Validation or Test loader
    device: torch.device,
    input_adapter: Optional[Callable] = None,  # e.g., flatten
) -> Dict[str, float]:
    """Evaluates the trained FF_MLP model using multi-pass inference."""
    model.eval()
    model.to(device)
    num_classes = model.num_classes
    logger.info(
        f"Evaluating FF model using multi-pass inference ({num_classes} passes per image)."
    )

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating FF Model", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]

            # Adapt original images (e.g., flatten) - MUST match training adaptation
            adapted_images = input_adapter(images) if input_adapter else images

            # Store summed goodness for each class for each image
            batch_total_goodness = torch.zeros((batch_size, num_classes), device=device)

            # Iterate through each possible class label candidate
            for label_candidate in range(num_classes):
                candidate_labels = torch.full(
                    (batch_size,), label_candidate, dtype=torch.long, device=device
                )
                # Prepare input with the candidate label embedded
                ff_input_candidate = model.prepare_ff_input(
                    adapted_images, candidate_labels
                )

                # Get goodness per layer for this candidate input
                try:
                    layer_goodness_list = model.forward_goodness_per_layer(
                        ff_input_candidate
                    )
                except Exception as e:
                    logger.error(
                        f"Error in FF goodness forward pass for candidate {label_candidate}: {e}",
                        exc_info=True,
                    )
                    # Assign very low goodness to avoid selecting this candidate
                    batch_total_goodness[:, label_candidate] = -torch.inf
                    continue  # Skip to next candidate

                # Sum goodness across layers
                if not layer_goodness_list:
                    total_goodness_candidate = torch.zeros((batch_size,), device=device)
                else:
                    try:
                        total_goodness_candidate = torch.stack(
                            layer_goodness_list, dim=0
                        ).sum(dim=0)
                    except Exception as e:
                        logger.error(
                            f"Error stacking/summing goodness: {e}", exc_info=True
                        )
                        batch_total_goodness[:, label_candidate] = -torch.inf
                        continue

                batch_total_goodness[:, label_candidate] = total_goodness_candidate

            # Predict the label with the highest total goodness
            try:
                predicted_labels = torch.argmax(batch_total_goodness, dim=1)
            except Exception as e:
                logger.error(f"Error during argmax prediction: {e}", exc_info=True)
                predicted_labels = torch.zeros_like(labels)  # Fallback prediction

            # Update correct count and total samples
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

            # Update progress bar if desired
            # current_acc = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
            # pbar.set_postfix(acc=f"{current_acc:.2f}%")

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"FF Evaluation Accuracy: {accuracy:.2f}%")

    results = {"eval_accuracy": accuracy}
    return results


# Removed the __main__ block for cleaner algorithm file
