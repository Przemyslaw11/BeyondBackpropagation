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
import os  # For checkpointing

from src.architectures.ff_mlp import FF_MLP, FF_Layer
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint  # Import checkpoint helper

logger = logging.getLogger(__name__)


def ff_loss_fn(
    pos_goodness: torch.Tensor, neg_goodness: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Calculates the Forward-Forward loss for a layer using logistic loss (softplus).
    Loss = softplus(-(pos_goodness - threshold)) + softplus(neg_goodness - threshold)
    """
    loss_pos = F.softplus(-(pos_goodness - threshold))
    loss_neg = F.softplus(neg_goodness - threshold)
    loss = torch.mean(loss_pos + loss_neg)
    return loss


def train_ff_layer(
    layer_module: FF_Layer,  # Rename 'layer' to avoid conflict with loop variable
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
    layer_module.train()  # Use the passed module instance
    layer_module.to(device)
    logger.info(f"Starting FF training for Layer {layer_index + 1}")

    total_steps_per_epoch = len(train_loader)
    global_step_offset = layer_index * epochs * total_steps_per_epoch

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
            _, pos_goodness = layer_module.forward_with_goodness(pos_input)
            _, neg_goodness = layer_module.forward_with_goodness(neg_input)

            # 3. Calculate loss
            loss = ff_loss_fn(pos_goodness, neg_goodness, threshold)

            # 4. Backpropagate and optimize (only for the current layer)
            optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping if needed
            # torch.nn.utils.clip_grad_norm_(layer_module.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_step = global_step_offset + epoch * total_steps_per_epoch + batch_idx

            # Logging
            if (
                batch_idx + 1
            ) % log_interval == 0 or batch_idx == total_steps_per_epoch - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.4f}")
                # Use consistent naming for W&B
                metrics = {
                    f"Layer_{layer_index+1}/Train_Loss_Batch": avg_loss_batch,
                    f"Layer_{layer_index+1}/Pos_Goodness_Mean": pos_goodness.mean().item(),
                    f"Layer_{layer_index+1}/Neg_Goodness_Mean": neg_goodness.mean().item(),
                }
                log_metrics(metrics, step=batch_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / total_steps_per_epoch
        logger.info(
            f"FF Layer {layer_index+1} Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}"
        )
        # Log epoch loss against global step number *within this layer's training*
        log_metrics(
            {f"Layer_{layer_index+1}/Train_Loss_Epoch": avg_epoch_loss},
            step=global_step_offset + (epoch + 1) * total_steps_per_epoch,
            wandb_run=wandb_run,
        )

    logger.info(f"Finished FF training for Layer {layer_index + 1}")
    layer_module.eval()


def train_ff_model(
    model: FF_MLP,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None,  # e.g., flatten
):
    """Orchestrates the layer-wise training of an FF_MLP model."""
    model.to(device)
    num_hidden_layers = len(model.hidden_dims)  # Number of actual hidden layers
    logger.info(
        f"Starting layer-wise FF training for {num_hidden_layers} hidden layers."
    )

    # --- Get Training Configuration ---
    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_type = algo_config.get("optimizer_type", "Adam")
    lr = algo_config.get("lr", 0.001)
    weight_decay = algo_config.get("weight_decay", 0.0)
    threshold = algo_config.get("threshold", 1.0)
    epochs_per_layer = algo_config.get("epochs_per_layer", 10)
    log_interval = algo_config.get("log_interval", 100)
    optimizer_params_extra = algo_config.get("optimizer_params", {})
    checkpoint_dir = config.get("checkpointing", {}).get(
        "checkpoint_dir", None
    )  # For saving model state

    # --- Helper function to generate pos/neg data (using external input adapter) ---
    @torch.no_grad()
    def generate_pos_neg_for_ff(base_images, base_labels):
        batch_size = base_images.shape[0]
        num_classes = model.num_classes
        device = base_images.device

        # 1. Adapt original images (e.g., flatten)
        adapted_images = input_adapter(base_images) if input_adapter else base_images
        if adapted_images.shape[1] != model.input_dim:
            raise ValueError(
                f"Adapted image dimension {adapted_images.shape[1]} does not match model input_dim {model.input_dim}"
            )

        # 2. Create one-hot labels
        one_hot_pos = F.one_hot(base_labels, num_classes=num_classes).float()

        # 3. Create negative labels
        rand_offset = torch.randint(
            1, num_classes, (batch_size,), device=device, dtype=torch.long
        )
        neg_labels = (base_labels + rand_offset) % num_classes
        collision = neg_labels == base_labels
        if torch.any(collision):
            neg_labels[collision] = (neg_labels[collision] + 1) % num_classes
        one_hot_neg = F.one_hot(neg_labels, num_classes=num_classes).float()

        # 4. Concatenate adapted images with labels
        # Shape: [B, num_classes + input_dim]
        pos_combined_input = torch.cat((one_hot_pos, adapted_images), dim=1)
        neg_combined_input = torch.cat((one_hot_neg, adapted_images), dim=1)

        return pos_combined_input, neg_combined_input

    # --- Train Layer by Layer ---
    # Handle input adapter layer separately first
    logger.info(f"--- Training Input Adapter Layer (Layer 0) ---")
    input_adapter_layer = model.input_adapter
    params_to_optimize = list(input_adapter_layer.parameters())
    if params_to_optimize:
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
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        # Train input adapter layer (using index -1 conceptually, logging as Layer 1)
        train_ff_layer(
            layer_module=input_adapter_layer,  # Train the linear layer directly
            optimizer=optimizer,
            train_loader=train_loader,
            get_layer_input_fn=generate_pos_neg_for_ff,  # Input is the combined pos/neg data
            threshold=threshold,
            epochs=epochs_per_layer,
            device=device,
            layer_index=0,  # Log as Layer 1
            wandb_run=wandb_run,
            log_interval=log_interval,
        )
        for param in input_adapter_layer.parameters():
            param.requires_grad = False
        input_adapter_layer.eval()
    else:
        logger.warning("Input adapter layer has no trainable parameters. Skipping.")

    # Train subsequent hidden FF_Layers (index i corresponds to FF_Layer i in model.layers)
    for i, ff_layer_module in enumerate(model.layers):
        layer_log_index = i + 1  # Layer 1, 2, ... in logs
        logger.info(
            f"--- Training Hidden FF_Layer {layer_log_index+1}/{num_hidden_layers} ---"
        )

        # Define function to get input for *this* FF_Layer 'i'
        def get_ff_layer_i_input_closure(
            current_ff_layer_idx: int,
        ):  # idx in model.layers list (0 to L-2)
            @torch.no_grad()
            def get_ff_layer_i_input(images, labels):
                base_pos_data, base_neg_data = generate_pos_neg_for_ff(images, labels)
                # Pass through input adapter AND layers 0 to i-1 to get input for layer i
                pos_input_i = model.forward_upto(base_pos_data, current_ff_layer_idx)
                neg_input_i = model.forward_upto(base_neg_data, current_ff_layer_idx)
                return pos_input_i.detach(), neg_input_i.detach()

            return get_ff_layer_i_input

        get_ff_layer_input_fn = get_ff_layer_i_input_closure(
            i
        )  # Closure needs the index in model.layers

        params_to_optimize = list(ff_layer_module.parameters())
        if not params_to_optimize:
            logger.warning(
                f"Hidden FF_Layer {layer_log_index+1} has no trainable parameters. Skipping training."
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
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        train_ff_layer(
            layer_module=ff_layer_module,
            optimizer=optimizer,
            train_loader=train_loader,
            get_layer_input_fn=get_ff_layer_input_fn,
            threshold=threshold,
            epochs=epochs_per_layer,
            device=device,
            layer_index=layer_log_index,  # Log as Layer 2, 3, ...
            wandb_run=wandb_run,
            log_interval=log_interval,
        )

        for param in ff_layer_module.parameters():
            param.requires_grad = False
        ff_layer_module.eval()

        # --- Checkpointing after each layer ---
        if checkpoint_dir:
            chkpt_filename = f"ff_layer_{layer_log_index}_complete.pth"
            save_checkpoint(
                state={
                    "state_dict": model.state_dict(),
                    "layer_trained": layer_log_index,
                },
                is_best=False,  # Just save sequentially
                filename=chkpt_filename,
                checkpoint_dir=checkpoint_dir,
            )

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
            if adapted_images.shape[1] != model.input_dim:
                raise ValueError(
                    f"Evaluation: Adapted image dimension {adapted_images.shape[1]} != model input_dim {model.input_dim}"
                )

            batch_total_goodness = torch.zeros((batch_size, num_classes), device=device)

            for label_candidate in range(num_classes):
                candidate_labels = torch.full(
                    (batch_size,), label_candidate, dtype=torch.long, device=device
                )

                # Create one-hot labels and concatenate
                one_hot_candidate = F.one_hot(
                    candidate_labels, num_classes=num_classes
                ).float()
                ff_input_candidate = torch.cat(
                    (one_hot_candidate, adapted_images), dim=1
                )

                try:
                    layer_goodness_list = model.forward_goodness_per_layer(
                        ff_input_candidate
                    )
                except Exception as e:
                    logger.error(
                        f"Error in FF goodness forward pass for candidate {label_candidate}: {e}",
                        exc_info=True,
                    )
                    batch_total_goodness[:, label_candidate] = -torch.inf
                    continue

                if not layer_goodness_list:
                    total_goodness_candidate = torch.zeros((batch_size,), device=device)
                else:
                    try:
                        # Sum goodness across all hidden layers
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

            try:
                predicted_labels = torch.argmax(batch_total_goodness, dim=1)
            except Exception as e:
                logger.error(f"Error during argmax prediction: {e}", exc_info=True)
                predicted_labels = torch.zeros_like(labels)

            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"FF Evaluation Accuracy: {accuracy:.2f}%")

    results = {"eval_accuracy": accuracy}  # Use consistent naming with other evals
    return results
