# File: src/algorithms/mf.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import logging
from tqdm import tqdm
from typing import Dict, Any, Optional, Callable, List, Tuple
import os  # For checkpointing

from src.architectures.mf_mlp import MF_MLP
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics
from src.utils.helpers import save_checkpoint  # Import checkpoint helper

logger = logging.getLogger(__name__)


def mf_local_loss_fn(
    activation_i: torch.Tensor,
    projection_matrix_i: nn.Parameter,
    targets: torch.Tensor,
    criterion: nn.Module = nn.CrossEntropyLoss(),
) -> torch.Tensor:
    """Calculates the Mono-Forward local cross-entropy loss for activation a_i using M_i."""
    # G_i = a_i @ M_i^T  -> [B, N_i] @ [N_i, C] -> [B, C]
    goodness_scores_i = torch.matmul(activation_i, projection_matrix_i.t())
    loss = criterion(goodness_scores_i, targets)
    return loss


def train_mf_layer(  # Renamed from train_mf_hidden_layer
    model: MF_MLP,
    layer_index: int,  # Index i for W_i and M_i (0-based for W_1/M_0, 1 for W_2/M_1 etc.)
    optimizer: optim.Optimizer,  # Optimizer containing params for W_i AND M_i
    criterion: nn.Module,  # Should be CrossEntropyLoss for MF local loss
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    get_layer_input_fn: Callable[
        [torch.Tensor], torch.Tensor
    ],  # Function to get input a_{i-1}
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
) -> None:
    """
    Trains a single layer 'i' (W_i and M_i) of an MF_MLP using local loss.
    Input is a_{i-1}, output is a_i. Loss is calculated on a_i using M_i.
    """
    # Indices in MF_MLP structure:
    # Linear layers W_1, W_2, ... W_L are at indices 0, 2, ..., 2*(L-1)
    # Activation layers sigma_1, ... sigma_L are at indices 1, 3, ..., 2*(L-1)+1
    # Projection matrices M_0, M_1, ..., M_{L-1} correspond to activations a_0, a_1, ..., a_{L-1}
    linear_layer_idx = layer_index * 2  # Index of W_{i+1}
    act_layer_idx = layer_index * 2 + 1  # Index of sigma_{i+1}
    proj_matrix_idx = layer_index + 1  # Index of M_{i+1} - Needs a_{i+1}

    linear_layer = model.layers[linear_layer_idx]  # W_{i+1}
    act_layer = model.layers[act_layer_idx]  # sigma_{i+1}
    # We need the *next* projection matrix to apply loss on the current activation output
    if proj_matrix_idx >= len(model.projection_matrices):
        logger.error(
            f"Attempting to train layer {layer_index} but projection matrix M_{proj_matrix_idx} is out of bounds."
        )
        raise IndexError("Projection matrix index out of bounds during MF training.")
    projection_matrix = model.get_projection_matrix(proj_matrix_idx)  # M_{i+1}

    # Set only the current layer's components to train mode
    model.eval()  # Default to eval
    linear_layer.train()
    act_layer.train()
    projection_matrix.requires_grad_(True)  # Ensure M_{i+1} is trainable
    model.to(device)

    logger.info(
        f"Starting MF training for Layer W_{layer_index+1} / Matrix M_{layer_index+1}"
    )

    total_steps_per_epoch = len(train_loader)
    global_step_offset = layer_index * epochs * total_steps_per_epoch

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"W{layer_index+1}/M{layer_index+1} Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # 1. Get input for the current layer (a_i) - DETACHED
            prev_activation_a_i = get_layer_input_fn(images).detach()

            # 2. Forward pass through W_{i+1} + sigma_{i+1} to get a_{i+1}
            # Gradients should flow back through these for W_{i+1} update
            pre_activation_z_next = linear_layer(prev_activation_a_i)  # z_{i+1}
            activation_a_next = act_layer(pre_activation_z_next)  # a_{i+1}

            # 3. Calculate local MF loss using a_{i+1} and M_{i+1}
            loss = mf_local_loss_fn(
                activation_a_next, projection_matrix, labels, criterion
            )

            # 4. Backpropagate and optimize W_{i+1} and M_{i+1}
            optimizer.zero_grad()
            loss.backward()  # Computes dL/da_{i+1} -> dL/dz_{i+1} -> dL/dW_{i+1} and dL/dM_{i+1}
            optimizer.step()  # Updates W_{i+1} and M_{i+1}

            epoch_loss += loss.item()
            current_global_step = (
                global_step_offset + epoch * total_steps_per_epoch + batch_idx
            )

            if (
                batch_idx + 1
            ) % log_interval == 0 or batch_idx == total_steps_per_epoch - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.6f}")
                # Log using projection matrix index for consistency
                metrics = {f"Layer_M{proj_matrix_idx}/Train_Loss_Batch": avg_loss_batch}
                log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / total_steps_per_epoch
        logger.info(
            f"Layer (W{layer_index+1}/M{proj_matrix_idx}) Epoch {epoch+1}/{epochs} - Avg Local Loss: {avg_epoch_loss:.6f}"
        )
        log_metrics(
            {f"Layer_M{proj_matrix_idx}/Train_Loss_Epoch": avg_epoch_loss},
            step=global_step_offset + (epoch + 1) * total_steps_per_epoch,
            wandb_run=wandb_run,
        )

    logger.info(f"Finished MF training for Layer (W{layer_index+1}/M{proj_matrix_idx})")
    linear_layer.eval()
    act_layer.eval()
    # Keep projection matrix trainable if needed for subsequent layers? MF doesn't reuse M?
    # If M_i is only updated based on a_i, it can be frozen here. Let's assume it's frozen.
    projection_matrix.requires_grad_(False)


def train_mf_output_layer(
    model: MF_MLP,
    optimizer: optim.Optimizer,
    criterion: nn.Module,  # Usually CrossEntropy
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    get_output_layer_input_fn: Callable[[torch.Tensor], torch.Tensor],  # Gets input a_L
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    config: Dict[str, Any] = {},  # Pass config for global step calculation
) -> None:
    """Trains the final output layer of an MF_MLP using standard supervised objective."""
    output_layer = model.output_layer
    model.eval()  # Ensure hidden layers are frozen
    output_layer.train()
    model.to(device)
    logger.info("Starting MF training for Output Layer")

    total_steps_per_epoch = len(train_loader)
    # Offset steps based on hidden layer training epochs
    epochs_per_layer = config.get("algorithm_params", {}).get("epochs_per_layer", 5)
    global_step_offset = (
        model.num_hidden_layers * epochs_per_layer * total_steps_per_epoch
    )

    for epoch in range(epochs):
        epoch_loss, epoch_accuracy = 0.0, 0.0
        pbar = tqdm(
            train_loader, desc=f"Output Layer Epoch {epoch+1}/{epochs}", leave=False
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():  # Input to output layer should be detached
                output_layer_input = get_output_layer_input_fn(images).detach()

            predictions = output_layer(output_layer_input)
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

            if (
                batch_idx + 1
            ) % log_interval == 0 or batch_idx == total_steps_per_epoch - 1:
                metrics = {
                    "Output_Layer/Train_Loss_Batch": loss.item(),
                    "Output_Layer/Train_Acc_Batch": batch_accuracy,
                }
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", acc=f"{batch_accuracy:.2f}%"
                )
                log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / total_steps_per_epoch
        avg_epoch_accuracy = epoch_accuracy / total_steps_per_epoch
        logger.info(
            f"Output Layer Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_accuracy:.2f}%"
        )
        log_metrics(
            {
                "Output_Layer/Train_Loss_Epoch": avg_epoch_loss,
                "Output_Layer/Train_Acc_Epoch": avg_epoch_accuracy,
            },
            step=global_step_offset + (epoch + 1) * total_steps_per_epoch,
            wandb_run=wandb_run,
        )
    logger.info("Finished MF training for Output Layer")
    output_layer.eval()


def train_mf_model(
    model: MF_MLP,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None,
):
    """Orchestrates the layer-wise training of an MF_MLP model."""
    model.to(device)
    num_hidden_layers = model.num_hidden_layers
    logger.info(
        f"Starting layer-wise MF training for {num_hidden_layers} hidden layers + 1 output layer."
    )

    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    # Use nested dict for optimizer params if defined that way
    optimizer_params_config = algo_config.get(
        "optimizer_params", {"lr": 0.001}
    )  # Default lr
    lr = optimizer_params_config.get("lr", 0.001)  # Extract specific lr
    weight_decay = optimizer_params_config.get("weight_decay", 0.0)

    output_criterion_name = algo_config.get("output_criterion", "CrossEntropyLoss")
    epochs_per_layer = algo_config.get("epochs_per_layer", 5)
    epochs_output_layer = algo_config.get(
        "epochs_output_layer", epochs_per_layer
    )  # Default to same as hidden
    log_interval = algo_config.get("log_interval", 100)
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None)

    mf_criterion = nn.CrossEntropyLoss()
    if output_criterion_name.lower() == "crossentropyloss":
        output_criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported output criterion: {output_criterion_name}")

    # --- Input Function for Layer 0 (provides a_0) ---
    def get_layer0_input(img_batch):
        adapted_input = input_adapter(img_batch) if input_adapter else img_batch
        return adapted_input.to(device)

    current_layer_input_fn = get_layer0_input

    # --- Train Hidden Layers ---
    # Loop from layer_index i = 0 to L-1. Trains W_{i+1} and M_{i+1} using input a_i.
    for i in range(num_hidden_layers):
        logger.info(f"--- Training Hidden Layer W_{i+1} / Matrix M_{i+1} ---")
        linear_layer = model.layers[i * 2]  # W_{i+1}
        act_layer = model.layers[i * 2 + 1]  # sigma_{i+1}
        projection_matrix = model.get_projection_matrix(i + 1)  # M_{i+1}

        params_to_optimize = list(linear_layer.parameters()) + [projection_matrix]
        if not any(p.requires_grad for p in params_to_optimize):
            logger.warning(
                f"Skipping training for layer W{i+1}/M{i+1} as no parameters require gradients."
            )
            continue  # Skip if no params need training (e.g., after loading checkpoint)

        optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay}
        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(params_to_optimize, **optimizer_kwargs)
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(params_to_optimize, **optimizer_kwargs)
        elif optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(params_to_optimize, **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        train_mf_layer(  # Use the unified training function
            model=model,
            layer_index=i,  # Pass index i (corresponds to W_{i+1}, M_{i+1})
            optimizer=optimizer,
            criterion=mf_criterion,
            train_loader=train_loader,
            epochs=epochs_per_layer,
            device=device,
            get_layer_input_fn=current_layer_input_fn,  # Provides a_i
            wandb_run=wandb_run,
            log_interval=log_interval,
        )

        for param in params_to_optimize:
            param.requires_grad = False
        linear_layer.eval()
        act_layer.eval()

        # --- Checkpointing ---
        if checkpoint_dir:
            chkpt_filename = (
                f"mf_hidden_layer_{i}_complete.pth"  # W_{i+1}/M_{i+1} trained
            )
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained": i},
                is_best=False,
                filename=chkpt_filename,
                checkpoint_dir=checkpoint_dir,
            )

        # --- Define the input function for the *next* layer (provides a_{i+1}) ---
        def create_next_input_fn(trained_layer_idx: int, previous_input_fn: Callable):
            lin_layer_k = model.layers[trained_layer_idx * 2]
            act_layer_k = model.layers[trained_layer_idx * 2 + 1]

            @torch.no_grad()
            def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                a_prev = previous_input_fn(img_batch)
                a_current = act_layer_k(lin_layer_k(a_prev))
                return a_current

            return next_input_fn

        current_layer_input_fn = create_next_input_fn(i, current_layer_input_fn)

    # --- Train Output Layer (W_L+1) ---
    logger.info(f"--- Training Output Layer ---")
    output_layer = model.output_layer
    output_params = list(output_layer.parameters())

    if not any(p.requires_grad for p in output_params):
        logger.warning(
            "Skipping output layer training as no parameters require gradients."
        )
    else:
        optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay}
        if optimizer_name.lower() == "adam":
            output_optimizer = optim.Adam(output_params, **optimizer_kwargs)
        elif optimizer_name.lower() == "sgd":
            output_optimizer = optim.SGD(output_params, **optimizer_kwargs)
        elif optimizer_name.lower() == "adamw":
            output_optimizer = optim.AdamW(output_params, **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # The input function needed provides a_L (output of last hidden layer)
        train_mf_output_layer(
            model=model,
            optimizer=output_optimizer,
            criterion=output_criterion,
            train_loader=train_loader,
            epochs=epochs_output_layer,
            device=device,
            get_output_layer_input_fn=current_layer_input_fn,  # Provides a_L
            wandb_run=wandb_run,
            log_interval=log_interval,
            config=config,  # Pass config for global step calculation
        )

        # --- Checkpointing after output layer ---
        if checkpoint_dir:
            chkpt_filename = "mf_output_layer_complete.pth"
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained": "output"},
                is_best=False,
                filename=chkpt_filename,
                checkpoint_dir=checkpoint_dir,
            )

    logger.info("Finished all layer-wise MF training.")


def evaluate_mf_model(
    model: MF_MLP,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    input_adapter: Optional[Callable] = None,
) -> Dict[str, float]:
    """Evaluates the trained MF_MLP model using its standard forward pass (BP-style)."""
    model.eval()
    model.to(device)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    logger.info("Evaluating MF model (standard forward pass / BP-style)")
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating MF", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            eval_input = input_adapter(images) if input_adapter else images

            predictions = model.forward(eval_input)

            loss_val = float("nan")
            if criterion:
                try:
                    loss = criterion(predictions, labels)
                    loss_val = loss.item()
                    total_loss += loss_val * images.size(0)
                except Exception as e:
                    logger.warning(f"Failed to compute evaluation loss: {e}")
                    total_loss = float("nan")

            predicted_labels = torch.argmax(predictions, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = (
        total_loss / total_samples
        if criterion and total_samples > 0 and not torch.isnan(torch.tensor(total_loss))
        else float("nan")
    )
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(
        f"Evaluation Results: Accuracy: {accuracy:.2f}%"
        + (f", Loss: {avg_loss:.4f}" if not torch.isnan(torch.tensor(avg_loss)) else "")
    )
    results = {"eval_accuracy": accuracy}
    if not torch.isnan(torch.tensor(avg_loss)):
        results["eval_loss"] = avg_loss
    return results
