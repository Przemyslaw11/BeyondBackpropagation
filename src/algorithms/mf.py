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
from src.utils.helpers import save_checkpoint, format_time  # Import format_time

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


def train_mf_matrix_only(  # New function to train only M0
    model: MF_MLP,
    matrix_index: int,  # Should be 0 for M0
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    get_matrix_input_fn: Callable[[torch.Tensor], torch.Tensor],  # Function to get a_0
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
) -> None:
    """
    Trains a single projection matrix (specifically M0) using local loss based on its corresponding activation (a0).
    """
    if matrix_index != 0:
        logger.warning(
            f"train_mf_matrix_only called for matrix index {matrix_index}, expected 0 for M0."
        )

    projection_matrix = model.get_projection_matrix(matrix_index)  # M_0
    projection_matrix.requires_grad_(True)  # Ensure M_0 is trainable

    model.to(device)
    model.eval()  # Keep rest of model in eval mode

    logger.info(f"Starting MF training for Projection Matrix M_{matrix_index}")

    total_steps_per_epoch = len(train_loader)
    # Global step offset is 0 for the first matrix training phase
    global_step_offset = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"M{matrix_index} Epoch {epoch+1}/{epochs}", leave=False
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # 1. Get input activation (a_0) - DETACHED
            activation_a_i = get_matrix_input_fn(images).detach()

            # 2. Calculate local MF loss using a_0 and M_0
            loss = mf_local_loss_fn(
                activation_a_i, projection_matrix, labels, criterion
            )

            # --- Check for NaN/Inf Loss ---
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"NaN or Inf loss detected at Matrix M{matrix_index}, Epoch {epoch+1}, Batch {batch_idx}. Stopping layer training."
                )
                break  # Exit batch loop

            # 3. Backpropagate and optimize ONLY M_0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Updates only M_0

            epoch_loss += loss.item()
            current_global_step = (
                global_step_offset + epoch * total_steps_per_epoch + batch_idx
            )

            if (
                batch_idx + 1
            ) % log_interval == 0 or batch_idx == total_steps_per_epoch - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.6f}")
                metrics = {f"Layer_M{matrix_index}/Train_Loss_Batch": avg_loss_batch}
                log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

        # --- End of Epoch ---
        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)):
            logger.error(
                f"Terminating training for Matrix M{matrix_index} due to invalid loss in epoch {epoch+1}."
            )
            break  # Exit epoch loop

        avg_epoch_loss = (
            epoch_loss / total_steps_per_epoch if total_steps_per_epoch > 0 else 0.0
        )
        logger.info(
            f"Matrix M{matrix_index} Epoch {epoch+1}/{epochs} - Avg Local Loss: {avg_epoch_loss:.6f}"
        )
        log_metrics(
            {f"Layer_M{matrix_index}/Train_Loss_Epoch": avg_epoch_loss},
            step=global_step_offset + (epoch + 1) * total_steps_per_epoch,
            wandb_run=wandb_run,
        )

    logger.info(f"Finished MF training for Matrix M_{matrix_index}")
    projection_matrix.requires_grad_(False)  # Freeze M_0 after training


def train_mf_layer(
    model: MF_MLP,
    layer_index: int,  # Index i=0..L-1. Trains W_{i+1} and M_{i+1}.
    optimizer: optim.Optimizer,  # Optimizer containing params for W_{i+1} AND M_{i+1}
    criterion: nn.Module,  # Should be CrossEntropyLoss for MF local loss
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    get_layer_input_fn: Callable[
        [torch.Tensor], torch.Tensor
    ],  # Function to get input a_i
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
) -> None:
    """
    Trains a single layer 'i+1' (W_{i+1} and M_{i+1}) of an MF_MLP using local loss.
    Input is a_i (from get_layer_input_fn), output is a_{i+1}. Loss is calculated on a_{i+1} using M_{i+1}.
    """
    # Indices in MF_MLP structure:
    # Linear layers W_1, W_2, ... W_L are at indices 0, 2, ..., 2*(L-1)
    # Activation layers sigma_1, ... sigma_L are at indices 1, 3, ..., 2*(L-1)+1
    # Projection matrices M_0, M_1, ..., M_L correspond to activations a_0, a_1, ..., a_L
    linear_layer_idx = layer_index * 2  # Index of W_{i+1}
    act_layer_idx = layer_index * 2 + 1  # Index of sigma_{i+1}
    proj_matrix_idx = layer_index + 1  # Index of M_{i+1} (needs activation a_{i+1})
    w_layer_log_idx = layer_index + 1  # 1-based index for logging W
    m_matrix_log_idx = proj_matrix_idx  # Use M matrix index for logging loss

    logger.debug(
        f"Training Layer Indices: W={linear_layer_idx}, Act={act_layer_idx}, M={proj_matrix_idx}"
    )

    linear_layer = model.layers[linear_layer_idx]  # W_{i+1}
    act_layer = model.layers[act_layer_idx]  # sigma_{i+1}
    projection_matrix = model.get_projection_matrix(proj_matrix_idx)  # M_{i+1}

    # Set only the current layer's components to train mode
    model.eval()  # Default to eval
    linear_layer.train()
    act_layer.train()
    projection_matrix.requires_grad_(True)  # Ensure M_{i+1} is trainable
    model.to(device)

    logger.info(
        f"Starting MF training for Layer W_{w_layer_log_idx} / Matrix M_{m_matrix_log_idx}"
    )

    total_steps_per_epoch = len(train_loader)
    # Adjust global step offset based on the fact M0 was trained first
    global_step_offset = (layer_index + 1) * epochs * total_steps_per_epoch

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"W{w_layer_log_idx}/M{m_matrix_log_idx} Epoch {epoch+1}/{epochs}",
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

            # --- Check for NaN/Inf Loss ---
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"NaN or Inf loss detected at Layer W{w_layer_log_idx}/M{m_matrix_log_idx}, "
                    f"Epoch {epoch+1}, Batch {batch_idx}. Stopping layer training."
                )
                # Skip optimizer step and break epoch
                break  # Exit batch loop for this epoch

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
                # Log using M matrix index for consistency
                metrics = {
                    f"Layer_M{m_matrix_log_idx}/Train_Loss_Batch": avg_loss_batch
                }
                log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

        # --- End of Epoch ---
        # Check if loop was broken due to NaN/Inf loss
        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)):
            logger.error(
                f"Terminating training for Layer W{w_layer_log_idx}/M{m_matrix_log_idx} due to invalid loss in epoch {epoch+1}."
            )
            break  # Exit epoch loop for this layer

        avg_epoch_loss = (
            epoch_loss / total_steps_per_epoch if total_steps_per_epoch > 0 else 0.0
        )
        logger.info(
            f"Layer (W{w_layer_log_idx}/M{m_matrix_log_idx}) Epoch {epoch+1}/{epochs} - Avg Local Loss: {avg_epoch_loss:.6f}"
        )
        log_metrics(
            {f"Layer_M{m_matrix_log_idx}/Train_Loss_Epoch": avg_epoch_loss},
            step=global_step_offset + (epoch + 1) * total_steps_per_epoch,
            wandb_run=wandb_run,
        )

    logger.info(
        f"Finished MF training for Layer (W{w_layer_log_idx}/M{m_matrix_log_idx})"
    )
    linear_layer.eval()
    act_layer.eval()
    projection_matrix.requires_grad_(False)


def train_mf_model(
    model: MF_MLP,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None,
):
    """
    Orchestrates the layer-wise training of an MF_MLP model.
    First trains M_0 using input a_0.
    Then iteratively trains W_{i+1} and M_{i+1} using input a_i and activation a_{i+1}.
    """
    model.to(device)
    num_hidden_layers = model.num_hidden_layers
    logger.info(
        f"Starting layer-wise MF training for M0 and {num_hidden_layers} hidden layers."
    )

    # --- Configuration ---
    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    optimizer_params_config = algo_config.get(
        "optimizer_params", {}
    )  # Get the nested dict
    lr = optimizer_params_config.get("lr", 0.001)
    weight_decay = optimizer_params_config.get("weight_decay", 0.0)
    # Extract other params, excluding lr and wd which are handled separately
    optimizer_extra_kwargs = {
        k: v
        for k, v in optimizer_params_config.items()
        if k not in ["lr", "weight_decay"]
    }

    epochs_per_layer = algo_config.get("epochs_per_layer", 5)
    log_interval = algo_config.get("log_interval", 100)
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None)

    mf_criterion = nn.CrossEntropyLoss()

    # --- Define Input Function for Layer 0 (provides a_0) ---
    def get_a0_input(img_batch):
        # Apply input adapter (e.g., flattening) if provided
        adapted_input = input_adapter(img_batch) if input_adapter else img_batch
        return adapted_input.to(device)

    # --- Train M_0 ---
    logger.info("--- Training Projection Matrix M_0 ---")
    projection_matrix_0 = model.get_projection_matrix(0)
    optimizer_0_kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
        **optimizer_extra_kwargs,
    }  # Add extra args
    params_m0 = [projection_matrix_0]
    if not any(p.requires_grad for p in params_m0):
        logger.warning("M_0 does not require gradients. Skipping training.")
    else:
        if optimizer_name.lower() == "adam":
            optimizer_0 = optim.Adam(params_m0, **optimizer_0_kwargs)
        elif optimizer_name.lower() == "sgd":
            optimizer_0 = optim.SGD(params_m0, **optimizer_0_kwargs)
        elif optimizer_name.lower() == "adamw":
            optimizer_0 = optim.AdamW(params_m0, **optimizer_0_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        train_mf_matrix_only(  # Use the new function
            model=model,
            matrix_index=0,
            optimizer=optimizer_0,
            criterion=mf_criterion,
            train_loader=train_loader,
            epochs=epochs_per_layer,
            device=device,
            get_matrix_input_fn=get_a0_input,  # Provides a_0
            wandb_run=wandb_run,
            log_interval=log_interval,
        )
        # Checkpointing after M0
        if checkpoint_dir:
            chkpt_filename = "mf_matrix_M0_complete.pth"
            save_checkpoint(
                state={
                    "state_dict": model.state_dict(),
                    "layer_trained_index": -1,
                },  # Indicate M0 trained
                is_best=False,
                filename=chkpt_filename,
                checkpoint_dir=checkpoint_dir,
            )

    # Update the input function for the *first hidden layer's* training (needs a_0)
    current_layer_input_fn = get_a0_input

    # --- Train Hidden Layers (W_{i+1}, M_{i+1} for i=0..L-1) ---
    for i in range(num_hidden_layers):
        w_layer_log_idx = i + 1
        m_matrix_log_idx = i + 1
        logger.info(
            f"--- Training Hidden Layer W_{w_layer_log_idx} / Matrix M_{m_matrix_log_idx} ---"
        )

        linear_layer = model.layers[i * 2]  # W_{i+1}
        projection_matrix = model.get_projection_matrix(i + 1)  # M_{i+1}

        params_to_optimize = list(linear_layer.parameters()) + [projection_matrix]
        if not any(p.requires_grad for p in params_to_optimize):
            logger.warning(
                f"Skipping training for layer W{w_layer_log_idx}/M{m_matrix_log_idx} as no parameters require gradients."
            )

            # --- Define the input function for the *next* layer even if skipping training ---
            def create_next_input_fn_skip(
                trained_layer_idx: int, previous_input_fn: Callable
            ):
                lin_layer_k = model.layers[trained_layer_idx * 2]
                act_layer_k = model.layers[trained_layer_idx * 2 + 1]
                lin_layer_k.eval()
                act_layer_k.eval()

                @torch.no_grad()
                def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                    a_prev = previous_input_fn(img_batch)
                    a_current = act_layer_k(lin_layer_k(a_prev))
                    return a_current.detach()  # Return detached

                return next_input_fn

            current_layer_input_fn = create_next_input_fn_skip(
                i, current_layer_input_fn
            )
            continue  # Skip to next layer

        optimizer_i_kwargs = {
            "lr": lr,
            "weight_decay": weight_decay,
            **optimizer_extra_kwargs,
        }  # Add extra args
        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(params_to_optimize, **optimizer_i_kwargs)
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(params_to_optimize, **optimizer_i_kwargs)
        elif optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(params_to_optimize, **optimizer_i_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        train_mf_layer(  # Use the unified training function
            model=model,
            layer_index=i,  # Pass index i (trains W_{i+1}, M_{i+1})
            optimizer=optimizer,
            criterion=mf_criterion,
            train_loader=train_loader,
            epochs=epochs_per_layer,
            device=device,
            get_layer_input_fn=current_layer_input_fn,  # Provides a_i
            wandb_run=wandb_run,
            log_interval=log_interval,
        )

        # Freeze trained parameters
        for param in params_to_optimize:
            param.requires_grad = False
        linear_layer.eval()
        model.layers[i * 2 + 1].eval()  # Ensure activation layer is also eval

        # --- Checkpointing ---
        if checkpoint_dir:
            chkpt_filename = (
                f"mf_hidden_layer_W{w_layer_log_idx}_M{m_matrix_log_idx}_complete.pth"
            )
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained_index": i},
                is_best=False,
                filename=chkpt_filename,
                checkpoint_dir=checkpoint_dir,
            )

        # --- Define the input function for the *next* layer (provides a_{i+1}) ---
        def create_next_input_fn(trained_layer_idx: int, previous_input_fn: Callable):
            # Get references to the layers we just trained and ensure eval mode
            lin_layer_k = model.layers[trained_layer_idx * 2]
            act_layer_k = model.layers[trained_layer_idx * 2 + 1]
            lin_layer_k.eval()
            act_layer_k.eval()

            @torch.no_grad()
            def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                a_prev = previous_input_fn(img_batch)
                a_current = act_layer_k(lin_layer_k(a_prev))
                return a_current.detach()  # Return detached

            return next_input_fn

        current_layer_input_fn = create_next_input_fn(i, current_layer_input_fn)

    logger.info("Finished all layer-wise MF training.")
    model.eval()  # Ensure entire model is in eval mode


def evaluate_mf_model(
    model: MF_MLP,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[
        nn.Module
    ] = None,  # Criterion usually not applicable for MF eval
    input_adapter: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Evaluates the trained MF_MLP model using the paper's "BP-style" approach:
    Calculates goodness scores G_L = a_L @ M_L^T using the last hidden layer's
    activation (a_L) and the corresponding projection matrix (M_L). Predicts based
    on the argmax of G_L.
    """
    model.eval()
    model.to(device)
    total_correct, total_samples = 0, 0
    num_hidden_layers = model.num_hidden_layers
    # Index of the last hidden activation a_L is L (list index)
    # Index of the last projection matrix M_L is L (ParameterList index)
    last_activation_index = num_hidden_layers
    last_projection_matrix_index = num_hidden_layers

    logger.info(
        f"Evaluating MF model using final hidden activation (a_{last_activation_index}) and projection matrix (M_{last_projection_matrix_index})."
    )

    if last_projection_matrix_index >= len(model.projection_matrices):
        logger.error(
            f"Index for last projection matrix M_{last_projection_matrix_index} is out of bounds ({len(model.projection_matrices)} matrices available). Evaluation failed."
        )
        return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}

    last_projection_matrix = model.get_projection_matrix(last_projection_matrix_index)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating MF (BP-style)", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            # Apply input adapter (e.g., flattening) before feeding to model
            eval_input = input_adapter(images) if input_adapter else images

            # Get all intermediate activations [a_0, a_1, ..., a_L]
            all_activations = model.forward_with_intermediate_activations(eval_input)

            if len(all_activations) <= last_activation_index:
                logger.error(
                    f"Activation list length ({len(all_activations)}) is insufficient to get activation a_{last_activation_index}. Evaluation failed for batch."
                )
                continue  # Skip batch

            # Get the last hidden activation a_L
            last_hidden_activation = all_activations[last_activation_index]  # a_L

            # Calculate goodness scores using M_L: G_L = a_L @ M_L^T
            goodness_scores = torch.matmul(
                last_hidden_activation, last_projection_matrix.t()
            )

            # Get predictions based on goodness scores
            predicted_labels = torch.argmax(goodness_scores, dim=1)

            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"MF Evaluation Results (BP-style): Accuracy: {accuracy:.2f}%")

    # MF evaluation doesn't typically have a directly comparable loss value
    results = {
        "eval_accuracy": accuracy,
        "eval_loss": float("nan"),  # Indicate loss is not applicable/calculated
    }
    return results
