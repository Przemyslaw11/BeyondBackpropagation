# File: src/algorithms/mf.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import logging
from tqdm import tqdm
from typing import Dict, Any, Optional, Callable, List, Tuple

from src.architectures.mf_mlp import MF_MLP
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics

logger = logging.getLogger(__name__)


def mf_local_loss_fn(
    activation_i: torch.Tensor,  # a_i (output of layer i) [B, N_i]
    projection_matrix_i: nn.Parameter,  # M_i [C, N_i]
    targets: torch.Tensor,  # Ground truth labels (indices) [B]
    criterion: nn.Module = nn.CrossEntropyLoss(),
) -> torch.Tensor:
    """Calculates the Mono-Forward local cross-entropy loss for layer i."""
    # G_i = a_i @ M_i^T  -> [B, N_i] @ [N_i, C] -> [B, C]
    goodness_scores_i = torch.matmul(activation_i, projection_matrix_i.t())
    loss = criterion(goodness_scores_i, targets)
    return loss


def train_mf_hidden_layer(
    model: MF_MLP,
    layer_index: int,  # Index of the hidden layer to train (0-based)
    optimizer: optim.Optimizer,  # Optimizer containing params for W_i AND M_i
    criterion: nn.Module,  # Should be CrossEntropyLoss
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    get_layer_input_fn: Callable[
        [torch.Tensor], torch.Tensor
    ],  # Function to get input a_{i}
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
) -> None:
    """
    Trains a single hidden layer 'i' (W_i) and its projection matrix (M_i) of an MF_MLP.
    Note: layer_index=0 trains W_1 and M_0 based on input a_0.
          layer_index=k trains W_{k+1} and M_k based on input a_k.
    """
    # Layer indices in MF_MLP structure:
    # Linear layers W_1, W_2, ... W_L are at indices 0, 2, ..., 2*(L-1)
    # Activations sigma_1, ... sigma_L are at indices 1, 3, ..., 2*(L-1)+1
    # Projection matrices M_0, M_1, ..., M_{L-1} correspond to activations a_1, a_2, ..., a_L
    linear_layer_idx = layer_index * 2  # Index of W_{i+1}
    act_layer_idx = layer_index * 2 + 1  # Index of sigma_{i+1}
    proj_matrix_idx = layer_index  # Index of M_i

    linear_layer = model.layers[linear_layer_idx]  # W_{i+1}
    act_layer = model.layers[act_layer_idx]  # sigma_{i+1}
    projection_matrix = model.get_projection_matrix(proj_matrix_idx)  # M_i

    # Set only the current layer's components to train mode
    model.eval()  # Default to eval
    linear_layer.train()
    act_layer.train()
    # Ensure projection matrix is trainable (should be by default)
    projection_matrix.requires_grad_(True)
    model.to(device)

    logger.info(
        f"Starting MF training for Hidden Layer (W_{layer_index+1}) and Projection Matrix (M_{layer_index})"
    )

    total_steps_per_epoch = len(train_loader)
    global_step_offset = layer_index * epochs * total_steps_per_epoch

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"W_{layer_index+1}/M_{layer_index} Epoch {epoch+1}/{epochs}",
            leave=False,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # 1. Get input for the current layer (a_i) - DETACHED
            layer_input_a_i = get_layer_input_fn(images).detach()

            # 2. Forward pass through W_{i+1} + sigma_{i+1} to get a_{i+1}
            # Gradients should flow back through these for W_{i+1} update
            pre_activation_z_next = linear_layer(
                layer_input_a_i
            )  # z_{i+1} = a_i W_{i+1}
            activation_a_next = act_layer(
                pre_activation_z_next
            )  # a_{i+1} = sigma(z_{i+1})

            # 3. Calculate local MF loss using a_{i+1} and M_i
            # Error: Loss should be calculated using activation a_i and M_i
            # Let's recalculate loss using layer_input_a_i and M_i (projection_matrix)
            loss = mf_local_loss_fn(
                layer_input_a_i, projection_matrix, labels, criterion
            )

            # 4. Backpropagate and optimize W_{i+1} and M_i
            optimizer.zero_grad()
            loss.backward()  # Computes gradients for M_i (and W_{i+1} if loss depended on a_{i+1})
            # Correction: Loss depends on a_i, so gradients flow to M_i and W_i.
            # Let's fix the calculation and loss application.

            # --- Corrected Logic ---
            # We need to calculate the loss based on a_i (layer_input_a_i) and M_i.
            # The optimizer contains parameters for W_i and M_i.
            # So, the forward pass to calculate the loss should only go up to a_i.
            # The backward pass from L_i = CE(softmax(a_i @ M_i^T), y) will compute dL/da_i.
            # We need dL/dW_i and dL/dM_i.
            # dL/dM_i comes directly from the loss term.
            # dL/dW_i = dL/da_i * da_i/dz_i * dz_i/dW_i
            # This requires the gradient from the *local* loss to flow back through the activation and linear layer *of the current stage*.

            # Revised steps:
            # a) Get input a_{i-1} (from get_layer_input_fn)
            # b) Compute z_i = a_{i-1} @ W_i
            # c) Compute a_i = sigma(z_i)
            # d) Compute G_i = a_i @ M_i^T
            # e) Compute L_i = CE(G_i, y)
            # f) loss.backward() -> calculates dL_i/dW_i and dL_i/dM_i
            # g) optimizer.step() -> updates W_i and M_i

            # Let's rewrite the loop for clarity with indices matching paper/plan:
            # Train layer 'i' (W_i, M_i) using input a_{i-1}
            current_lin_layer = linear_layer  # W_i
            current_act_layer = act_layer  # sigma_i
            current_proj_matrix = projection_matrix  # M_i
            prev_activation_a_im1 = layer_input_a_i  # a_{i-1}

            # Forward for loss calculation (tracking grads for W_i, M_i)
            z_i = current_lin_layer(prev_activation_a_im1)
            a_i = current_act_layer(z_i)
            loss = mf_local_loss_fn(a_i, current_proj_matrix, labels, criterion)

            # Backpropagate and optimize W_i and M_i
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # --- End Corrected Logic ---

            epoch_loss += loss.item()
            current_global_step = (
                global_step_offset + epoch * total_steps_per_epoch + batch_idx
            )

            # Logging
            if (
                batch_idx + 1
            ) % log_interval == 0 or batch_idx == total_steps_per_epoch - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.6f}")
                metrics = {
                    f"layer{proj_matrix_idx}_train_loss_batch": avg_loss_batch,  # Use M matrix index
                }
                log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / total_steps_per_epoch
        logger.info(
            f"Layer (W{layer_index+1}/M{layer_index}) Epoch {epoch+1}/{epochs} - Average Local Loss: {avg_epoch_loss:.6f}"
        )
        log_metrics(
            {
                f"layer{proj_matrix_idx}_train_loss_epoch": avg_epoch_loss
            },  # Use M matrix index
            step=global_step_offset + (epoch + 1) * total_steps_per_epoch,
            wandb_run=wandb_run,
        )

    logger.info(f"Finished MF training for Layer (W{layer_index+1}/M{layer_index})")
    linear_layer.eval()
    act_layer.eval()


def train_mf_output_layer(
    model: MF_MLP,
    optimizer: optim.Optimizer,
    criterion: nn.Module,  # Usually CrossEntropy
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    get_output_layer_input_fn: Callable[
        [torch.Tensor], torch.Tensor
    ],  # Gets input a_{L-1}
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
) -> None:
    """Trains the final output layer of an MF_MLP using standard supervised objective."""
    output_layer = model.output_layer
    model.eval()  # Ensure hidden layers are frozen and in eval mode
    output_layer.train()  # Set only output layer to train mode
    model.to(device)
    logger.info("Starting MF training for Output Layer")

    total_steps_per_epoch = len(train_loader)
    # Offset steps based on hidden layer training epochs
    global_step_offset = (
        model.num_hidden_layers
        * config.get("algorithm_params", {}).get("epochs_per_layer", 5)
        * total_steps_per_epoch
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        pbar = tqdm(
            train_loader, desc=f"Output Layer Epoch {epoch+1}/{epochs}", leave=False
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # 1. Get input for the output layer (output of last hidden layer a_{L-1}) - DETACHED
            with torch.no_grad():
                output_layer_input = get_output_layer_input_fn(images).detach()

            # 2. Forward pass through the output layer (tracks gradients)
            predictions = output_layer(output_layer_input)

            # 3. Calculate standard supervised loss
            loss = criterion(predictions, labels)

            # 4. Backpropagate and optimize (only for the output layer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics and Logging
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
                    "output_layer_train_loss_batch": loss.item(),
                    "output_layer_train_acc_batch": batch_accuracy,
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
                "output_layer_train_loss_epoch": avg_epoch_loss,
                "output_layer_train_acc_epoch": avg_epoch_accuracy,
            },
            step=global_step_offset + (epoch + 1) * total_steps_per_epoch,
            wandb_run=wandb_run,
        )
    logger.info("Finished MF training for Output Layer")
    output_layer.eval()  # Set output layer to eval mode


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

    # --- Get Training Configuration ---
    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    optimizer_params = algo_config.get("optimizer_params", {"lr": 0.001})
    output_criterion_name = algo_config.get("output_criterion", "CrossEntropyLoss")
    epochs_per_layer = algo_config.get("epochs_per_layer", 5)
    epochs_output_layer = algo_config.get("epochs_output_layer", 10)
    log_interval = algo_config.get("log_interval", 100)

    # Local loss for hidden layers is always CrossEntropy
    mf_criterion = nn.CrossEntropyLoss()

    # Output layer loss
    if output_criterion_name.lower() == "crossentropyloss":
        output_criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported output criterion: {output_criterion_name}")

    # --- Input Function for Layer 0 (provides a_0) ---
    def get_layer0_input(img_batch):
        adapted_input = input_adapter(img_batch) if input_adapter else img_batch
        return adapted_input.to(device)

    current_layer_input_fn = get_layer0_input

    # --- Train Hidden Layers (W_i, M_{i-1}) ---
    for i in range(num_hidden_layers):  # i goes from 0 to L-1
        logger.info(f"--- Training Hidden Layer W_{i+1} / Matrix M_{i} ---")
        linear_layer = model.layers[i * 2]  # W_{i+1}
        act_layer = model.layers[i * 2 + 1]  # sigma_{i+1}
        projection_matrix = model.get_projection_matrix(i)  # M_i

        # Parameters for this stage: W_{i+1} and M_i
        params_to_optimize = list(linear_layer.parameters()) + [projection_matrix]

        # Create optimizer for W_{i+1} and M_i
        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(params_to_optimize, **optimizer_params)
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(params_to_optimize, **optimizer_params)
        elif optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(params_to_optimize, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Train the current hidden layer W_{i+1} and M_i
        train_mf_hidden_layer(
            model=model,
            layer_index=i,  # Pass index i
            optimizer=optimizer,
            criterion=mf_criterion,
            train_loader=train_loader,
            epochs=epochs_per_layer,
            device=device,
            get_layer_input_fn=current_layer_input_fn,  # Provides a_i
            wandb_run=wandb_run,
            log_interval=log_interval,
        )

        # Freeze the trained parameters (W_{i+1} and M_i)
        for param in params_to_optimize:
            param.requires_grad = False
        linear_layer.eval()
        act_layer.eval()

        # --- Define the input function for the *next* layer (provides a_{i+1}) ---
        def create_next_input_fn(trained_layer_idx: int):
            # trained_layer_idx is 'i' from the loop (0 to L-1)
            lin_layer_k = model.layers[trained_layer_idx * 2]  # W_{i+1}
            act_layer_k = model.layers[trained_layer_idx * 2 + 1]  # sigma_{i+1}
            # Need the input function that yielded a_i for this stage
            prev_input_fn = current_layer_input_fn

            @torch.no_grad()
            def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                # Get input a_i
                a_prev = prev_input_fn(img_batch)
                # Compute output a_{i+1} = sigma( a_i @ W_{i+1} )
                a_current = act_layer_k(lin_layer_k(a_prev))
                return a_current

            return next_input_fn

        current_layer_input_fn = create_next_input_fn(
            i
        )  # Function now provides a_{i+1}

    # --- Train Output Layer (W_L) ---
    logger.info(f"--- Training Output Layer ---")
    output_layer = model.output_layer
    output_params = list(output_layer.parameters())

    if not output_params:
        logger.warning("Output layer has no parameters to optimize.")
        output_optimizer = None
    else:
        if optimizer_name.lower() == "adam":
            output_optimizer = optim.Adam(output_params, **optimizer_params)
        elif optimizer_name.lower() == "sgd":
            output_optimizer = optim.SGD(output_params, **optimizer_params)
        elif optimizer_name.lower() == "adamw":
            output_optimizer = optim.AdamW(output_params, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # The input function needed provides a_{L-1} (output of last hidden layer)
    # This is `current_layer_input_fn` after the loop finishes.
    if output_optimizer:
        train_mf_output_layer(
            model=model,
            optimizer=output_optimizer,
            criterion=output_criterion,
            train_loader=train_loader,
            epochs=epochs_output_layer,
            device=device,
            get_output_layer_input_fn=current_layer_input_fn,  # Provides a_{L-1}
            wandb_run=wandb_run,
            log_interval=log_interval,
        )
    else:
        logger.warning("Skipping output layer training.")

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

            predictions = model.forward(eval_input)  # Use standard forward pass

            if criterion:
                try:
                    loss = criterion(predictions, labels)
                    total_loss += loss.item() * images.size(0)
                except Exception as e:
                    logger.warning(f"Failed to compute evaluation loss: {e}")
                    total_loss = float("nan")  # Mark loss as invalid

            predicted_labels = torch.argmax(predictions, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            # Optional: Update pbar postfix
            # current_acc = (total_correct / total_samples * 100.0 if total_samples > 0 else 0.0)
            # pbar.set_postfix(acc=f"{current_acc:.2f}%")

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


# Removed the __main__ block for cleaner algorithm file
