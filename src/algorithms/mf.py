import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import logging
from tqdm import tqdm
from typing import Dict, Any, Optional, Callable, List, Tuple
import os  # For checkpointing
import time # For step timing

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


def train_mf_matrix_only(
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
    max_step_tracker: Dict[str, int] = {"value": 0}, # Use dict for mutable tracking
) -> None:
    """
    Trains a single projection matrix (specifically M0) using local loss based on its corresponding activation (a0).
    """
    if matrix_index != 0:
        logger.warning(f"train_mf_matrix_only called for index {matrix_index}, expected 0.")

    projection_matrix = model.get_projection_matrix(matrix_index)  # M_0
    projection_matrix.requires_grad_(True)
    model.to(device)
    model.eval()

    logger.info(f"Starting MF training for Projection Matrix M_{matrix_index}")

    total_steps_per_epoch = len(train_loader)
    global_step_offset = 0 # M0 training happens first

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"M{matrix_index} Epoch {epoch+1}/{epochs}", leave=False
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            activation_a_i = get_matrix_input_fn(images).detach() # Get a_0
            loss = mf_local_loss_fn(activation_a_i, projection_matrix, labels, criterion)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss at M{matrix_index}, Epoch {epoch+1}, Batch {batch_idx}. Stop.")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            current_global_step = (
                global_step_offset + epoch * total_steps_per_epoch + batch_idx + 1 # 1-based batch index
            )
            max_step_tracker["value"] = max(max_step_tracker["value"], current_global_step) # Update tracker

            if (batch_idx + 1) % log_interval == 0 or batch_idx == total_steps_per_epoch - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.6f}")
                metrics = {f"Layer_M{matrix_index}/Train_Loss_Batch": avg_loss_batch}
                log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)):
            logger.error(f"Terminating M{matrix_index} training due to invalid loss.")
            break

        avg_epoch_loss = epoch_loss / total_steps_per_epoch if total_steps_per_epoch > 0 else 0.0
        epoch_end_step = global_step_offset + (epoch + 1) * total_steps_per_epoch
        max_step_tracker["value"] = max(max_step_tracker["value"], epoch_end_step) # Update tracker

        logger.info(f"Matrix M{matrix_index} Epoch {epoch+1}/{epochs} - Avg Local Loss: {avg_epoch_loss:.6f}")
        log_metrics(
            {f"Layer_M{matrix_index}/Train_Loss_Epoch": avg_epoch_loss},
            step=epoch_end_step, # Log at correct end step
            wandb_run=wandb_run,
        )

    logger.info(f"Finished MF training for Matrix M_{matrix_index}")
    projection_matrix.requires_grad_(False)


def train_mf_layer(
    model: MF_MLP,
    layer_index: int,  # Index i=0..L-1. Trains W_{i+1} and M_{i+1}.
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    get_layer_input_fn: Callable[[torch.Tensor], torch.Tensor],  # Function to get a_i
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
    max_step_tracker: Dict[str, int] = {"value": 0}, # Use dict for mutable tracking
) -> None:
    """
    Trains a single layer 'i+1' (W_{i+1} and M_{i+1}) of an MF_MLP using local loss.
    """
    linear_layer_idx = layer_index * 2
    act_layer_idx = layer_index * 2 + 1
    proj_matrix_idx = layer_index + 1
    w_layer_log_idx = layer_index + 1
    m_matrix_log_idx = proj_matrix_idx

    linear_layer = model.layers[linear_layer_idx]  # W_{i+1}
    act_layer = model.layers[act_layer_idx]  # sigma_{i+1}
    projection_matrix = model.get_projection_matrix(proj_matrix_idx)  # M_{i+1}

    model.eval()
    linear_layer.train()
    act_layer.train()
    projection_matrix.requires_grad_(True)
    model.to(device)

    logger.info(f"Starting MF training for Layer W_{w_layer_log_idx} / Matrix M_{m_matrix_log_idx}")

    total_steps_per_epoch = len(train_loader)
    # Correct offset: Accounts for M0 training AND previous W/M layers
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
            prev_activation_a_i = get_layer_input_fn(images).detach() # Get a_i
            pre_activation_z_next = linear_layer(prev_activation_a_i) # z_{i+1}
            activation_a_next = act_layer(pre_activation_z_next) # a_{i+1}
            loss = mf_local_loss_fn(activation_a_next, projection_matrix, labels, criterion)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss at W{w_layer_log_idx}/M{m_matrix_log_idx}, Epoch {epoch+1}, Batch {batch_idx}. Stop.")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # Updates W_{i+1} and M_{i+1}

            epoch_loss += loss.item()
            current_global_step = (
                global_step_offset + epoch * total_steps_per_epoch + batch_idx + 1 # 1-based batch index
            )
            max_step_tracker["value"] = max(max_step_tracker["value"], current_global_step) # Update tracker

            if (batch_idx + 1) % log_interval == 0 or batch_idx == total_steps_per_epoch - 1:
                avg_loss_batch = loss.item()
                pbar.set_postfix(loss=f"{avg_loss_batch:.6f}")
                metrics = {f"Layer_M{m_matrix_log_idx}/Train_Loss_Batch": avg_loss_batch}
                log_metrics(metrics, step=current_global_step, wandb_run=wandb_run)

        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)):
            logger.error(f"Terminating W{w_layer_log_idx}/M{m_matrix_log_idx} training due to invalid loss.")
            break

        avg_epoch_loss = epoch_loss / total_steps_per_epoch if total_steps_per_epoch > 0 else 0.0
        epoch_end_step = global_step_offset + (epoch + 1) * total_steps_per_epoch
        max_step_tracker["value"] = max(max_step_tracker["value"], epoch_end_step) # Update tracker

        logger.info(f"Layer (W{w_layer_log_idx}/M{m_matrix_log_idx}) Epoch {epoch+1}/{epochs} - Avg Local Loss: {avg_epoch_loss:.6f}")
        log_metrics(
            {f"Layer_M{m_matrix_log_idx}/Train_Loss_Epoch": avg_epoch_loss},
            step=epoch_end_step, # Log at correct end step
            wandb_run=wandb_run,
        )

    logger.info(f"Finished MF training for Layer (W{w_layer_log_idx}/M{m_matrix_log_idx})")
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
    max_step_tracker: Dict[str, int] = {"value": 0}, # Pass down tracker
):
    """
    Orchestrates the layer-wise training of an MF_MLP model.
    """
    model.to(device)
    num_hidden_layers = model.num_hidden_layers
    logger.info(f"Starting layer-wise MF training for M0 and {num_hidden_layers} hidden layers.")

    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    # Correctly access nested optimizer params
    optimizer_params_config = algo_config  # Use algo_config directly for LR/WD
    lr = optimizer_params_config.get("lr", 0.001)
    weight_decay = optimizer_params_config.get("weight_decay", 0.0)
    # Extract other params from the MAIN algo_config level if needed, not nested optimizer_params
    optimizer_extra_kwargs = {} # Example: {k: v for k, v in algo_config.items() if k not in ['lr', 'weight_decay', 'optimizer_type', 'epochs_per_layer', etc...]}

    epochs_per_layer = algo_config.get("epochs_per_layer", 5)
    log_interval = algo_config.get("log_interval", 100)
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None)
    mf_criterion = nn.CrossEntropyLoss()

    def get_a0_input(img_batch):
        adapted_input = input_adapter(img_batch) if input_adapter else img_batch
        return adapted_input.to(device)

    logger.info("--- Training Projection Matrix M_0 ---")
    projection_matrix_0 = model.get_projection_matrix(0)
    optimizer_0_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_extra_kwargs}
    params_m0 = [projection_matrix_0]
    if not any(p.requires_grad for p in params_m0):
        logger.warning("M_0 does not require gradients. Skipping training.")
    else:
        optimizer_0 = getattr(optim, optimizer_name)(params_m0, **optimizer_0_kwargs)
        train_mf_matrix_only(
            model=model, matrix_index=0, optimizer=optimizer_0, criterion=mf_criterion,
            train_loader=train_loader, epochs=epochs_per_layer, device=device,
            get_matrix_input_fn=get_a0_input, wandb_run=wandb_run, log_interval=log_interval,
            max_step_tracker=max_step_tracker, # Pass tracker
        )
        if checkpoint_dir:
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained_index": -1},
                is_best=False, filename="mf_matrix_M0_complete.pth", checkpoint_dir=checkpoint_dir,
            )

    current_layer_input_fn = get_a0_input

    for i in range(num_hidden_layers):
        w_layer_log_idx = i + 1
        m_matrix_log_idx = i + 1
        logger.info(f"--- Training Hidden Layer W_{w_layer_log_idx} / Matrix M_{m_matrix_log_idx} ---")

        linear_layer = model.layers[i * 2]
        projection_matrix = model.get_projection_matrix(i + 1)
        params_to_optimize = list(linear_layer.parameters()) + [projection_matrix]

        if not any(p.requires_grad for p in params_to_optimize):
            logger.warning(f"Skipping W{w_layer_log_idx}/M{m_matrix_log_idx}: No grads.")
            def create_next_input_fn_skip(trained_layer_idx: int, previous_input_fn: Callable):
                lin_layer_k, act_layer_k = model.layers[trained_layer_idx*2], model.layers[trained_layer_idx*2+1]
                lin_layer_k.eval(); act_layer_k.eval()
                @torch.no_grad()
                def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                    a_prev = previous_input_fn(img_batch); a_current = act_layer_k(lin_layer_k(a_prev))
                    return a_current.detach()
                return next_input_fn
            current_layer_input_fn = create_next_input_fn_skip(i, current_layer_input_fn)
            continue

        optimizer_i_kwargs = {"lr": lr, "weight_decay": weight_decay, **optimizer_extra_kwargs}
        optimizer = getattr(optim, optimizer_name)(params_to_optimize, **optimizer_i_kwargs)

        train_mf_layer(
            model=model, layer_index=i, optimizer=optimizer, criterion=mf_criterion,
            train_loader=train_loader, epochs=epochs_per_layer, device=device,
            get_layer_input_fn=current_layer_input_fn, wandb_run=wandb_run, log_interval=log_interval,
            max_step_tracker=max_step_tracker, # Pass tracker
        )

        for param in params_to_optimize: param.requires_grad = False
        linear_layer.eval(); model.layers[i * 2 + 1].eval()

        if checkpoint_dir:
            chkpt_filename = f"mf_hidden_layer_W{w_layer_log_idx}_M{m_matrix_log_idx}_complete.pth"
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained_index": i},
                is_best=False, filename=chkpt_filename, checkpoint_dir=checkpoint_dir,
            )

        def create_next_input_fn(trained_layer_idx: int, previous_input_fn: Callable):
            lin_layer_k, act_layer_k = model.layers[trained_layer_idx*2], model.layers[trained_layer_idx*2+1]
            lin_layer_k.eval(); act_layer_k.eval()
            @torch.no_grad()
            def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                a_prev = previous_input_fn(img_batch); a_current = act_layer_k(lin_layer_k(a_prev))
                return a_current.detach()
            return next_input_fn
        current_layer_input_fn = create_next_input_fn(i, current_layer_input_fn)

    logger.info("Finished all layer-wise MF training.")
    model.eval()


def evaluate_mf_model(
    model: MF_MLP,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    input_adapter: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Evaluates the trained MF_MLP model using the paper's "BP-style" approach.
    """
    model.eval()
    model.to(device)
    total_correct, total_samples = 0, 0
    num_hidden_layers = model.num_hidden_layers
    last_activation_index = num_hidden_layers
    last_projection_matrix_index = num_hidden_layers

    logger.info(
        f"Evaluating MF model using a_{last_activation_index} and M_{last_projection_matrix_index}."
    )

    if last_projection_matrix_index >= len(model.projection_matrices):
        logger.error(f"Index M_{last_projection_matrix_index} out of bounds.")
        return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}

    last_projection_matrix = model.get_projection_matrix(last_projection_matrix_index)

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating MF (BP-style)", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            eval_input = input_adapter(images) if input_adapter else images
            all_activations = model.forward_with_intermediate_activations(eval_input)

            if len(all_activations) <= last_activation_index:
                logger.error(f"Activation list len ({len(all_activations)}) too short for a_{last_activation_index}.")
                continue

            last_hidden_activation = all_activations[last_activation_index]
            goodness_scores = torch.matmul(last_hidden_activation, last_projection_matrix.t())
            predicted_labels = torch.argmax(goodness_scores, dim=1)

            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"MF Evaluation Results (BP-style): Accuracy: {accuracy:.2f}%")
    results = {"eval_accuracy": accuracy, "eval_loss": float("nan")}
    return results