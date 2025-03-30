# File: src/algorithms/mf.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import logging
from tqdm import tqdm  # For progress bars
from typing import Dict, Any, Optional, Callable, List, Tuple

# Assuming architectures and utils are accessible via src package
from src.architectures.mf_mlp import MF_MLP  # Use the corrected MF_MLP
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics

logger = logging.getLogger(__name__)


def mf_local_loss_fn(
    activation_i: torch.Tensor,  # a_i (output of layer i)
    projection_matrix_i: nn.Parameter,  # M_i
    targets: torch.Tensor,  # Ground truth labels (indices)
    criterion: nn.Module = nn.CrossEntropyLoss(),
) -> torch.Tensor:
    """
    Calculates the Mono-Forward local cross-entropy loss for layer i.
    Loss = CE(softmax(a_i @ M_i^T), y)

    Args:
        activation_i: Activation output of the current hidden layer i (shape: [batch_size, num_neurons_i]).
        projection_matrix_i: Learnable projection matrix M_i for layer i (shape: [num_classes, num_neurons_i]).
        targets: Ground truth labels (shape: [batch_size]).
        criterion: The loss function (should be CrossEntropyLoss).

    Returns:
        The mean local cross-entropy loss for the batch.
    """
    # Calculate goodness scores G_i = a_i @ M_i^T
    # (B, N_i) @ (N_i, C) -> (B, C)  (where N_i = num_neurons_i, C = num_classes)
    goodness_scores_i = torch.matmul(activation_i, projection_matrix_i.t())

    # Calculate cross-entropy loss. nn.CrossEntropyLoss expects raw logits (goodness scores)
    # and target indices. It applies softmax internally.
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
    ],  # Function to get input a_{i-1}
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
) -> None:
    """
    Trains a single hidden layer 'i' of an MF_MLP using the local CE loss.
    Updates both the feedforward weights W_i and the projection matrix M_i.
    Assumes previous layers (0 to i-1) are frozen.

    Args:
        model: The MF_MLP model instance.
        layer_index: The index of the hidden layer being trained (0 means first hidden layer).
        optimizer: PyTorch optimizer containing parameters for W_i and M_i ONLY.
        criterion: CrossEntropyLoss instance.
        train_loader: DataLoader providing batches of original (images, labels).
        epochs: Number of training epochs for this layer.
        device: Device to perform training on ('cuda' or 'cpu').
        get_layer_input_fn: A function that takes original images and returns the
                            detached input tensor a_{i-1} for the *current* layer.
        wandb_run: Optional Weights & Biases run object for logging.
        log_interval: How often to log batch metrics.
    """
    # Identify the modules for this layer
    linear_layer_idx = layer_index * 2
    act_layer_idx = layer_index * 2 + 1
    linear_layer = model.layers[linear_layer_idx]  # W_i layer
    act_layer = model.layers[act_layer_idx]  # Activation for layer i
    projection_matrix = model.get_projection_matrix(layer_index)  # M_i parameter

    # Set only the current layer's components to train mode
    model.eval()  # Put model in eval mode by default
    linear_layer.train()
    act_layer.train()
    # Projection matrix M_i requires_grad should be True by default as it's nn.Parameter
    model.to(device)  # Ensure model is on device

    logger.info(
        f"Starting MF training for Hidden Layer {layer_index + 1} (W_{layer_index+1} and M_{layer_index})"
    )

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"Layer {layer_index+1} Epoch {epoch+1}/{epochs}"
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            # Input images are the original images, labels are ground truth
            images, labels = images.to(device), labels.to(device)

            # 1. Get input for the current layer (a_{i-1}) - DETACHED
            # Input adapter (e.g., flatten) should be handled by get_layer_input_fn
            layer_input_a_prev = get_layer_input_fn(images).detach()

            # 2. Forward pass through the current layer W_i + activation
            # Only operations involving W_i and M_i should track gradients for this optimizer
            pre_activation_z = linear_layer(layer_input_a_prev)  # z_i = a_{i-1} W_i
            activation_a = act_layer(pre_activation_z)  # a_i = sigma(z_i)

            # 3. Calculate local MF loss using a_i and M_i
            loss = mf_local_loss_fn(activation_a, projection_matrix, labels, criterion)

            # 4. Backpropagate and optimize W_i and M_i
            optimizer.zero_grad()
            loss.backward()  # Computes gradients for W_i and M_i w.r.t. local loss
            optimizer.step()  # Updates W_i and M_i

            epoch_loss += loss.item()
            global_step += 1

            # Logging
            if batch_idx % log_interval == 0:
                avg_loss = loss.item()
                pbar.set_postfix(loss=f"{avg_loss:.6f}")
                metrics = {
                    f"layer_{layer_index+1}_train_loss_batch": avg_loss,
                }
                log_metrics(metrics, step=global_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(
            f"Layer {layer_index+1} Epoch {epoch+1}/{epochs} - Average Local Loss: {avg_epoch_loss:.6f}"
        )
        log_metrics(
            {f"layer_{layer_index+1}_train_loss_epoch": avg_epoch_loss},
            step=epoch + 1,
            wandb_run=wandb_run,
        )

    logger.info(f"Finished MF training for Hidden Layer {layer_index + 1}")
    linear_layer.eval()  # Set layer back to eval mode
    act_layer.eval()
    # Note: Projection matrix M_i remains learnable if needed for future layers, but often trained only once


# Keep train_mf_output_layer as it was, it trains the final layer with standard BP logic
# Just ensure the get_output_layer_input_fn provides the correct detached input
# (output of the last hidden layer)
def train_mf_output_layer(
    model: MF_MLP,
    optimizer: optim.Optimizer,
    criterion: nn.Module,  # Usually CrossEntropy for the final layer
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    get_output_layer_input_fn: Callable[
        [torch.Tensor], torch.Tensor
    ],  # Gets input from last hidden layer
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
) -> None:
    """
    Trains the final output layer of an MF_MLP using a standard supervised objective.
    Assumes all hidden layers are frozen. (Implementation mostly unchanged)
    """
    output_layer = model.output_layer
    model.eval()  # Put model in eval mode by default
    output_layer.train()
    model.to(device)
    logger.info("Starting MF training for Output Layer")
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        pbar = tqdm(train_loader, desc=f"Output Layer Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            # 1. Get input for the output layer (output of last hidden layer) - DETACHED
            with torch.no_grad():
                output_layer_input = get_output_layer_input_fn(images).detach()
            # 2. Forward pass through the output layer
            predictions = output_layer(output_layer_input)
            # 3. Calculate standard supervised loss
            loss = criterion(predictions, labels)
            # 4. Backpropagate and optimize (only for the output layer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Metrics and Logging (unchanged)
            batch_accuracy = calculate_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_accuracy += batch_accuracy
            global_step += 1
            if batch_idx % log_interval == 0:
                metrics = {
                    "output_layer_train_loss_batch": loss.item(),
                    "output_layer_train_acc_batch": batch_accuracy,
                }
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", acc=f"{batch_accuracy:.2f}%"
                )
                log_metrics(metrics, step=global_step, wandb_run=wandb_run)
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_accuracy = epoch_accuracy / len(train_loader)
        logger.info(
            f"Output Layer Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_accuracy:.2f}%"
        )
        log_metrics(
            {
                "output_layer_train_loss_epoch": avg_epoch_loss,
                "output_layer_train_acc_epoch": avg_epoch_accuracy,
            },
            step=epoch + 1,
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
    input_adapter: Optional[Callable] = None,  # e.g., flatten
):
    """
    Orchestrates the layer-wise training of an MF_MLP model according to the corrected algorithm.

    Args:
        model: The MF_MLP model instance.
        train_loader: DataLoader providing original (images, labels).
        config: Dictionary with training configurations:
            - optimizer: Name of the optimizer (e.g., 'Adam', 'SGD').
            - optimizer_params: Dict of parameters for the optimizer (e.g., {'lr': 0.001}).
            - mf_criterion: Ignored (uses CrossEntropyLoss internally).
            - output_criterion: Loss for output layer (e.g., 'CrossEntropyLoss').
            - epochs_per_layer: Epochs for hidden layers.
            - epochs_output_layer: Epochs for the output layer.
            - log_interval: Frequency of logging batch metrics.
        device: Device to run training on.
        wandb_run: Optional W&B run object.
        input_adapter: Function to adapt input images (e.g., flatten).
    """
    model.to(device)
    num_hidden_layers = model.num_hidden_layers
    logger.info(
        f"Starting layer-wise MF training for {num_hidden_layers} hidden layers + 1 output layer."
    )

    # --- Get Training Configuration ---
    optimizer_name = config.get("optimizer", "Adam")
    optimizer_params = config.get("optimizer_params", {"lr": 0.001})
    # mf_criterion_name = config.get('mf_criterion', 'CrossEntropyLoss') # MF uses CE locally
    output_criterion_name = config.get("output_criterion", "CrossEntropyLoss")
    epochs_per_layer = config.get("epochs_per_layer", 5)
    epochs_output_layer = config.get("epochs_output_layer", 10)
    log_interval = config.get("log_interval", 100)

    # Local loss for hidden layers is always CrossEntropy
    mf_criterion = nn.CrossEntropyLoss()

    # Output layer loss
    if output_criterion_name.lower() == "crossentropyloss":
        output_criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported output criterion: {output_criterion_name}")

    # Define the function to get input for the first hidden layer
    def get_first_layer_input(img_batch):
        if input_adapter:
            return input_adapter(img_batch).to(device)
        else:
            # Assume input is already correct (e.g., flattened)
            return img_batch.to(device)

    current_layer_input_fn = get_first_layer_input

    # --- Train Hidden Layers ---
    for i in range(num_hidden_layers):
        logger.info(f"--- Training Hidden Layer {i+1}/{num_hidden_layers} ---")
        linear_layer = model.layers[i * 2]
        act_layer = model.layers[i * 2 + 1]
        projection_matrix = model.get_projection_matrix(i)

        # Parameters for this layer's optimizer: W_i and M_i
        params_to_optimize = list(linear_layer.parameters()) + [projection_matrix]

        # Create optimizer for W_i and M_i
        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(params_to_optimize, **optimizer_params)
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(params_to_optimize, **optimizer_params)
        elif optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(params_to_optimize, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Train the current hidden layer W_i and M_i
        train_mf_hidden_layer(
            model=model,
            layer_index=i,
            optimizer=optimizer,
            criterion=mf_criterion,  # Local CE Loss
            train_loader=train_loader,
            epochs=epochs_per_layer,
            device=device,
            get_layer_input_fn=current_layer_input_fn,  # Provides a_{i-1}
            wandb_run=wandb_run,
            log_interval=log_interval,
        )

        # Freeze the trained parameters (W_i and M_i)
        for param in params_to_optimize:
            param.requires_grad = False
        linear_layer.eval()
        act_layer.eval()
        # Projection matrix M_i is also frozen after its layer is trained

        # Define the input function for the *next* layer
        # It needs to compute the activation a_i based on input and frozen W_i
        def create_next_input_fn(trained_layer_idx: int):
            # trained_layer_idx is the index of the layer whose W and M were just trained (0-based)
            lin_layer_k = model.layers[trained_layer_idx * 2]
            act_layer_k = model.layers[trained_layer_idx * 2 + 1]
            # Need the input function that yielded input for *this* layer (a_{k-1})
            prev_input_fn = current_layer_input_fn  # Capture from outer scope

            @torch.no_grad()  # Ensure computation is gradient-free
            def next_input_fn(img_batch: torch.Tensor) -> torch.Tensor:
                # Get input a_{k-1}
                a_prev = prev_input_fn(img_batch)
                # Compute output a_k = sigma( a_{k-1} W_k )
                a_current = act_layer_k(lin_layer_k(a_prev))
                return a_current

            return next_input_fn

        # Update the input function for the subsequent iteration
        current_layer_input_fn = create_next_input_fn(i)

    # --- Train Output Layer ---
    logger.info(f"--- Training Output Layer ---")
    output_layer = model.output_layer
    output_params = list(output_layer.parameters())

    # Create optimizer for the output layer ONLY
    if not output_params:
        logger.warning("Output layer has no parameters to optimize.")
        output_optimizer = None  # Or handle appropriately
    elif optimizer_name.lower() == "adam":
        output_optimizer = optim.Adam(output_params, **optimizer_params)
    elif optimizer_name.lower() == "sgd":
        output_optimizer = optim.SGD(output_params, **optimizer_params)
    elif optimizer_name.lower() == "adamw":
        output_optimizer = optim.AdamW(output_params, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # The input function required here is the one that computes the output
    # of the *last* hidden layer (a_{L-1})
    # This is the `current_layer_input_fn` after the loop finishes.
    if output_optimizer:
        train_mf_output_layer(
            model=model,
            optimizer=output_optimizer,
            criterion=output_criterion,  # Standard CE loss
            train_loader=train_loader,
            epochs=epochs_output_layer,
            device=device,
            get_output_layer_input_fn=current_layer_input_fn,  # Provides a_{L-1}
            wandb_run=wandb_run,
            log_interval=log_interval,
        )
    else:
        logger.warning(
            "Skipping output layer training as it has no parameters or optimizer failed."
        )

    logger.info("Finished all layer-wise MF training.")


# Keep evaluate_mf_model as it was - uses standard forward pass for BP-style prediction
def evaluate_mf_model(
    model: MF_MLP,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,  # e.g., CrossEntropyLoss
    input_adapter: Optional[Callable] = None,  # e.g., flatten
) -> Dict[str, float]:
    """
    Evaluates the trained MF_MLP model using its standard forward pass (BP-style).
    (Implementation unchanged)
    """
    model.eval()
    model.to(device)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    logger.info("Evaluating MF model (standard forward pass / BP-style)")
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            eval_input = input_adapter(images) if input_adapter else images
            predictions = model.forward(eval_input)  # Use standard forward
            if criterion:
                loss = criterion(predictions, labels)
                total_loss += loss.item() * images.size(0)
            predicted_labels = torch.argmax(predictions, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            pbar.set_postfix(acc=f"{(total_correct / total_samples * 100):.2f}%")
    avg_loss = total_loss / total_samples if criterion and total_samples > 0 else 0.0
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(
        f"Evaluation Results: Accuracy: {accuracy:.2f}%"
        + (f", Loss: {avg_loss:.4f}" if criterion else "")
    )
    results = {"eval_accuracy": accuracy}
    if criterion:
        results["eval_loss"] = avg_loss
    return results


# --- Update __main__ for testing the new logic ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    print("Testing MF algorithm components (Corrected)...")

    # Setup
    batch_size = 4
    input_dim = 784
    num_classes = 10
    hidden_dims = [100, 50]  # Two hidden layers
    epochs_per_layer = 1
    epochs_output_layer = 1
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy Data (Flat)
    num_batches = 5
    dummy_images_flat = torch.randn(batch_size * num_batches, input_dim)
    dummy_labels = torch.randint(0, num_classes, (batch_size * num_batches,))
    dummy_dataset = TensorDataset(dummy_images_flat, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=batch_size)

    # Input adapter (already flat, but good practice)
    input_adapter_flat = lambda x: x.view(x.shape[0], -1)

    try:
        model = MF_MLP(
            input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes
        ).to(device)
        print("Model Instantiated:")
        # print(model) # Optional: print structure

        # Dummy Config for training orchestrator
        config = {
            "optimizer": "AdamW",
            "optimizer_params": {"lr": lr, "weight_decay": 0.01},
            "output_criterion": "CrossEntropyLoss",
            "epochs_per_layer": epochs_per_layer,
            "epochs_output_layer": epochs_output_layer,
            "log_interval": 2,
        }

        print("\n--- Testing train_mf_model (Corrected Logic) ---")
        train_mf_model(
            model, dummy_loader, config, device, input_adapter=input_adapter_flat
        )
        print("Finished MF model training.")

        # Check if parameters are frozen correctly after training
        print("\nParameter Frozen Status after Training:")
        for i, layer in enumerate(model.layers):
            if isinstance(layer, nn.Linear):
                print(
                    f"  Hidden Linear Layer {i//2}: requires_grad = {next(layer.parameters()).requires_grad}"
                )
                # Hidden layers should be frozen
                assert not next(layer.parameters()).requires_grad
        for i, M in enumerate(model.projection_matrices):
            print(f"  Projection Matrix M_{i}: requires_grad = {M.requires_grad}")
            # Projection matrices should be frozen
            assert not M.requires_grad
        print(
            f"  Output Layer: requires_grad = {next(model.output_layer.parameters()).requires_grad}"
        )
        # Output layer should be frozen after train_mf_output_layer finishes
        assert not next(model.output_layer.parameters()).requires_grad

        print("\n--- Testing evaluate_mf_model (Corrected Model) ---")
        eval_metrics = evaluate_mf_model(
            model,
            dummy_loader,
            device,
            criterion=nn.CrossEntropyLoss(),
            input_adapter=input_adapter_flat,
        )
        print("Evaluation metrics:", eval_metrics)
        assert "eval_accuracy" in eval_metrics

        print("\nMF Algorithm (Corrected) tests seem functional.")

    except Exception as e:
        print(f"\nError during Corrected MF testing: {e}", exc_info=True)
