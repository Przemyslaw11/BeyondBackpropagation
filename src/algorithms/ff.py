# File: src/algorithms/ff.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm  # For progress bars
from typing import Dict, Any, Optional, Tuple, List, Callable  # Added List, Callable
import functools  # For partial function if needed

# Assuming architectures and utils are accessible via src package
from src.architectures.ff_mlp import FF_MLP, FF_Layer  # Import updated FF_MLP

# FF eval does not directly use calculate_accuracy
from src.utils.logging_utils import log_metrics  # For logging during training

logger = logging.getLogger(__name__)


def ff_loss_fn(
    pos_goodness: torch.Tensor, neg_goodness: torch.Tensor, threshold: float
) -> torch.Tensor:
    """
    Calculates the Forward-Forward loss for a layer.
    Encourages goodness of positive samples above threshold and negative samples below.
    Uses a logistic function approach as described in Hinton's paper/code.

    Args:
        pos_goodness: Goodness scores for positive samples (shape: [batch_size]).
        neg_goodness: Goodness scores for negative samples (shape: [batch_size]).
        threshold: The goodness threshold theta.

    Returns:
        The mean loss for the batch.
    """
    # Loss for positive samples: encourage goodness > threshold
    # log(1 + exp(-(pos_goodness - threshold))) -> pushes pos_goodness higher
    loss_pos = F.softplus(
        -(pos_goodness - threshold)
    )  # More stable than log(1+exp(-x))

    # Loss for negative samples: encourage goodness < threshold
    # log(1 + exp(neg_goodness - threshold)) -> pushes neg_goodness lower
    loss_neg = F.softplus(neg_goodness - threshold)  # More stable than log(1+exp(x))

    # Combine and average
    loss = torch.mean(loss_pos + loss_neg)
    return loss


def train_ff_layer(
    layer: FF_Layer,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    get_layer_input_fn: Callable[
        [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ],  # Function to get input for current layer
    threshold: float,
    epochs: int,
    device: torch.device,
    layer_index: int,  # For logging
    wandb_run: Optional[Any] = None,
    log_interval: int = 100,
) -> None:
    """
    Trains a single FF_Layer using the Forward-Forward algorithm.

    Args:
        layer: The FF_Layer module to train.
        optimizer: PyTorch optimizer for this layer's parameters.
        train_loader: DataLoader providing batches of original (images, labels).
        get_layer_input_fn: A function that takes (images, labels) and returns the
                            (pos_input, neg_input) tensors for the *current* layer,
                            already processed by previous frozen layers and detached.
        threshold: Goodness threshold for the loss function.
        epochs: Number of training epochs for this layer.
        device: Device to perform training on ('cuda' or 'cpu').
        layer_index: Index of the current layer (0-based) for logging.
        wandb_run: Optional Weights & Biases run object for logging.
        log_interval: How often to log batch metrics.
    """
    layer.train()
    layer.to(device)
    logger.info(f"Starting FF training for Layer {layer_index + 1}")

    # Calculate total steps for accurate logging if needed later
    total_steps = epochs * len(train_loader)

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"FF Layer {layer_index+1} Epoch {epoch+1}/{epochs}"
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            # Original images/labels used by the input function
            images, labels = images.to(device), labels.to(device)

            # 1. Get input for the current layer (already processed and detached)
            # Input function handles generating pos/neg and passing through frozen layers
            pos_input, neg_input = get_layer_input_fn(images, labels)
            # No need to detach here, input_fn should ensure detachment

            # 2. Forward pass through current layer to get goodness
            _, pos_goodness = layer.forward_with_goodness(pos_input)
            _, neg_goodness = layer.forward_with_goodness(neg_input)

            # 3. Calculate loss
            loss = ff_loss_fn(pos_goodness, neg_goodness, threshold)

            # 4. Backpropagate and optimize (only for the current layer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_step = (
                epoch * len(train_loader) + batch_idx
            )  # Global step within this layer's training

            # Logging
            if batch_idx % log_interval == 0:
                avg_loss = loss.item()
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
                metrics = {
                    f"layer_{layer_index+1}_train_loss_batch": avg_loss,
                    f"layer_{layer_index+1}_pos_goodness_mean": pos_goodness.mean().item(),
                    f"layer_{layer_index+1}_neg_goodness_mean": neg_goodness.mean().item(),
                }
                log_metrics(metrics, step=batch_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(
            f"Layer {layer_index+1} Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}"
        )
        # Log epoch loss against a global epoch number if tracked outside, or vs local epoch
        log_metrics(
            {f"layer_{layer_index+1}_train_loss_epoch": avg_epoch_loss},
            step=epoch + 1,
            wandb_run=wandb_run,
        )

    logger.info(f"Finished FF training for Layer {layer_index + 1}")


# --- NEW: FF Model Training Orchestration ---
def train_ff_model(
    model: FF_MLP,
    train_loader: DataLoader,
    config: Dict[
        str, Any
    ],  # Containing optimizer params, threshold, epochs per layer etc.
    device: torch.device,
    wandb_run: Optional[Any] = None,
    input_adapter: Optional[Callable] = None,  # e.g., flatten
):
    """
    Orchestrates the layer-wise training of an FF_MLP model.

    Args:
        model: The FF_MLP model instance.
        train_loader: DataLoader providing original (images, labels).
        config: Dictionary with training configurations:
            - optimizer_type: Name of the optimizer (e.g., 'Adam').
            - lr: Learning rate for each layer's optimizer.
            - weight_decay: Weight decay (typically 0 for FF).
            - threshold: Goodness threshold theta.
            - epochs_per_layer: Number of epochs to train each layer.
            - log_interval: Frequency of logging batch metrics.
        device: Device to run training on.
        wandb_run: Optional W&B run object.
        input_adapter: Function to adapt original images (e.g., flatten).
    """
    model.to(device)
    num_layers = len(model.layers)
    logger.info(f"Starting layer-wise FF training for {num_layers} layers.")

    # --- Get Training Configuration ---
    optimizer_type = config.get("optimizer_type", "Adam")
    lr = config.get("lr", 0.001)
    weight_decay = config.get("weight_decay", 0.0)
    threshold = config.get("threshold", 1.0)
    epochs_per_layer = config.get("epochs_per_layer", 10)
    log_interval = config.get("log_interval", 100)

    # --- Helper function to generate pos/neg data (using model's method) ---
    def generate_pos_neg_for_ff(base_images, base_labels):
        """Generates FF positive/negative inputs from base images/labels."""
        batch_size = base_images.shape[0]
        device = base_images.device
        num_classes = model.num_classes  # Get from model instance

        # Positive Data
        pos_data = model.prepare_ff_input(base_images, base_labels)

        # Negative Data
        # Ensure random integer is generated correctly on the target device
        rand_offset = torch.randint(
            1, num_classes, (batch_size,), device=device, dtype=torch.long
        )
        neg_labels = (base_labels + rand_offset) % num_classes
        # Simple check for collisions (unlikely for large num_classes)
        collision = neg_labels == base_labels
        if torch.any(collision):
            # Just shift collisions by 1 more (simple fix)
            neg_labels[collision] = (neg_labels[collision] + 1) % num_classes

        neg_data = model.prepare_ff_input(base_images, neg_labels)

        return pos_data, neg_data

    # --- Train Layer by Layer ---
    for i, layer in enumerate(model.layers):
        logger.info(f"--- Training Layer {i+1}/{num_layers} ---")

        # Define the function to get input for the *current* layer 'i'
        # This uses a closure to capture the current layer index 'i'
        def get_layer_i_input_closure(current_layer_idx: int):
            def get_layer_i_input(images, labels):
                # Adapt original images first (e.g., flatten)
                adapted_images = input_adapter(images) if input_adapter else images

                # Generate base positive/negative data with embedded labels
                base_pos_data, base_neg_data = generate_pos_neg_for_ff(
                    adapted_images, labels
                )

                if current_layer_idx == 0:
                    # For the first layer, the input is just the base data
                    return (
                        base_pos_data.detach(),
                        base_neg_data.detach(),
                    )  # Detach for safety
                else:
                    # For subsequent layers, pass base data through frozen layers 0 to i-1
                    with torch.no_grad():  # IMPORTANT: No gradients through previous layers
                        pos_input_i = model.forward_upto(
                            base_pos_data, current_layer_idx
                        )  # Run layers 0..i-1
                        neg_input_i = model.forward_upto(
                            base_neg_data, current_layer_idx
                        )  # Run layers 0..i-1
                    return (
                        pos_input_i.detach(),
                        neg_input_i.detach(),
                    )  # Return detached activations

            return get_layer_i_input

        get_layer_input_fn = get_layer_i_input_closure(i)

        # Create optimizer for the current layer *only*
        params_to_optimize = list(layer.parameters())
        if not params_to_optimize:
            logger.warning(
                f"Layer {i+1} has no parameters to optimize. Skipping training."
            )
            continue

        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            # Common to use momentum with SGD
            sgd_momentum = config.get("momentum", 0.9)
            optimizer = optim.SGD(
                params_to_optimize,
                lr=lr,
                weight_decay=weight_decay,
                momentum=sgd_momentum,
            )
        elif optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(
                params_to_optimize, lr=lr, weight_decay=weight_decay
            )
        else:
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

        # Freeze the trained layer
        for param in layer.parameters():
            param.requires_grad = False
        layer.eval()  # Set to eval mode after training

    logger.info("Finished all layer-wise FF training.")


# --- NEW: FF Model Evaluation ---
def evaluate_ff_model(
    model: FF_MLP,
    data_loader: DataLoader,  # Validation or Test loader
    device: torch.device,
    input_adapter: Optional[Callable] = None,  # e.g., flatten
) -> Dict[str, float]:
    """
    Evaluates the trained FF_MLP model using multi-pass inference.

    Args:
        model: The trained FF_MLP model (layers should be frozen).
        data_loader: DataLoader for the evaluation dataset.
        device: Device to run evaluation on.
        input_adapter: Function to adapt original images (e.g., flatten).

    Returns:
        Dictionary containing evaluation metrics (e.g., 'eval_accuracy').
    """
    model.eval()
    model.to(device)
    num_classes = model.num_classes
    logger.info(
        f"Evaluating FF model using multi-pass inference ({num_classes} passes per image)."
    )

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating FF Model")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]

            # Adapt original images (e.g., flatten)
            adapted_images = input_adapter(images) if input_adapter else images

            # Store summed goodness for each class for each image in the batch
            batch_total_goodness = torch.zeros((batch_size, num_classes), device=device)

            # Iterate through each possible class label candidate
            for label_candidate in range(num_classes):
                # Prepare input with the candidate label embedded
                candidate_labels = torch.full(
                    (batch_size,), label_candidate, dtype=torch.long, device=device
                )
                ff_input_candidate = model.prepare_ff_input(
                    adapted_images, candidate_labels
                )

                # Get goodness per layer for this candidate input
                try:
                    layer_goodness_list = model.forward_goodness_per_layer(
                        ff_input_candidate
                    )  # List of tensors [B]
                except Exception as e:
                    logger.error(
                        f"Error during FF goodness forward pass for candidate {label_candidate}: {e}",
                        exc_info=True,
                    )
                    # Handle error gracefully, e.g., assign very low goodness
                    total_goodness_candidate = torch.full(
                        (batch_size,), -float("inf"), device=device
                    )
                    batch_total_goodness[:, label_candidate] = total_goodness_candidate
                    continue  # Skip to next candidate

                # Sum goodness across layers for this candidate
                # Check if list is empty (shouldn't happen with >=1 layer)
                if not layer_goodness_list:
                    logger.warning("Received empty goodness list during FF evaluation.")
                    total_goodness_candidate = torch.zeros((batch_size,), device=device)
                else:
                    try:
                        # Stack list of [B] tensors into [L, B], then sum over L (dim=0)
                        total_goodness_candidate = torch.stack(
                            layer_goodness_list, dim=0
                        ).sum(dim=0)
                    except Exception as e:
                        logger.error(
                            f"Error stacking/summing goodness scores: {e}",
                            exc_info=True,
                        )
                        total_goodness_candidate = torch.full(
                            (batch_size,), -float("inf"), device=device
                        )

                batch_total_goodness[:, label_candidate] = total_goodness_candidate

            # Predict the label with the highest total goodness for each image
            try:
                predicted_labels = torch.argmax(batch_total_goodness, dim=1)
            except Exception as e:
                logger.error(f"Error during argmax prediction: {e}", exc_info=True)
                # Handle error, e.g., predict class 0
                predicted_labels = torch.zeros_like(labels)

            # Calculate accuracy for the batch
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

            current_acc = (
                (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
            )
            pbar.set_postfix(acc=f"{current_acc:.2f}%")

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"FF Evaluation Accuracy: {accuracy:.2f}%")

    results = {
        # Use a consistent key name like other algorithms
        "eval_accuracy": accuracy
    }
    return results


# --- Update __main__ block for testing ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    print("Testing FF algorithm components (including orchestration)...")

    # Setup
    batch_size = 4
    input_dim = 784
    num_classes = 10
    hidden_dims = [100, 50]  # Small MLP for testing
    threshold = 1.5
    epochs_per_layer = 1  # Minimal epochs
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy Data (Flat input expected by adapter)
    num_batches = 5
    dummy_images = torch.randn(batch_size * num_batches, input_dim)
    dummy_labels = torch.randint(0, num_classes, (batch_size * num_batches,))
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=batch_size)

    # Input adapter (identity since data is already flat for MLP)
    input_adapter_flat = lambda x: x

    # Dummy Model
    try:
        model = FF_MLP(
            input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes
        ).to(device)
        print("\n--- FF Model Instance ---")
        # print(model)

        # Dummy Training Config (matching args for train_ff_model)
        # Use keys expected by train_ff_model
        config_ff = {
            "optimizer_type": "AdamW",  # Use AdamW example
            "lr": lr,
            "weight_decay": 0.01,  # Example WD
            "threshold": threshold,
            "epochs_per_layer": epochs_per_layer,
            "log_interval": 2,
        }

        print("\n--- Testing train_ff_model ---")
        train_ff_model(
            model=model,
            train_loader=dummy_loader,
            config=config_ff,  # Pass the config dict
            device=device,
            input_adapter=input_adapter_flat,
        )
        print("Finished FF model training simulation.")

        # Verify layers are frozen
        print("\nChecking layer freeze status:")
        for i, layer in enumerate(model.layers):
            frozen = not any(p.requires_grad for p in layer.parameters())
            print(f"  Layer {i+1} frozen: {frozen}")
            assert frozen  # All layers should be frozen after training

        print("\n--- Testing evaluate_ff_model ---")
        eval_metrics = evaluate_ff_model(
            model=model,
            data_loader=dummy_loader,  # Evaluate on same dummy data for test
            device=device,
            input_adapter=input_adapter_flat,
        )
        print("Evaluation metrics:", eval_metrics)
        assert "eval_accuracy" in eval_metrics

        print("\nFF Algorithm orchestration tests seem functional.")

    except Exception as e:
        print(f"\nError during FF testing: {e}", exc_info=True)
