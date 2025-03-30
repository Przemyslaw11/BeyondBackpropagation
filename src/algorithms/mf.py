import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import logging
from tqdm import tqdm # For progress bars
from typing import Dict, Any, Optional, Callable, List, Tuple

# Assuming architectures and utils are accessible via src package
from src.architectures.mf_mlp import MF_MLP
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics

logger = logging.getLogger(__name__)

def mf_loss_fn(
    layer_output_projected: torch.Tensor,
    target_projected: torch.Tensor,
    criterion: nn.Module = nn.MSELoss() # Typically MSE loss in projection space
) -> torch.Tensor:
    """
    Calculates the Mono-Forward loss for a layer.
    Minimizes the distance between the projected layer output and the projected target.

    Args:
        layer_output_projected: Projected output of the current layer (M_i * h_i).
                                Shape: [batch_size, projection_dim].
        target_projected: Projected target label (M_L * y_one_hot or similar).
                          Shape: [batch_size, projection_dim].
        criterion: The loss function to use (default: MSELoss).

    Returns:
        The mean loss for the batch.
    """
    loss = criterion(layer_output_projected, target_projected)
    return loss


def train_mf_layer(
    model: MF_MLP, # Pass the whole model to access projection matrices
    layer_index: int, # Index of the hidden layer to train (0-based)
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader, # Loader providing (images, labels)
    epochs: int,
    device: torch.device,
    get_layer_input_fn: Callable[[torch.Tensor], torch.Tensor], # Function to get input for the current layer
    target_projection_matrix: torch.Tensor, # M_L used for projecting targets
    wandb_run: Optional[Any] = None, # For logging
    log_interval: int = 100
) -> None:
    """
    Trains a single hidden layer of an MF_MLP using the Mono-Forward objective.
    Assumes previous layers (0 to layer_index-1) are frozen.

    Args:
        model: The MF_MLP model instance.
        layer_index: The index of the hidden layer being trained (0 corresponds to the first hidden layer).
        optimizer: PyTorch optimizer for the parameters of the *current* layer only.
        criterion: Loss function (e.g., nn.MSELoss) for the projection space.
        train_loader: DataLoader providing batches of original (images, labels).
        epochs: Number of training epochs for this layer.
        device: Device to perform training on ('cuda' or 'cpu').
        get_layer_input_fn: A function that takes a batch of original images
                            and returns the input tensor for the *current* layer
                            (output of previous frozen layers, or original image).
        target_projection_matrix: The projection matrix M_L used for the target labels.
        wandb_run: Optional Weights & Biases run object for logging.
        log_interval: How often to log batch metrics.
    """
    # Identify the actual nn.Linear and nn.Activation modules for this layer
    # Layer indices in model.layers: 0,1 (layer 0), 2,3 (layer 1), etc.
    linear_layer_idx = layer_index * 2
    act_layer_idx = layer_index * 2 + 1
    if linear_layer_idx >= len(model.layers) or act_layer_idx >= len(model.layers):
         raise IndexError(f"Layer index {layer_index} out of bounds for model with {len(model.layers)//2} hidden layers.")

    linear_layer = model.layers[linear_layer_idx]
    act_layer = model.layers[act_layer_idx]

    # Set only the current layer to train mode
    model.eval() # Put model in eval mode by default
    linear_layer.train()
    act_layer.train() # Activation might have state (though ReLU/Tanh usually don't)
    model.to(device) # Ensure model is on device

    # Get the projection matrix for this layer's output (M_i)
    projection_matrix_i = model.get_projection_matrix(layer_index + 1).to(device) # M_{i+1} projects h_{i+1}

    logger.info(f"Starting MF training for Hidden Layer {layer_index + 1}")

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Layer {layer_index+1} Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # 1. Get input for the current layer using the provided function
            with torch.no_grad(): # Ensure previous layer computation doesn't track gradients
                 layer_input = get_layer_input_fn(images)

            # 2. Forward pass through the current layer (Linear + Activation)
            # Input is already detached from previous layers
            pre_activation = linear_layer(layer_input)
            layer_output = act_layer(pre_activation) # h_{i+1}

            # 3. Project layer output using M_{i+1}
            # Output shape: [batch_size, hidden_dim] -> [batch_size, projection_dim]
            # M_{i+1} shape: [projection_dim, hidden_dim]
            layer_output_projected = torch.matmul(layer_output, projection_matrix_i.t()) # (B, d_h) @ (d_h, d_p) -> (B, d_p)

            # 4. Project target labels using M_L
            with torch.no_grad():
                 # Create one-hot labels
                 labels_one_hot = F.one_hot(labels, num_classes=model.num_classes).float()
                 # Project targets: M_L * y
                 # M_L shape: [projection_dim, num_classes]
                 target_projected = torch.matmul(labels_one_hot, target_projection_matrix.t()) # (B, C) @ (C, d_p) -> (B, d_p)

            # 5. Calculate MF loss
            loss = mf_loss_fn(layer_output_projected, target_projected, criterion)

            # 6. Backpropagate and optimize (only for the current layer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # Logging
            if batch_idx % log_interval == 0:
                avg_loss = loss.item()
                pbar.set_postfix(loss=f"{avg_loss:.6f}")
                metrics = {
                    f'layer_{layer_index+1}_train_loss_batch': avg_loss,
                }
                log_metrics(metrics, step=global_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Layer {layer_index+1} Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.6f}")
        log_metrics({f'layer_{layer_index+1}_train_loss_epoch': avg_epoch_loss}, step=epoch+1, wandb_run=wandb_run)

    logger.info(f"Finished MF training for Hidden Layer {layer_index + 1}")
    linear_layer.eval() # Set layer back to eval mode
    act_layer.eval()


def train_mf_output_layer(
    model: MF_MLP, # Pass the whole model
    optimizer: optim.Optimizer,
    criterion: nn.Module, # Usually CrossEntropy for the final layer
    train_loader: DataLoader, # Loader providing (images, labels)
    epochs: int,
    device: torch.device,
    get_output_layer_input_fn: Callable[[torch.Tensor], torch.Tensor], # Gets input from last hidden layer
    wandb_run: Optional[Any] = None, # For logging
    log_interval: int = 100
) -> None:
    """
    Trains the final output layer of an MF_MLP using a standard supervised objective.
    Assumes all hidden layers are frozen.

    Args:
        model: The MF_MLP model instance.
        optimizer: PyTorch optimizer for the parameters of the *output layer* only.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        train_loader: DataLoader providing batches of original (images, labels).
        epochs: Number of training epochs for the output layer.
        device: Device to perform training on ('cuda' or 'cpu').
        get_output_layer_input_fn: Function that takes original images and returns the
                                   activations of the last hidden layer.
        wandb_run: Optional Weights & Biases run object for logging.
        log_interval: How often to log batch metrics.
    """
    output_layer = model.output_layer

    # Set only the output layer to train mode
    model.eval() # Put model in eval mode by default
    output_layer.train()
    model.to(device) # Ensure model is on device

    logger.info("Starting MF training for Output Layer")

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        pbar = tqdm(train_loader, desc=f"Output Layer Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # 1. Get input for the output layer using the provided function
            with torch.no_grad(): # Ensure hidden layer computation doesn't track gradients
                 output_layer_input = get_output_layer_input_fn(images)

            # 2. Forward pass through the output layer
            predictions = output_layer(output_layer_input.detach()) # Detach just in case

            # 3. Calculate standard supervised loss
            loss = criterion(predictions, labels)

            # 4. Backpropagate and optimize (only for the output layer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate batch accuracy
            batch_accuracy = calculate_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_accuracy += batch_accuracy
            global_step += 1

            # Logging
            if batch_idx % log_interval == 0:
                avg_loss = loss.item()
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{batch_accuracy:.2f}%")
                metrics = {
                    f'output_layer_train_loss_batch': avg_loss,
                    f'output_layer_train_acc_batch': batch_accuracy,
                }
                log_metrics(metrics, step=global_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_accuracy = epoch_accuracy / len(train_loader)
        logger.info(f"Output Layer Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_accuracy:.2f}%")
        log_metrics({
            f'output_layer_train_loss_epoch': avg_epoch_loss,
            f'output_layer_train_acc_epoch': avg_epoch_accuracy
        }, step=epoch+1, wandb_run=wandb_run)

    logger.info("Finished MF training for Output Layer")
    output_layer.eval() # Set layer back to eval mode


def train_mf_model(
    model: MF_MLP,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: Optional[Any] = None
):
    """
    Orchestrates the layer-wise training of an MF_MLP model.

    Args:
        model: The MF_MLP model instance.
        train_loader: DataLoader providing original (images, labels).
        config: Dictionary with training configurations:
            - optimizer: Name of the optimizer (e.g., 'Adam', 'SGD').
            - optimizer_params: Dict of parameters for the optimizer (e.g., {'lr': 0.001}).
            - mf_criterion: Loss for hidden layers (e.g., 'MSELoss').
            - output_criterion: Loss for output layer (e.g., 'CrossEntropyLoss').
            - epochs_per_layer: Epochs for hidden layers.
            - epochs_output_layer: Epochs for the output layer.
            - log_interval: Frequency of logging batch metrics.
        device: Device to run training on.
        wandb_run: Optional W&B run object.
    """
    model.to(device)
    num_hidden_layers = len(model.hidden_dims)
    logger.info(f"Starting layer-wise MF training for {num_hidden_layers} hidden layers + 1 output layer.")

    # --- Get Training Configuration ---
    optimizer_name = config.get('optimizer', 'Adam')
    optimizer_params = config.get('optimizer_params', {'lr': 0.001})
    mf_criterion_name = config.get('mf_criterion', 'MSELoss')
    output_criterion_name = config.get('output_criterion', 'CrossEntropyLoss')
    epochs_per_layer = config.get('epochs_per_layer', 5)
    epochs_output_layer = config.get('epochs_output_layer', 10)
    log_interval = config.get('log_interval', 100)

    # Select loss functions
    if mf_criterion_name.lower() == 'mseloss':
        mf_criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported MF criterion: {mf_criterion_name}")

    if output_criterion_name.lower() == 'crossentropyloss':
        output_criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported output criterion: {output_criterion_name}")

    # Get the target projection matrix M_L (projection matrix for the output layer's *input*)
    # This corresponds to the projection matrix stored for the last hidden layer's output.
    target_projection_matrix = model.get_projection_matrix(num_hidden_layers).to(device) # M_{num_hidden}

    # --- Train Hidden Layers ---
    current_layer_input_fn = lambda img: img.view(img.shape[0], -1).to(device) # Input for first layer is flattened image

    for i in range(num_hidden_layers):
        logger.info(f"--- Training Hidden Layer {i+1}/{num_hidden_layers} ---")
        linear_layer = model.layers[i*2]
        act_layer = model.layers[i*2 + 1]
        params_to_optimize = list(linear_layer.parameters()) + list(act_layer.parameters())

        # Select optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(params_to_optimize, **optimizer_params)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(params_to_optimize, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Train the current hidden layer
        train_mf_layer(
            model=model,
            layer_index=i,
            optimizer=optimizer,
            criterion=mf_criterion,
            train_loader=train_loader,
            epochs=epochs_per_layer,
            device=device,
            get_layer_input_fn=current_layer_input_fn,
            target_projection_matrix=target_projection_matrix,
            wandb_run=wandb_run,
            log_interval=log_interval
        )

        # Freeze the trained layer (redundant due to model.eval() in train_mf_layer, but explicit)
        for param in params_to_optimize:
            param.requires_grad = False
        linear_layer.eval()
        act_layer.eval()

        # Update the input function for the *next* layer
        def create_next_input_fn(trained_layer_idx: int):
             def next_input_fn(img: torch.Tensor) -> torch.Tensor:
                 # Run image through layers 0 to trained_layer_idx (inclusive)
                 activation = img.view(img.shape[0], -1).to(device)
                 for k in range(trained_layer_idx + 1):
                     lin_k = model.layers[k*2]
                     act_k = model.layers[k*2 + 1]
                     activation = act_k(lin_k(activation))
                 return activation
             return next_input_fn

        current_layer_input_fn = create_next_input_fn(i)

    # --- Train Output Layer ---
    logger.info(f"--- Training Output Layer ---")
    output_layer = model.output_layer
    output_params = list(output_layer.parameters())

    if optimizer_name.lower() == 'adam':
        output_optimizer = optim.Adam(output_params, **optimizer_params)
    elif optimizer_name.lower() == 'sgd':
        output_optimizer = optim.SGD(output_params, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # The input function is the one created after the last hidden layer loop
    train_mf_output_layer(
        model=model,
        optimizer=output_optimizer,
        criterion=output_criterion,
        train_loader=train_loader,
        epochs=epochs_output_layer,
        device=device,
        get_output_layer_input_fn=current_layer_input_fn,
        wandb_run=wandb_run,
        log_interval=log_interval
    )

    logger.info("Finished all layer-wise MF training.")


def evaluate_mf_model(
    model: MF_MLP,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None # e.g., CrossEntropyLoss
) -> Dict[str, float]:
    """
    Evaluates the trained MF_MLP model using its standard forward pass.

    Args:
        model: The trained MF_MLP model.
        data_loader: DataLoader for the evaluation dataset (validation or test).
        device: Device to run evaluation on.
        criterion: Optional loss function for calculating evaluation loss.

    Returns:
        Dictionary containing evaluation metrics (e.g., 'loss', 'accuracy').
    """
    model.eval() # Ensure model is in eval mode
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    logger.info("Evaluating MF model (standard forward pass)")

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating")
        for images, labels in pbar:
            # MF MLP expects flattened input for standard forward pass
            images_flat = images.view(images.shape[0], -1).to(device)
            labels = labels.to(device)

            # Use the standard forward pass for predictions
            predictions = model.forward(images_flat)

            # Calculate loss if criterion is provided
            if criterion:
                loss = criterion(predictions, labels)
                total_loss += loss.item() * images.size(0) # Accumulate total loss

            # Calculate accuracy
            predicted_labels = torch.argmax(predictions, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix(acc=f"{(total_correct / total_samples * 100):.2f}%")


    avg_loss = total_loss / total_samples if criterion and total_samples > 0 else 0.0
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0

    logger.info(f"Evaluation Results: Accuracy: {accuracy:.2f}%" + (f", Loss: {avg_loss:.4f}" if criterion else ""))

    results = {
        'eval_accuracy': accuracy
    }
    if criterion:
        results['eval_loss'] = avg_loss

    return results


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Testing MF algorithm components...")

    # Setup
    batch_size = 4
    input_dim = 784 # F-MNIST like
    num_classes = 10
    hidden_dims = [100, 50] # Two hidden layers
    projection_dim = num_classes
    epochs_per_layer = 1
    epochs_output_layer = 1
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy Data
    num_batches = 5
    dummy_images = torch.randn(batch_size * num_batches, input_dim)
    dummy_labels = torch.randint(0, num_classes, (batch_size * num_batches,))
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    # Need to reshape images for loader if model expects spatial dims (but MF MLP uses flat)
    # If using CNN-like data loader:
    # dummy_images_spatial = dummy_images.view(batch_size * num_batches, 1, 28, 28)
    # dummy_dataset = TensorDataset(dummy_images_spatial, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=batch_size)


    # Dummy Model
    try:
        model = MF_MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            projection_dim=projection_dim
        ).to(device)
        print(model)

        # Dummy Config
        config = {
            'optimizer': 'Adam',
            'optimizer_params': {'lr': lr},
            'mf_criterion': 'MSELoss',
            'output_criterion': 'CrossEntropyLoss',
            'epochs_per_layer': epochs_per_layer,
            'epochs_output_layer': epochs_output_layer,
            'log_interval': 2
        }

        print("\n--- Testing train_mf_model ---")
        # Need to adapt loader if it yields spatial images
        # Assuming loader yields flat images for MLP test
        train_mf_model(model, dummy_loader, config, device)
        print("Finished model training.")

        print("\n--- Testing evaluate_mf_model ---")
        eval_metrics = evaluate_mf_model(model, dummy_loader, device, criterion=nn.CrossEntropyLoss())
        print("Evaluation metrics:", eval_metrics)
        assert 'eval_accuracy' in eval_metrics

    except Exception as e:
        print(f"\nError during MF testing: {e}", exc_info=True)
