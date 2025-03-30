import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import tqdm # For progress bars
from typing import Dict, Any, Optional, Tuple

# Assuming architectures and utils are accessible via src package
from src.architectures.ff_mlp import FF_MLP, FF_Layer
from src.utils.metrics import calculate_accuracy # For potential evaluation
from src.utils.logging_utils import log_metrics # For logging during training

logger = logging.getLogger(__name__)

def generate_positive_negative_data(images: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates positive and negative samples for FF training.
    Positive: Real image with correct label embedded.
    Negative: Real image with incorrect label embedded.

    Args:
        images: Batch of input images (shape: [batch_size, input_dim]).
        labels: Corresponding correct labels (shape: [batch_size]).
        num_classes: Total number of classes in the dataset.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (positive_data, negative_data)
        Both tensors have shape [batch_size, input_dim + num_classes].
    """
    batch_size = images.shape[0]
    input_dim = images.shape[1]
    device = images.device

    # --- Create Positive Data ---
    # Pad images with space for labels
    positive_data = torch.zeros((batch_size, input_dim + num_classes), device=device)
    positive_data[:, num_classes:] = images
    # Embed correct labels
    pos_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    positive_data[:, :num_classes] = pos_one_hot

    # --- Create Negative Data ---
    # Pad images
    negative_data = torch.zeros((batch_size, input_dim + num_classes), device=device)
    negative_data[:, num_classes:] = images
    # Generate incorrect labels (simple strategy: shift labels by 1, wrap around)
    neg_labels = (labels + torch.randint(1, num_classes, (batch_size,), device=device)) % num_classes
    # Ensure neg_label is different from pos_label (highly likely with random int, but double check)
    while torch.any(neg_labels == labels):
         neg_labels = (labels + torch.randint(1, num_classes, (batch_size,), device=device)) % num_classes

    neg_one_hot = F.one_hot(neg_labels, num_classes=num_classes).float()
    negative_data[:, :num_classes] = neg_one_hot

    return positive_data, negative_data


def ff_loss_fn(pos_goodness: torch.Tensor, neg_goodness: torch.Tensor, threshold: float) -> torch.Tensor:
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
    loss_pos = torch.log(1 + torch.exp(-torch.clamp(pos_goodness - threshold, min=-10, max=10)))

    # Loss for negative samples: encourage goodness < threshold
    # log(1 + exp(neg_goodness - threshold)) -> pushes neg_goodness lower
    loss_neg = torch.log(1 + torch.exp(torch.clamp(neg_goodness - threshold, min=-10, max=10)))

    # Combine and average
    loss = torch.mean(loss_pos + loss_neg)
    return loss


def train_ff_layer(
    layer: FF_Layer,
    train_loader: DataLoader, # Loader providing (images, labels)
    optimizer: optim.Optimizer,
    num_classes: int,
    threshold: float,
    epochs: int,
    device: torch.device,
    prev_layer_forward_fn: callable, # Function to get input for current layer
    wandb_run: Optional[Any] = None, # For logging
    log_interval: int = 100
) -> None:
    """
    Trains a single FF_Layer using the Forward-Forward algorithm.

    Args:
        layer: The FF_Layer module to train.
        train_loader: DataLoader providing batches of original (images, labels).
        optimizer: PyTorch optimizer for the layer's parameters.
        num_classes: Number of classes for label embedding.
        threshold: Goodness threshold for the loss function.
        epochs: Number of training epochs for this layer.
        device: Device to perform training on ('cuda' or 'cpu').
        prev_layer_forward_fn: A function that takes a batch of (images, labels)
                               and returns the input tensor for the *current* layer.
                               For the first layer, this function generates pos/neg data.
                               For subsequent layers, it runs the previous trained layers.
        wandb_run: Optional Weights & Biases run object for logging.
        log_interval: How often to log batch metrics.
    """
    layer.train()
    layer.to(device)
    logger.info(f"Starting FF training for layer: {layer}")

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # 1. Get input for the current layer using the provided function
            # This function handles generating pos/neg data or running previous layers
            # It should return inputs ready for the current layer's forward_with_goodness
            pos_input, neg_input = prev_layer_forward_fn(images, labels)

            # Ensure inputs are on the correct device
            pos_input, neg_input = pos_input.to(device), neg_input.to(device)

            # 2. Forward pass to get goodness
            # Detach inputs if they came from previous frozen layers to avoid backprop
            _, pos_goodness = layer.forward_with_goodness(pos_input.detach())
            _, neg_goodness = layer.forward_with_goodness(neg_input.detach())

            # 3. Calculate loss
            loss = ff_loss_fn(pos_goodness, neg_goodness, threshold)

            # 4. Backpropagate and optimize (only for the current layer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # Logging
            if batch_idx % log_interval == 0:
                avg_loss = loss.item()
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
                metrics = {
                    f'layer_{layer.__class__.__name__}_train_loss_batch': avg_loss,
                    f'layer_{layer.__class__.__name__}_pos_goodness_mean': pos_goodness.mean().item(),
                    f'layer_{layer.__class__.__name__}_neg_goodness_mean': neg_goodness.mean().item(),
                }
                log_metrics(metrics, step=global_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")
        log_metrics({f'layer_{layer.__class__.__name__}_train_loss_epoch': avg_epoch_loss}, step=epoch+1, wandb_run=wandb_run)

    logger.info(f"Finished FF training for layer: {layer}")


# --- Orchestration Function (To be implemented) ---
# This function will iterate through the layers of an FF_MLP,
# set up the `prev_layer_forward_fn` correctly for each layer,
# create an optimizer for each layer, and call `train_ff_layer`.

# def train_ff_model(
#     model: FF_MLP,
#     train_loader: DataLoader,
#     config: Dict[str, Any], # Containing optimizer params, threshold, epochs per layer etc.
#     device: torch.device,
#     wandb_run: Optional[Any] = None
# ):
#     model.to(device)
#     num_layers = len(model.layers)
#     logger.info(f"Starting layer-wise FF training for {num_layers} layers.")

#     trained_layers_forward = None # Keep track of the forward pass of already trained layers

#     for i, layer in enumerate(model.layers):
#         logger.info(f"--- Training Layer {i+1}/{num_layers} ---")

#         # Define the forward function to get input for the current layer
#         if i == 0:
#             # First layer: input is generated positive/negative data
#             def get_layer_input(images, labels):
#                 # Need original image dimensions here
#                 img_flat = images.view(images.shape[0], -1)
#                 return generate_positive_negative_data(img_flat, labels, model.num_classes)
#         else:
#             # Subsequent layers: input is the output of the previous *trained* layers
#             # Need to capture the state of previously trained layers
#             # This requires careful handling of closures or partial functions
#             # We need the forward pass using layers 0 to i-1
#             # Let's assume we have a function `get_intermediate_output(model, layer_idx, data)`
#             # This part is tricky and needs robust implementation.
#             # Placeholder:
#             def get_layer_input(images, labels):
#                  # This needs the model state *up to layer i-1*
#                  # Run previously trained layers (frozen)
#                  with torch.no_grad():
#                      # Generate pos/neg data based on original images/labels
#                      img_flat = images.view(images.shape[0], -1)
#                      pos_data_base, neg_data_base = generate_positive_negative_data(img_flat, labels, model.num_classes)

#                      # Pass base data through trained layers 0 to i-1
#                      pos_input_current = model.forward(pos_data_base, embed_labels=False)[i] # Get output of layer i-1
#                      neg_input_current = model.forward(neg_data_base, embed_labels=False)[i] # Get output of layer i-1
#                      # NOTE: model.forward needs adjustment or a dedicated function
#                      # to run only up to a certain layer index.

#                  return pos_input_current, neg_input_current


#         # Create optimizer for the current layer only
#         # TODO: Get optimizer params from config
#         optimizer = optim.Adam(layer.parameters(), lr=config.get('learning_rate', 0.001))
#         layer_epochs = config.get('epochs_per_layer', 10)
#         threshold = config.get('threshold', 1.0) # Example threshold

#         train_ff_layer(
#             layer=layer,
#             train_loader=train_loader,
#             optimizer=optimizer,
#             num_classes=model.num_classes,
#             threshold=threshold,
#             epochs=layer_epochs,
#             device=device,
#             prev_layer_forward_fn=get_layer_input, # Pass the correct input function
#             wandb_run=wandb_run,
#             log_interval=config.get('log_interval', 100)
#         )

#         # Freeze the trained layer
#         for param in layer.parameters():
#             param.requires_grad = False
#         layer.eval() # Set to eval mode after training

#     logger.info("Finished all layer-wise FF training.")


# --- Evaluation Function (To be implemented) ---
# Evaluation in FF often involves presenting an image with each possible label
# and seeing which label yields the highest average goodness across layers,
# or training a final linear classifier on the last layer's activations.

# def evaluate_ff_model(model: FF_MLP, data_loader: DataLoader, num_classes: int, device: torch.device):
#     model.eval()
#     model.to(device)
#     # ... implementation ...


if __name__ == '__main__':
    # Example usage (requires dummy data and model)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Testing FF algorithm components...")

    # Setup
    batch_size = 4
    input_dim = 784
    num_classes = 10
    hidden_dims = [100] # Single hidden layer for simplicity
    threshold = 1.5
    epochs_per_layer = 2
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy Data
    dummy_images = torch.randn(batch_size * 5, input_dim) # 5 batches
    dummy_labels = torch.randint(0, num_classes, (batch_size * 5,))
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=batch_size)

    # Dummy Model (single layer)
    # Input to first layer includes label embedding
    first_layer_input_dim = input_dim + num_classes
    layer1 = FF_Layer(first_layer_input_dim, hidden_dims[0]).to(device)
    optimizer1 = optim.Adam(layer1.parameters(), lr=lr)

    # Define the input function for the first layer
    def get_first_layer_input(images, labels):
        img_flat = images.view(images.shape[0], -1) # Ensure flat
        return generate_positive_negative_data(img_flat, labels, num_classes)

    print("\n--- Testing generate_positive_negative_data ---")
    test_imgs, test_lbls = next(iter(dummy_loader))
    pos_data, neg_data = get_first_layer_input(test_imgs, test_lbls)
    print("Image shape:", test_imgs.shape)
    print("Label shape:", test_lbls.shape)
    print("Positive data shape:", pos_data.shape)
    print("Negative data shape:", neg_data.shape)
    assert pos_data.shape == (batch_size, input_dim + num_classes)
    assert neg_data.shape == (batch_size, input_dim + num_classes)
    # Check labels are different
    pos_lbl_idx = torch.argmax(pos_data[:, :num_classes], dim=1)
    neg_lbl_idx = torch.argmax(neg_data[:, :num_classes], dim=1)
    assert torch.all(pos_lbl_idx == test_lbls)
    assert torch.all(neg_lbl_idx != test_lbls)
    print("Positive/Negative data generation looks OK.")


    print("\n--- Testing ff_loss_fn ---")
    pos_g = torch.tensor([1.8, 2.5, 1.0])
    neg_g = torch.tensor([0.5, -0.2, 1.2])
    loss = ff_loss_fn(pos_g, neg_g, threshold=1.5)
    print(f"Pos Goodness: {pos_g}")
    print(f"Neg Goodness: {neg_g}")
    print(f"Threshold: 1.5")
    print(f"Loss: {loss.item():.4f}")
    # Expect loss for pos_g[0] (1.8 > 1.5), pos_g[1] (2.5 > 1.5) to be low
    # Expect loss for pos_g[2] (1.0 < 1.5) to be high
    # Expect loss for neg_g[0] (0.5 < 1.5), neg_g[1] (-0.2 < 1.5) to be low
    # Expect loss for neg_g[2] (1.2 < 1.5) to be low-ish
    # Overall loss should be positive.


    print("\n--- Testing train_ff_layer (Layer 1) ---")
    train_ff_layer(
        layer=layer1,
        train_loader=dummy_loader,
        optimizer=optimizer1,
        num_classes=num_classes,
        threshold=threshold,
        epochs=epochs_per_layer,
        device=device,
        prev_layer_forward_fn=get_first_layer_input,
        log_interval=2 # Log more often for small test
    )
    print("Finished training layer 1.")

    # --- Example for Layer 2 (Conceptual) ---
    # if len(hidden_dims) > 1:
    #     print("\n--- Testing train_ff_layer (Layer 2 - Conceptual) ---")
    #     layer1.eval() # Freeze layer 1
    #     for param in layer1.parameters(): param.requires_grad = False
    #
    #     layer2 = FF_Layer(hidden_dims[0], hidden_dims[1]).to(device)
    #     optimizer2 = optim.Adam(layer2.parameters(), lr=lr)
    #
    #     # Define input function for layer 2
    #     def get_second_layer_input(images, labels):
    #         with torch.no_grad():
    #             pos_data_l1_in, neg_data_l1_in = get_first_layer_input(images, labels)
    #             pos_data_l1_out, _ = layer1.forward_with_goodness(pos_data_l1_in.to(device))
    #             neg_data_l1_out, _ = layer1.forward_with_goodness(neg_data_l1_in.to(device))
    #         return pos_data_l1_out, neg_data_l1_out
    #
    #     train_ff_layer(
    #         layer=layer2,
    #         train_loader=dummy_loader,
    #         optimizer=optimizer2,
    #         num_classes=num_classes, # Not directly used by layer 2 input fn, but good practice
    #         threshold=threshold,
    #         epochs=epochs_per_layer,
    #         device=device,
    #         prev_layer_forward_fn=get_second_layer_input,
    #         log_interval=2
    #     )
    #     print("Finished training layer 2.")
