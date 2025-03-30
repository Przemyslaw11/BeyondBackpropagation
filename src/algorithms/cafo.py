import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from tqdm import tqdm # For progress bars
from typing import Dict, Any, Optional, Callable, List

# Assuming architectures and utils are accessible via src package
from src.architectures.cafo_cnn import CaFo_CNN, CaFoBlock, CaFoPredictor
from src.utils.metrics import calculate_accuracy
from src.utils.logging_utils import log_metrics

logger = logging.getLogger(__name__)

def train_cafo_block_and_predictor(
    block: CaFoBlock,
    predictor: CaFoPredictor,
    optimizer: optim.Optimizer,
    criterion: nn.Module, # Loss function (e.g., CrossEntropyLoss)
    train_loader: DataLoader, # Loader providing original (images, labels)
    epochs: int,
    device: torch.device,
    get_block_input_fn: Callable[[torch.Tensor], torch.Tensor], # Function to get input for the current block
    wandb_run: Optional[Any] = None, # For logging
    log_interval: int = 100,
    block_index: int = 0 # For logging purposes
) -> None:
    """
    Trains a single CaFoBlock and its associated CaFoPredictor layer-wise.

    Args:
        block: The CaFoBlock module to train.
        predictor: The CaFoPredictor module to train.
        optimizer: PyTorch optimizer for the parameters of *both* block and predictor.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        train_loader: DataLoader providing batches of original (images, labels).
        epochs: Number of training epochs for this block/predictor pair.
        device: Device to perform training on ('cuda' or 'cpu').
        get_block_input_fn: A function that takes a batch of original images
                            and returns the input tensor for the *current* block
                            (output of previous frozen blocks, or original image).
        wandb_run: Optional Weights & Biases run object for logging.
        log_interval: How often to log batch metrics.
        block_index: Index of the current block (for clearer logging).
    """
    block.train()
    predictor.train()
    block.to(device)
    predictor.to(device)
    logger.info(f"Starting CaFo training for Block {block_index + 1}")

    global_step = 0 # Track steps across epochs for this block
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        pbar = tqdm(train_loader, desc=f"Block {block_index+1} Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # 1. Get input for the current block using the provided function
            # This function runs the previous frozen blocks
            with torch.no_grad(): # Ensure previous block computation doesn't track gradients
                 block_input = get_block_input_fn(images)

            # 2. Forward pass through current block and predictor
            # Input to the block should be detached if it came from frozen layers,
            # but get_block_input_fn runs within no_grad, so it's already detached.
            block_output = block(block_input)
            predictions = predictor(block_output) # Gradients flow back to block_output and block

            # 3. Calculate loss
            loss = criterion(predictions, labels)

            # 4. Backpropagate and optimize (only for the current block and predictor)
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
                    f'block_{block_index+1}_train_loss_batch': avg_loss,
                    f'block_{block_index+1}_train_acc_batch': batch_accuracy,
                }
                log_metrics(metrics, step=global_step, wandb_run=wandb_run)

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_accuracy = epoch_accuracy / len(train_loader)
        logger.info(f"Block {block_index+1} Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}, Avg Acc: {avg_epoch_accuracy:.2f}%")
        log_metrics({
            f'block_{block_index+1}_train_loss_epoch': avg_epoch_loss,
            f'block_{block_index+1}_train_acc_epoch': avg_epoch_accuracy
        }, step=epoch+1, wandb_run=wandb_run)

    logger.info(f"Finished CaFo training for Block {block_index + 1}")


def train_cafo_model(
    model: CaFo_CNN,
    train_loader: DataLoader,
    config: Dict[str, Any], # Containing optimizer params, loss, epochs per block etc.
    device: torch.device,
    wandb_run: Optional[Any] = None
):
    """
    Orchestrates the layer-wise training of a CaFo_CNN model.

    Args:
        model: The CaFo_CNN model instance.
        train_loader: DataLoader providing original (images, labels).
        config: Dictionary with training configurations:
            - optimizer: Name of the optimizer (e.g., 'Adam', 'SGD').
            - optimizer_params: Dictionary of parameters for the optimizer (e.g., {'lr': 0.001}).
            - criterion: Name of the loss function (e.g., 'CrossEntropyLoss').
            - epochs_per_block: Number of epochs to train each block/predictor pair.
            - log_interval: Frequency of logging batch metrics.
        device: Device to run training on.
        wandb_run: Optional W&B run object.
    """
    model.to(device)
    num_blocks = len(model.blocks)
    logger.info(f"Starting layer-wise CaFo training for {num_blocks} blocks.")

    # --- Get Training Configuration ---
    optimizer_name = config.get('optimizer', 'Adam')
    optimizer_params = config.get('optimizer_params', {'lr': 0.001})
    criterion_name = config.get('criterion', 'CrossEntropyLoss')
    epochs_per_block = config.get('epochs_per_block', 10)
    log_interval = config.get('log_interval', 100)

    # Select loss function
    if criterion_name.lower() == 'crossentropyloss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

    # --- Train Block by Block ---
    current_block_input_fn = lambda img: img # Input for the first block is the image itself

    for i in range(num_blocks):
        logger.info(f"--- Training Block {i+1}/{num_blocks} ---")
        block = model.blocks[i]
        predictor = model.predictors[i]

        # Parameters to optimize for this stage: current block and predictor
        params_to_optimize = list(block.parameters()) + list(predictor.parameters())

        # Select optimizer
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(params_to_optimize, **optimizer_params)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(params_to_optimize, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Train the current block and predictor
        train_cafo_block_and_predictor(
            block=block,
            predictor=predictor,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            epochs=epochs_per_block,
            device=device,
            get_block_input_fn=current_block_input_fn,
            wandb_run=wandb_run,
            log_interval=log_interval,
            block_index=i
        )

        # Freeze the trained block and predictor
        for param in block.parameters():
            param.requires_grad = False
        for param in predictor.parameters():
            param.requires_grad = False
        block.eval()
        predictor.eval()

        # Update the input function for the *next* block
        # Need to capture the current block 'i' in the closure correctly
        def create_next_input_fn(trained_block_idx: int):
             def next_input_fn(img: torch.Tensor) -> torch.Tensor:
                 # Run image through blocks 0 to trained_block_idx (inclusive)
                 activation = img
                 for k in range(trained_block_idx + 1):
                     activation = model.blocks[k](activation)
                 return activation
             return next_input_fn

        current_block_input_fn = create_next_input_fn(i)


    logger.info("Finished all layer-wise CaFo training.")


def evaluate_cafo_model(
    model: CaFo_CNN,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None, # Optional: if loss calculation is needed
    use_predictor_index: int = -1 # Which predictor to use for evaluation (default: last)
) -> Dict[str, float]:
    """
    Evaluates the CaFo model using a specific predictor (usually the last one).

    Args:
        model: The trained CaFo_CNN model (with frozen blocks/predictors).
        data_loader: DataLoader for the evaluation dataset (validation or test).
        device: Device to run evaluation on.
        criterion: Optional loss function for calculating evaluation loss.
        use_predictor_index: Index of the predictor to use for evaluation (-1 for the last).

    Returns:
        Dictionary containing evaluation metrics (e.g., 'loss', 'accuracy').
    """
    model.eval() # Ensure model is in eval mode
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    predictor_idx = use_predictor_index if use_predictor_index >= 0 else len(model.predictors) - 1
    if predictor_idx >= len(model.predictors):
         raise ValueError(f"Invalid predictor index {predictor_idx} for model with {len(model.predictors)} predictors.")

    logger.info(f"Evaluating CaFo model using Predictor {predictor_idx + 1}")
    predictor_to_use = model.predictors[predictor_idx]

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Evaluating Predictor {predictor_idx+1}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Get the output of the block preceding the chosen predictor
            block_activation = images
            for i in range(predictor_idx + 1): # Run blocks 0 up to predictor_idx
                block_activation = model.blocks[i](block_activation)

            # Get predictions from the chosen predictor
            predictions = predictor_to_use(block_activation)

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

    logger.info(f"Evaluation Results (Predictor {predictor_idx+1}): Accuracy: {accuracy:.2f}%" + (f", Loss: {avg_loss:.4f}" if criterion else ""))

    results = {
        f'eval_predictor_{predictor_idx+1}_accuracy': accuracy
    }
    if criterion:
        results[f'eval_predictor_{predictor_idx+1}_loss'] = avg_loss

    return results


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Testing CaFo algorithm components...")

    # Setup
    batch_size = 4
    input_channels = 1 # Grayscale like F-MNIST
    image_size = 28
    num_classes = 10
    block_channels = [16, 32] # Two blocks for simplicity
    epochs_per_block = 1 # Minimal epochs for testing
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy Data (F-MNIST like)
    num_batches = 5
    dummy_images = torch.randn(batch_size * num_batches, input_channels, image_size, image_size)
    dummy_labels = torch.randint(0, num_classes, (batch_size * num_batches,))
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=batch_size)

    # Dummy Model
    try:
        model = CaFo_CNN(
            input_channels=input_channels,
            block_channels=block_channels,
            image_size=image_size,
            num_classes=num_classes
        ).to(device)
        print(model)

        # Dummy Config
        config = {
            'optimizer': 'Adam',
            'optimizer_params': {'lr': lr},
            'criterion': 'CrossEntropyLoss',
            'epochs_per_block': epochs_per_block,
            'log_interval': 2
        }

        print("\n--- Testing train_cafo_model ---")
        train_cafo_model(model, dummy_loader, config, device)
        print("Finished model training.")

        print("\n--- Testing evaluate_cafo_model (using last predictor) ---")
        eval_metrics = evaluate_cafo_model(model, dummy_loader, device, criterion=nn.CrossEntropyLoss())
        print("Evaluation metrics:", eval_metrics)
        assert 'eval_predictor_2_accuracy' in eval_metrics

        print("\n--- Testing evaluate_cafo_model (using first predictor) ---")
        eval_metrics_p1 = evaluate_cafo_model(model, dummy_loader, device, use_predictor_index=0)
        print("Evaluation metrics (Predictor 1):", eval_metrics_p1)
        assert 'eval_predictor_1_accuracy' in eval_metrics_p1


    except Exception as e:
        print(f"\nError during CaFo testing: {e}", exc_info=True)
