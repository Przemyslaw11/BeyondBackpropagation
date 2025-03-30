# File: src/utils/metrics.py
import torch
import logging

logger = logging.getLogger(__name__)


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates the classification accuracy.

    Args:
        outputs: The model's output logits or probabilities (shape: [batch_size, num_classes]).
        targets: The ground truth labels (shape: [batch_size]).

    Returns:
        The accuracy as a percentage (float between 0.0 and 100.0). Returns 0.0 for empty inputs.
    """
    total = targets.size(0)
    if total == 0:
        return 0.0  # Handle empty batch case

    with torch.no_grad():
        # Get the index of the max log-probability/probability
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == targets).sum().item()
        accuracy = (correct / total) * 100.0
    return accuracy


# Removed the __main__ block for cleaner utils file
