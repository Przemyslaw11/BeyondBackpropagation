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
        return 0.0

    with torch.no_grad():
        if outputs.device != targets.device:
            outputs = outputs.to(targets.device)

        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == targets).sum().item()
        accuracy = (correct / total) * 100.0
    return accuracy
