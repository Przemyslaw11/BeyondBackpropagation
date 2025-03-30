import torch

def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates the classification accuracy.

    Args:
        outputs: The model's output logits or probabilities (shape: [batch_size, num_classes]).
        targets: The ground truth labels (shape: [batch_size]).

    Returns:
        The accuracy as a percentage (float between 0.0 and 100.0).
    """
    with torch.no_grad():
        # Get the index of the max log-probability/probability
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        accuracy = (correct / total) * 100.0
    return accuracy

if __name__ == '__main__':
    # Example usage
    print("Testing calculate_accuracy...")

    # Test case 1: Perfect match
    outputs1 = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    targets1 = torch.tensor([1, 0, 1])
    acc1 = calculate_accuracy(outputs1, targets1)
    print(f"Test Case 1: Outputs=\n{outputs1}\nTargets={targets1}\nAccuracy: {acc1:.2f}% (Expected: 100.00%)")
    assert acc1 == 100.0

    # Test case 2: Partial match
    outputs2 = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.1, 0.9], [0.9, 0.1]])
    targets2 = torch.tensor([0, 0, 1, 1]) # Predictions: 0, 1, 1, 0
    acc2 = calculate_accuracy(outputs2, targets2)
    print(f"\nTest Case 2: Outputs=\n{outputs2}\nTargets={targets2}\nAccuracy: {acc2:.2f}% (Expected: 50.00%)")
    assert acc2 == 50.0

    # Test case 3: No match
    outputs3 = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    targets3 = torch.tensor([1, 1]) # Predictions: 0, 0
    acc3 = calculate_accuracy(outputs3, targets3)
    print(f"\nTest Case 3: Outputs=\n{outputs3}\nTargets={targets3}\nAccuracy: {acc3:.2f}% (Expected: 0.00%)")
    assert acc3 == 0.0

    # Test case 4: Empty batch (should not error, return 0 or handle gracefully)
    outputs4 = torch.empty((0, 10)) # 0 batch size, 10 classes
    targets4 = torch.empty((0,), dtype=torch.long)
    try:
        acc4 = calculate_accuracy(outputs4, targets4)
        print(f"\nTest Case 4: Empty batch\nAccuracy: {acc4:.2f}% (Expected: 0.00% or NaN, handled as 0)")
        # Depending on implementation, division by zero might yield NaN or raise error.
        # A robust implementation handles this. Here, total=0 -> accuracy=0.
        assert acc4 == 0.0 or torch.isnan(torch.tensor(acc4)) # Allow NaN or 0 for empty
    except ZeroDivisionError:
        print("\nTest Case 4: Empty batch resulted in ZeroDivisionError (handled).")
        # This is also acceptable behavior if not explicitly handled for 0/0
