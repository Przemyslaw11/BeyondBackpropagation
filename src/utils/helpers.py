import random
import numpy as np
import torch
import os
import time

def set_seed(seed: int):
    """
    Sets the seed for reproducibility across different libraries.

    Args:
        seed: The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        # Ensure deterministic behavior for cuDNN (can impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {seed}")

def create_directory_if_not_exists(path: str):
    """
    Creates a directory if it doesn't already exist.

    Args:
        path: The directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def format_time(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable string (HH:MM:SS).

    Args:
        seconds: The duration in seconds.

    Returns:
        A string representing the formatted time.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

if __name__ == '__main__':
    # Example usage
    print("Testing set_seed...")
    set_seed(42)
    print(f"Python random: {random.random()}")
    print(f"Numpy random: {np.random.rand()}")
    print(f"Torch random: {torch.rand(1)}")
    if torch.cuda.is_available():
        print(f"Torch CUDA random: {torch.cuda.FloatTensor(1).normal_()}")

    print("\nTesting create_directory_if_not_exists...")
    test_dir = "temp_test_dir_helpers"
    create_directory_if_not_exists(test_dir)
    create_directory_if_not_exists(test_dir) # Should not print again
    if os.path.exists(test_dir):
        os.rmdir(test_dir)
        print(f"Cleaned up directory: {test_dir}")

    print("\nTesting format_time...")
    print(f"1 second: {format_time(1)}")
    print(f"90 seconds: {format_time(90)}")
    print(f"3661 seconds: {format_time(3661)}")
    print(f"86400 seconds: {format_time(86400)}")
