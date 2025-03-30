import random
import numpy as np
import torch
import os

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across different libraries.

    Args:
        seed: The integer value to use as the seed.
    """
    if seed is None:
        print("Warning: No seed provided. Results may not be reproducible.")
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set seed for CUDA operations if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # Ensure deterministic behavior for CuDNN
        # Note: This can impact performance, enable if strict reproducibility is paramount
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False # Disable benchmark mode for determinism

    # Set PYTHONHASHSEED environment variable (optional, affects hash-based operations)
    # os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Set random seed to {seed}")


if __name__ == '__main__':
    # Example usage
    print("Setting seed to 123...")
    set_seed(123)
    print("Numpy random:", np.random.rand(1))
    print("Torch random:", torch.rand(1))
    print("Python random:", random.random())

    print("\nSetting seed to 456...")
    set_seed(456)
    print("Numpy random:", np.random.rand(1))
    print("Torch random:", torch.rand(1))
    print("Python random:", random.random())

    print("\nTesting without seed...")
    set_seed(None)
    print("Numpy random:", np.random.rand(1))
    print("Torch random:", torch.rand(1))
    print("Python random:", random.random())
