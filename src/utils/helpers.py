# File: src/utils/helpers.py
import random
import numpy as np
import torch
import os
import time
import logging

logger = logging.getLogger(__name__)


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
        # Set these based on config or environment needs, as they affect performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        logger.info(f"Set random seed to {seed} (including CUDA)")
    else:
        logger.info(f"Set random seed to {seed} (CUDA not available)")


def create_directory_if_not_exists(path: str):
    """
    Creates a directory if it doesn't already exist.

    Args:
        path: The directory path to create.
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            logger.info(f"Created directory: {path}")
        except OSError as e:
            logger.error(f"Failed to create directory {path}: {e}", exc_info=True)
            raise  # Re-raise error if creation fails


def format_time(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable string (HH:MM:SS).

    Args:
        seconds: The duration in seconds.

    Returns:
        A string representing the formatted time.
    """
    seconds = max(0, seconds)  # Ensure non-negative
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


# Removed the __main__ block for cleaner utils file
