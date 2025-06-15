import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Sets the seed for reproducibility across different libraries.

    Args:
        seed: The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"Set random seed to {seed} (including CUDA)")
    else:
        logger.info(f"Set random seed to {seed} (CUDA not available)")


def create_directory_if_not_exists(path: str):
    """Creates a directory if it doesn't already exist.

    Args:
        path: The directory path to create.
    """
    if path and not os.path.exists(path):
        try:
            os.makedirs(path)
            logger.info(f"Created directory: {path}")
        except OSError as e:
            logger.error(f"Failed to create directory {path}: {e}", exc_info=True)
            raise

def format_time(seconds: float) -> str:
    """Formats a duration in seconds into a human-readable string (HH:MM:SS).

    Args:
        seconds: The duration in seconds.

    Returns:
        A string representing the formatted time.
    """
    seconds = max(0, seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    filename: str = "checkpoint.pth",
    best_filename: str = "model_best.pth",
    checkpoint_dir: str = "checkpoints",
):
    """Saves model checkpoint.

    Args:
        state: Dictionary containing model state and other info (e.g., epoch, optimizer state).
        is_best: Boolean flag indicating if this is the best model seen so far.
        filename: Base filename for the checkpoint.
        best_filename: Filename for the best model checkpoint.
        checkpoint_dir: Directory to save checkpoints.
    """
    if not checkpoint_dir:
        logger.warning("Checkpoint directory not specified, cannot save checkpoint.")
        return

    create_directory_if_not_exists(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    best_filepath = os.path.join(checkpoint_dir, best_filename)

    try:
        torch.save(state, filepath)
        logger.debug(f"Saved checkpoint to {filepath}")
        if is_best:
            torch.save(
                state["state_dict"], best_filepath
            )
            logger.info(
                f"Saved best model state_dict to {best_filepath} (Epoch {state.get('epoch', '?')}, Metric: {state.get('best_metric_value', '?'):.4f})"
            )
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {filepath}: {e}", exc_info=True)
