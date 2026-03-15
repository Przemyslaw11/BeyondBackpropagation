"""Helper functions for general utility tasks."""

import hashlib
import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def architecture_identifier(config: Dict[str, Any]) -> str:
    """Builds a stable string identifying the architecture a config trains.

    Includes the optimiser variants because two studies can otherwise share a
    model name, dataset and layer widths while being genuinely different studies.
    """
    model_config = config.get("model", {})
    model_params = model_config.get("params", {})
    algo_params = config.get("algorithm_params", {})

    parts = [str(model_config.get("name", "unknown"))]
    for key in ("hidden_dims", "block_channels"):
        value = model_params.get(key)
        if value is not None:
            parts.append(f"{key}={list(value)}")
    for key in ("optimizer_type", "predictor_optimizer_type"):
        value = algo_params.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    optimizer_type = config.get("optimizer", {}).get("type")
    if optimizer_type is not None:
        parts.append(f"optimizer={optimizer_type}")
    if algo_params.get("train_blocks", False):
        parts.append("dfa_blocks")
    return "|".join(parts)


def derive_study_seed(
    algorithm_name: str, dataset_name: str, architecture_id: str
) -> int:
    """Derives a reproducible sampler seed unique to one tuning study.

    Seeding every study from ``general.seed`` makes independent searches explore
    the same sequence of points, which biases cross-algorithm comparisons.
    """
    key = f"{algorithm_name.upper()}|{dataset_name.upper()}|{architecture_id}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def set_seed(seed: int) -> None:
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


def create_directory_if_not_exists(path: str) -> None:
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
    keep_best_only: bool = False,
) -> None:
    """Saves model checkpoint.

    Args:
        state: Dictionary containing model state and other info
            (e.g., epoch, optimizer state).
        is_best: Boolean flag indicating if this is the best model seen so far.
        filename: Base filename for the checkpoint.
        best_filename: Filename for the best model checkpoint.
        checkpoint_dir: Directory to save checkpoints.
        keep_best_only: Skip the per-epoch snapshot and write only the best model.
            Set False to retain resumable per-epoch checkpoints.
    """
    if not checkpoint_dir:
        logger.warning("Checkpoint directory not specified, cannot save checkpoint.")
        return

    if keep_best_only and not is_best:
        return

    create_directory_if_not_exists(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    best_filepath = os.path.join(checkpoint_dir, best_filename)

    try:
        if not keep_best_only:
            torch.save(state, filepath)
            logger.debug(f"Saved checkpoint to {filepath}")
        if is_best:
            epoch = state.get("epoch", "?")
            metric = state.get("best_metric_value", "?")
            metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else "?"
            logger.info(
                f"Saved best model state_dict to {best_filepath} "
                f"(Epoch {epoch}, Metric: {metric_str})"
            )
            torch.save(state["state_dict"], best_filepath)
    except Exception as e:
        logger.error(
            f"Failed to save checkpoint to {checkpoint_dir}: {e}", exc_info=True
        )
