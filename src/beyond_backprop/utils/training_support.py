"""Training-support helpers moved verbatim from the legacy ``src.utils`` modules.

These are the exact implementations the algorithm trainers rely on (checkpoint
naming, metric formatting, GPU memory queries); they live inside the canonical
package so trainers no longer import the legacy namespace.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)

try:  # pragma: no cover - environment dependent
    import pynvml
except ImportError:  # pragma: no cover - environment dependent
    pynvml = None  # type: ignore[assignment]

_nvml_initialized = False


def create_directory_if_not_exists(path: str) -> None:
    """Creates a directory if it doesn't already exist."""
    if path and not os.path.exists(path):
        try:
            os.makedirs(path)
            logger.info(f"Created directory: {path}")
        except OSError as e:
            logger.error(f"Failed to create directory {path}: {e}", exc_info=True)
            raise


def format_time(seconds: float) -> str:
    """Formats a duration in seconds into a human-readable string (HH:MM:SS)."""
    seconds = max(0, seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def save_checkpoint(
    state: dict[str, Any],
    is_best: bool,
    filename: str = "checkpoint.pth",
    best_filename: str = "model_best.pth",
    checkpoint_dir: str = "checkpoints",
) -> None:
    """Saves model checkpoint (verbatim legacy checkpoint naming semantics)."""
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
            epoch = state.get("epoch", "?")
            metric = state.get("best_metric_value", "?")
            metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else "?"
            logger.info(
                f"Saved best model state_dict to {best_filepath} "
                f"(Epoch {epoch}, Metric: {metric_str})"
            )
            torch.save(state["state_dict"], best_filepath)
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {filepath}: {e}", exc_info=True)


def _format_metric_for_logging(key: str, value: Any) -> str:
    """Formats a metric value into a string for console logging."""
    if not isinstance(value, float):
        return str(value)

    key_lower = key.lower()
    # Check for small values, emissions, or energy to use high precision
    if (
        "emission" in key_lower or "energy" in key_lower or abs(value) < 1e-3
    ) and value != 0.0:
        return f"{value:.6f}"
    # Check for gflops
    if "gflops" in key_lower:
        return f"{value:.4f}"
    # Default float formatting
    return f"{value:.4f}"


def log_metrics(
    metrics: dict[str, Any],
    wandb_run: Any | None = None,
    commit: bool = True,
) -> None:
    """Logs metrics to W&B (if enabled) and standard logger.

    Assumes the 'global_step' key is present in the metrics dictionary.
    """
    step_val = metrics.get("global_step", "N/A")

    is_final_summary = any(key.startswith("final/") for key in metrics)

    logger.info("")
    if is_final_summary:
        logger.info(f"--- Final Summary Metrics (Step: {step_val}) ---")
    else:
        logger.info(f"--- Metrics Log (Step: {step_val}) ---")

    for k, v in metrics.items():
        if k in ("global_step", "final/codecarbon_emissions_kgCO2e"):
            continue
        log_value = _format_metric_for_logging(k, v)
        logger.info(f"  {k}: {log_value}")

    if is_final_summary:
        logger.info("--- End Final Summary ---")
    else:
        logger.info("--- End Metrics Log ---")
    logger.info("")

    try:
        import wandb

        active_run = wandb_run or wandb.run
    except ImportError:
        active_run = wandb_run
    if active_run:
        try:
            active_run.log(metrics, commit=commit)
        except Exception as e:
            logger.error(
                f"Failed to log metrics to Weights & Biases: {e}", exc_info=True
            )


def get_gpu_memory_usage(
    handle: Any,
) -> tuple[float, float, float] | None:
    """Gets the memory usage of the GPU in MiB (Used, Total, Free).

    Returns:
        Tuple[Used MiB, Total MiB, Free MiB] or None if failed.
    """
    if pynvml is None or not handle:
        logger.debug("Invalid GPU handle provided for memory usage query.")
        return None
    if not _nvml_initialized:
        logger.warning("NVML not initialized, cannot get memory usage.")
        return None
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        bytes_to_mib = 1 / (1024**2)
        total_mib = mem_info.total * bytes_to_mib
        used_mib = mem_info.used * bytes_to_mib
        free_mib = mem_info.free * bytes_to_mib
        return used_mib, total_mib, free_mib
    except pynvml.NVMLError as error:  # type: ignore[union-attr]
        logger.error(f"Failed to get memory usage: {error}", exc_info=True)
        return None


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculates the classification accuracy in percent (0.0 for empty inputs)."""
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


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configures the root logger.

    Args:
        log_level: Logging level string (e.g., 'DEBUG', 'INFO', 'WARNING').
        log_file: Optional path to a file for logging.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root_logger = logging.getLogger()
    if (
        not root_logger.hasHandlers()
        or os.environ.get("LOGGING_SETUP_COMPLETE") is None
    ):
        root_logger.setLevel(level)

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging to file: {log_file}")

        os.environ["LOGGING_SETUP_COMPLETE"] = "1"
        root_logger.info(f"Root logger setup complete. Level: {log_level.upper()}")
    else:
        root_logger.info("Root logger already configured.")


logger = logging.getLogger(__name__)


def setup_wandb(
    config: dict[str, Any],
    project_name: str = "BeyondBackpropagation",
    entity: str | None = None,
    run_name: str | None = None,
    notes: list[str] | None = None,
    tags: list[str] | None = None,
    job_type: str = "training",
) -> Any | None:
    """Initializes a Weights & Biases run."""
    try:
        import wandb
    except ImportError:
        logger.error("wandb library not found. Install with `pip install wandb`")
        return None

    wandb_config = config.get("logging", {}).get("wandb", {})
    if not wandb_config.get("use_wandb", True):
        logger.info("Weights & Biases logging is disabled in the configuration.")
        return None

    try:
        if not os.getenv("WANDB_API_KEY"):
            logger.warning(
                "WANDB_API_KEY environment variable not set. W&B logging might fail or prompt."
            )

        resolved_entity = (
            entity or os.getenv("WANDB_ENTITY") or wandb_config.get("entity")
        )
        if not resolved_entity:
            logger.warning(
                "W&B entity not specified via args, config, or WANDB_ENTITY "
                "env var. Using W&B default."
            )

        resolved_project = wandb_config.get("project", project_name)
        resolved_run_name = (
            run_name or wandb_config.get("run_name") or config.get("experiment_name")
        )
        if not resolved_run_name:
            resolved_run_name = f"run_{int(time.time())}"

        run = wandb.init(
            project=resolved_project,
            entity=resolved_entity,
            config=config,
            name=resolved_run_name,
            notes=notes,  # type: ignore[arg-type]
            tags=tags,
            job_type=job_type,
            reinit=True,
        )
        logger.info(f"Weights & Biases run initialized: {run.url if run else 'Failed'}")
        return run
    except Exception as e:
        logger.error(f"Failed to initialize Weights & Biases: {e}", exc_info=True)
        return None


__all__ = [
    "calculate_accuracy",
    "create_directory_if_not_exists",
    "format_time",
    "get_gpu_memory_usage",
    "log_metrics",
    "save_checkpoint",
    "setup_logging",
    "setup_wandb",
]
