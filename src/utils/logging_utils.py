"""Logging configuration and utilities for the project."""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import wandb


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
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
    config: Dict[str, Any],
    project_name: str = "BeyondBackpropagation",
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[List[str]] = None,
    job_type: str = "training",
) -> Optional["wandb.sdk.wandb_run.Run"]:
    """Initializes a Weights & Biases run."""
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
            notes=notes,
            tags=tags,
            job_type=job_type,
            reinit=True,
        )
        logger.info(f"Weights & Biases run initialized: {run.url if run else 'Failed'}")
        return run
    except ImportError:
        logger.error("wandb library not found. Install with `pip install wandb`")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Weights & Biases: {e}", exc_info=True)
        return None


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
    metrics: Dict[str, Any],
    wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    commit: bool = True,
) -> None:
    """Logs metrics to W&B (if enabled) and standard logger.

    Assumes the 'global_step' key is present in the metrics dictionary.
    MODIFIED: Logs metrics to console line-by-line for readability with spacing.
    MODIFIED: Skips logging 'final/codecarbon_emissions_kgCO2e' to console.

    Args:
        metrics: Dictionary of metric names and values, MUST include 'global_step'.
        wandb_run: The active W&B run object. If None, tries to use the global run.
        commit: If True (default), commits the log to W&B. Set to False to batch logs.
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

    active_run = wandb_run or wandb.run
    if active_run:
        try:
            active_run.log(metrics, commit=commit)
        except Exception as e:
            logger.error(
                f"Failed to log metrics to Weights & Biases: {e}", exc_info=True
            )
