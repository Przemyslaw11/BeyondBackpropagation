# File: src/utils/logging_utils.py
import wandb
import os
import logging
from typing import Dict, Any, Optional, List  # Added List
import sys  # For stdout handler


# --- Centralized Logging Setup ---
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configures the root logger.

    Args:
        log_level: Logging level string (e.g., 'DEBUG', 'INFO', 'WARNING').
        log_file: Optional path to a file for logging.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Get root logger
    root_logger = logging.getLogger()
    # Set level ONLY if handlers are not present or if this setup is meant to override
    if (
        not root_logger.hasHandlers()
        or os.environ.get("LOGGING_SETUP_COMPLETE") is None
    ):
        root_logger.setLevel(level)

        # Clear existing handlers (optional, prevents duplicate logs if called multiple times within same initial setup)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()  # Close handlers before removing

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File Handler (if specified)
        if log_file:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="a")  # Append mode
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging to file: {log_file}")

        os.environ["LOGGING_SETUP_COMPLETE"] = (
            "1"  # Flag to prevent resetting level/handlers
        )
        root_logger.info(f"Root logger setup complete. Level: {log_level.upper()}")
    else:
        root_logger.info("Root logger already configured.")


# Get logger instance *after* potential setup
logger = logging.getLogger(__name__)


def setup_wandb(
    config: Dict[str, Any],
    project_name: str = "BeyondBackpropagation",
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[List[str]] = None,
    job_type: str = "training",  # Added job_type
) -> Optional["wandb.sdk.wandb_run.Run"]:  # Use quotes for type hint
    """
    Initializes a Weights & Biases run.

    Args:
        config: Dictionary containing the experiment configuration.
        project_name: Name of the W&B project (fallback).
        entity: W&B entity (username or team name). Reads from WANDB_ENTITY env var if None.
        run_name: Optional name for the W&B run. If None, W&B generates one based on experiment_name.
        notes: Optional notes for the W&B run.
        tags: Optional list of tags for the W&B run.
        job_type: Type of job (e.g., 'training', 'evaluation', 'tuning').

    Returns:
        The initialized W&B run object, or None if W&B is disabled or fails.
    """
    # Check if W&B is enabled in config
    wandb_config = config.get("logging", {}).get("wandb", {})
    if not wandb_config.get("use_wandb", True):
        logger.info("Weights & Biases logging is disabled in the configuration.")
        return None

    try:
        # Ensure API key is set (usually via environment variable WANDB_API_KEY)
        if not os.getenv("WANDB_API_KEY"):
            logger.warning(
                "WANDB_API_KEY environment variable not set. W&B logging might fail or prompt."
            )
            # You might choose to return None here if API key is strictly required
            # return None

        # Determine entity
        resolved_entity = (
            entity or os.getenv("WANDB_ENTITY") or wandb_config.get("entity")
        )
        if not resolved_entity:
            logger.warning(
                "W&B entity not specified via args, config, or WANDB_ENTITY env var. Using W&B default."
            )

        # Determine project name
        resolved_project = wandb_config.get("project", project_name)

        # Determine run name (can be constructed from config for better identification)
        resolved_run_name = (
            run_name or wandb_config.get("run_name") or config.get("experiment_name")
        )  # Use explicit, then config, then experiment name
        if not resolved_run_name:
            resolved_run_name = f"run_{int(time.time())}"  # Fallback name

        run = wandb.init(
            project=resolved_project,
            entity=resolved_entity,
            config=config,  # Log the entire configuration
            name=resolved_run_name,
            notes=notes,
            tags=tags,
            job_type=job_type,  # Log job type
            reinit=True,
            # Allow calling init multiple times, but manage state carefully
            # E.g., use separate runs for optuna trials vs final run
            # Consider setting WANDB_RUN_ID environment variable for resuming
        )
        logger.info(f"Weights & Biases run initialized: {run.url}")
        return run
    except ImportError:
        logger.error(
            "wandb library not found. Please install it (`pip install wandb`) to use W&B logging."
        )
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Weights & Biases: {e}", exc_info=True)
        return None


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
    commit: bool = True,  # Allow controlling commit behavior
):
    """
    Logs metrics to W&B (if enabled) and standard logger.

    Args:
        metrics: Dictionary of metric names and values.
        step: Optional step number (e.g., epoch or batch number).
        wandb_run: The active W&B run object. If None, tries to use the global run.
        commit: If True (default), commits the log to W&B. Set to False to batch logs.
    """
    # Log to standard logger
    metrics_str = ", ".join(
        [
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        ]
    )
    step_str = f"Step {step}: " if step is not None else ""
    logger.info(f"{step_str}{metrics_str}")

    # Log to W&B
    active_run = wandb_run or wandb.run
    if active_run:
        try:
            active_run.log(metrics, step=step, commit=commit)
        except Exception as e:
            logger.error(
                f"Failed to log metrics to Weights & Biases: {e}", exc_info=True
            )
