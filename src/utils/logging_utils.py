import wandb
import os
import logging
from typing import Dict, Any, Optional

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_wandb(config: Dict[str, Any], project_name: str = "BeyondBackpropagation", entity: Optional[str] = None, run_name: Optional[str] = None, notes: Optional[str] = None, tags: Optional[list[str]] = None) -> Optional[wandb.sdk.wandb_run.Run]:
    """
    Initializes a Weights & Biases run.

    Args:
        config: Dictionary containing the experiment configuration.
        project_name: Name of the W&B project.
        entity: W&B entity (username or team name). Reads from WANDB_ENTITY env var if None.
        run_name: Optional name for the W&B run. If None, W&B generates one.
        notes: Optional notes for the W&B run.
        tags: Optional list of tags for the W&B run.

    Returns:
        The initialized W&B run object, or None if W&B is disabled or fails.
    """
    if not config.get('wandb', {}).get('enabled', True):
        logger.info("Weights & Biases logging is disabled in the configuration.")
        return None

    try:
        # Ensure API key is set (usually via environment variable WANDB_API_KEY)
        if not os.getenv('WANDB_API_KEY'):
            logger.warning("WANDB_API_KEY environment variable not set. W&B logging might fail.")
            # Optionally, you could disable W&B here or let it try and potentially fail.
            # return None # Uncomment to strictly require API key

        # Determine entity
        resolved_entity = entity or os.getenv('WANDB_ENTITY') or config.get('wandb', {}).get('entity')
        if not resolved_entity:
            logger.warning("W&B entity not specified via args, config, or WANDB_ENTITY env var. Using default.")
            # W&B might use a default entity or prompt, depending on setup.

        # Determine project name
        resolved_project = config.get('wandb', {}).get('project', project_name)

        # Determine run name (can be constructed from config for better identification)
        if run_name is None:
            run_name = config.get('run_name') # Allow specifying directly in config
        # Example: Construct a more descriptive name if not provided
        # if run_name is None:
        #     run_name = f"{config.get('algorithm', 'unknown')}_{config.get('dataset', 'unknown')}_{config.get('model', {}).get('type', 'unknown')}"


        run = wandb.init(
            project=resolved_project,
            entity=resolved_entity,
            config=config,  # Log the entire configuration
            name=run_name,
            notes=notes,
            tags=tags,
            reinit=True, # Allows calling init multiple times in the same process (useful for Optuna)
            # mode="disabled" # Use this to temporarily disable W&B without changing config
        )
        logger.info(f"Weights & Biases run initialized: {run.url}")
        return run
    except Exception as e:
        logger.error(f"Failed to initialize Weights & Biases: {e}", exc_info=True)
        return None

def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None, wandb_run: Optional[wandb.sdk.wandb_run.Run] = None):
    """
    Logs metrics to W&B (if enabled) and standard logger.

    Args:
        metrics: Dictionary of metric names and values.
        step: Optional step number (e.g., epoch or batch number).
        wandb_run: The active W&B run object. If None, tries to use the global run.
    """
    # Log to standard logger
    metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
    step_str = f"Step {step}: " if step is not None else ""
    logger.info(f"{step_str}{metrics_str}")

    # Log to W&B
    active_run = wandb_run or wandb.run
    if active_run:
        try:
            active_run.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics to Weights & Biases: {e}", exc_info=True)


if __name__ == '__main__':
    # Example Usage (requires WANDB_API_KEY to be set in environment)
    print("Testing logging_utils...")

    # Dummy config
    dummy_config = {
        'learning_rate': 0.01,
        'epochs': 10,
        'algorithm': 'TEST',
        'dataset': 'DummyData',
        'model': {'type': 'TestNet'},
        'seed': 123,
        'wandb': {
            'enabled': True, # Set to False to test disabling
            'project': 'Test-Project-Logging',
            # 'entity': 'your_entity' # Optional: replace or set WANDB_ENTITY
        }
    }

    print("\nAttempting to set up W&B...")
    # Note: This will actually try to connect to W&B if API key is available
    # Set WANDB_MODE=offline for offline testing
    # os.environ['WANDB_MODE'] = 'offline'
    run = setup_wandb(dummy_config, run_name="logging_util_test", tags=["test", "utils"])

    if run:
        print(f"W&B Run URL: {run.url}")
        print("\nLogging example metrics...")
        log_metrics({'accuracy': 95.5, 'loss': 0.1234}, step=1, wandb_run=run)
        log_metrics({'val_accuracy': 92.1, 'val_loss': 0.2345}, step=1, wandb_run=run)
        log_metrics({'epoch_time': 60.5}, wandb_run=run) # Log without step

        # Finish the run (important!)
        run.finish()
        print("W&B run finished.")
    else:
        print("W&B setup failed or was disabled.")

    # Test logging without W&B run active (should just log to console)
    print("\nTesting log_metrics without active W&B run...")
    log_metrics({'test_metric': 1.23}, step=5)

    # Test disabling W&B via config
    print("\nTesting W&B disabled via config...")
    dummy_config['wandb']['enabled'] = False
    disabled_run = setup_wandb(dummy_config)
    if disabled_run is None:
        print("W&B correctly disabled.")
        log_metrics({'accuracy': 99.9}, step=10, wandb_run=disabled_run) # Should only log to console
    else:
        print("Error: W&B was not disabled as expected.")
        if disabled_run: disabled_run.finish()
