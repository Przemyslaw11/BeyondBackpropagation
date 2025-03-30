# File: scripts/run_experiment.py
#!/usr/bin/env python
import argparse
import pprint
import sys
import os
import yaml  # For safe_load and error handling
import logging  # For basic logging before config setup

# Add src directory to Python path to allow importing modules from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import necessary functions AFTER path modification
from src.utils.config_parser import load_config
from src.utils.helpers import set_seed, create_directory_if_not_exists
from src.training.engine import run_training  # Import the main training engine function
from src.utils.logging_utils import setup_logging, logger  # Import logger and setup

# Setup basic logging config initially
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main(args):
    """
    Main function to run an experiment.
    """
    print(f"Loading and merging configuration from: {args.config}")
    try:
        # Load and merge base and experiment configurations
        config = load_config(args.config)
        print("Configuration loaded and merged successfully:")
        pprint.pprint(config)

        # --- Setup Logging based on loaded config ---
        log_config = config.get("logging", {})
        # Define default log file path if not specified
        results_dir = config.get("results", {}).get("dir", "results")
        exp_name = config.get(
            "experiment_name", os.path.splitext(os.path.basename(args.config))[0]
        )
        default_log_file = os.path.join(results_dir, exp_name, f"{exp_name}_run.log")
        log_file_path = log_config.get("log_file", default_log_file)

        # Ensure results directory exists for the log file
        log_dir = os.path.dirname(log_file_path)
        create_directory_if_not_exists(log_dir)

        setup_logging(log_level=log_config.get("level", "INFO"), log_file=log_file_path)
        # Re-get logger instance after setup if needed elsewhere, though basicConfig sets root
        # logger = logging.getLogger(__name__) # Usually not needed after basicConfig/root setup

    except (FileNotFoundError, yaml.YAMLError, IOError) as e:
        logger.error(f"Error loading or merging configuration: {e}", exc_info=True)
        sys.exit(1)  # Exit if config loading fails
    except Exception as e:
        logger.error(f"An unexpected error occurred during setup: {e}", exc_info=True)
        sys.exit(1)

    # --- Start Training ---
    # Seed is set within the train function using the loaded config
    logger.info("\n--- Starting Experiment ---")
    try:
        # Call the main training function from engine.py
        # W&B setup is handled within run_training
        results = run_training(
            config, wandb_run=None
        )  # Pass config, W&B run is handled internally
        logger.info("\n--- Experiment Finished ---")
        logger.info("Results:")
        # Use logger for results too
        results_str = pprint.pformat(results)
        for line in results_str.split("\n"):
            logger.info(line)

        if "error" in results:
            logger.error(
                f"Warning: Experiment completed with error: {results['error']}"
            )
            sys.exit(1)  # Exit with error status if training reported an error
        else:
            logger.info("\n--- Experiment Finished Successfully ---")

    except Exception as e:
        logger.critical(f"\n--- Experiment Failed ---")
        logger.critical(
            f"Error during training: {e}", exc_info=True
        )  # Log full traceback
        # Optionally re-raise the exception for more detailed traceback
        # raise e
        sys.exit(1)  # Exit with error status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a deep learning experiment based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the experiment.",
    )

    args = parser.parse_args()
    main(args)
