# File: ./scripts/run_experiment.py
"""Script to run a single experiment based on a YAML config file."""

import argparse
import logging
import os
import pprint
import sys

import yaml
from dotenv import load_dotenv

from src.utils.backend_policy import get_execution_backend
from src.training.engine import run_training
from src.utils.config_parser import load_config
from src.utils.helpers import create_directory_if_not_exists
from src.utils.logging_utils import logger, setup_logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main(args: argparse.Namespace) -> None:
    """Run a single experiment based on a config file."""
    print(f"Loading and merging configuration from: {args.config}")
    try:
        config = load_config(args.config)
        if args.backend:
            config.setdefault("general", {})["backend"] = args.backend
        backend = get_execution_backend(config)
        print("Configuration loaded and merged successfully:")
        pprint.pprint(config)

        exp_name = config.get(
            "experiment_name", os.path.splitext(os.path.basename(args.config))[0]
        )
        log_file_path = backend.resolve_log_file(config, exp_name)

        log_dir = os.path.dirname(log_file_path)
        create_directory_if_not_exists(log_dir)

        log_level = config.get("logging", {}).get("level", "INFO")
        setup_logging(log_level=log_level, log_file=log_file_path)

    except (OSError, FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"Error loading or merging configuration: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during setup: {e}", exc_info=True)
        sys.exit(1)

    logger.info("\n--- Starting Experiment ---")
    try:
        results = run_training(config, wandb_run=None)
        results.pop("codecarbon_emissions_kgCO2e", None)
        logger.debug("Removed kgCO2e emissions from results dict before printing.")
        logger.info("\n--- Experiment Finished ---")
        logger.info("Results:")
        results_str = pprint.pformat(results)
        for line in results_str.split("\n"):
            logger.info(line)

        if "error" in results:
            logger.error(
                f"Warning: Experiment completed with error: {results['error']}"
            )
            sys.exit(1)
        else:
            logger.info("\n--- Experiment Finished Successfully ---")

    except Exception as e:
        logger.critical("\n--- Experiment Failed ---")
        logger.critical(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run a deep learning experiment based on a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the experiment.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["slurm", "local"],
        default=None,
        help="Execution backend to use. Defaults to the config value or Slurm.",
    )

    cli_args = parser.parse_args()
    main(cli_args)
