# File: scripts/run_experiment.py
#!/usr/bin/env python
import argparse
import pprint
import sys
import os

# Add src directory to Python path to allow importing modules from src
# This assumes the script is run from the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import necessary functions
import yaml  # Import yaml to catch YAMLError
from src.utils.config_parser import load_config  # FIXED: Use the correct function name
from src.utils.helpers import set_seed
from src.training.engine import run_training  # Import the main training engine function


def main(args):
    """
    Main function to run an experiment.
    """
    print(f"Loading and merging configuration from: {args.config}")
    try:
        # Load and merge base and experiment configurations
        config = load_config(args.config)  # FIXED: Use the correct function name
        print("Configuration loaded and merged successfully:")
        pprint.pprint(config)  # Pretty print the final config
    except (FileNotFoundError, yaml.YAMLError, IOError) as e:
        print(f"Error loading or merging configuration: {e}")
        sys.exit(1)  # Exit if config loading fails

    # --- Start Training ---
    # Seed is set within the train function using the loaded config
    print("\n--- Starting Experiment ---")
    try:
        # Call the main training function from engine.py
        # W&B setup is handled within run_training
        results = run_training(
            config, wandb_run=None
        )  # Pass config, W&B run is handled internally
        print("\n--- Experiment Finished ---")
        print("Results:")
        pprint.pprint(results)
        if "error" in results:
            print(f"Warning: Experiment completed with error: {results['error']}")
            sys.exit(1)  # Exit with error status if training reported an error
        else:
            print("\n--- Experiment Finished Successfully ---")

    except Exception as e:
        print(f"\n--- Experiment Failed ---")
        print(f"Error during training: {e}", exc_info=True)  # Log full traceback
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
