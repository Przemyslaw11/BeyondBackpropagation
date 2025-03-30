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
import yaml # Import yaml to catch YAMLError
from src.utils.config_parser import load_and_merge_config # Use the merging function
from src.utils.helpers import set_seed
from src.training.engine import train # Import the main training function

def main(args):
    """
    Main function to run an experiment.
    """
    print(f"Loading and merging configuration from: {args.config}")
    try:
        # Load and merge base and experiment configurations
        config = load_and_merge_config(args.config)
        print("Configuration loaded and merged successfully:")
        pprint.pprint(config) # Pretty print the final config
    except (FileNotFoundError, yaml.YAMLError, IOError) as e:
        print(f"Error loading or merging configuration: {e}")
        sys.exit(1) # Exit if config loading fails

    # --- Start Training ---
    # Seed is set within the train function using the loaded config
    print("\n--- Starting Experiment ---")
    try:
        train(config) # Call the main training function
        print("\n--- Experiment Finished Successfully ---")
    except Exception as e:
        print(f"\n--- Experiment Failed ---")
        print(f"Error during training: {e}")
        # Optionally re-raise the exception for more detailed traceback
        # raise e
        sys.exit(1) # Exit with error status

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a deep learning experiment based on a configuration file.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file for the experiment.')

    args = parser.parse_args()
    main(args)
