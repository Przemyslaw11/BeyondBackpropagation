#!/usr/bin/env python
import argparse
import pprint
import sys
import os

# Add src directory to Python path to allow importing modules from src
# This assumes the script is run from the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config_parser import load_config
from src.utils.helpers import set_seed

def main(args):
    """
    Main function to run an experiment.
    """
    print(f"Loading configuration from: {args.config}")
    try:
        # TODO: Eventually use load_and_merge_config if implemented
        config = load_config(args.config)
        print("Configuration loaded successfully:")
        pprint.pprint(config) # Pretty print the config for verification
    except (FileNotFoundError, yaml.YAMLError, IOError) as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1) # Exit if config loading fails

    # Set random seed for reproducibility
    seed = config.get('seed', None) # Get seed from config, default to None
    set_seed(seed)

    # --- Placeholder for future steps ---
    print("\n--- Experiment Setup Complete ---")
    print("Next steps would involve:")
    print("1. Setting up logging (W&B)")
    print("2. Loading dataset")
    print("3. Initializing model architecture")
    print("4. Initializing algorithm/training logic")
    print("5. Starting the training engine")
    # --- End Placeholder ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a deep learning experiment based on a configuration file.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file for the experiment.')

    args = parser.parse_args()
    main(args)
