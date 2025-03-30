import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is not valid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: # Handle empty YAML file case
             config = {}
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    except Exception as e:
        # Catch other potential file reading errors
        raise IOError(f"Error reading file {config_path}: {e}")

# TODO: Implement merging with base.yaml functionality if needed later.
# def load_and_merge_config(config_path: str, base_config_path: str = "configs/base.yaml") -> Dict[str, Any]:
#     """Loads experiment config and merges it with a base config."""
#     base_config = load_config(base_config_path)
#     experiment_config = load_config(config_path)
#
#     # Simple dictionary update (experiment overrides base)
#     # More sophisticated merging (e.g., deep merge) could be added if needed
#     merged_config = base_config.copy()
#     merged_config.update(experiment_config)
#
#     return merged_config

if __name__ == '__main__':
    # Example usage (for testing purposes)
    try:
        # Create a dummy test config file
        dummy_config_path = "dummy_test_config.yaml"
        with open(dummy_config_path, 'w') as f:
            yaml.dump({'learning_rate': 0.01, 'optimizer': 'adam'}, f)

        print(f"Loading dummy config: {dummy_config_path}")
        loaded_settings = load_config(dummy_config_path)
        print("Loaded settings:")
        print(loaded_settings)

        # Clean up dummy file
        os.remove(dummy_config_path)

        # Test loading base config
        print("\nLoading base config: configs/base.yaml")
        base_settings = load_config("configs/base.yaml")
        print("Loaded base settings:")
        print(base_settings)

    except (FileNotFoundError, yaml.YAMLError, IOError) as e:
        print(f"Error during test: {e}")
