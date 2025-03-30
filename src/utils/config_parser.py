import yaml
import os
from typing import Dict, Any

def load_config(config_path: str, base_config_path: str = 'configs/base.yaml') -> Dict[str, Any]:
    """
    Loads a YAML configuration file and merges it with a base configuration file.

    Args:
        config_path: Path to the specific experiment configuration file.
        base_config_path: Path to the base configuration file. Defaults to 'configs/base.yaml'.

    Returns:
        A dictionary containing the merged configuration.

    Raises:
        FileNotFoundError: If either the config_path or base_config_path does not exist.
        yaml.YAMLError: If there is an error parsing the YAML files.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if not os.path.exists(base_config_path):
        # Allow running without a base config if it's explicitly not found or not needed
        print(f"Warning: Base configuration file not found at {base_config_path}. Proceeding without it.")
        base_config = {}
    else:
        try:
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
                if base_config is None: # Handle empty base file
                    base_config = {}
        except yaml.YAMLError as e:
            print(f"Error parsing base configuration file {base_config_path}: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while reading {base_config_path}: {e}")
            raise

    try:
        with open(config_path, 'r') as f:
            specific_config = yaml.safe_load(f)
            if specific_config is None: # Handle empty specific config file
                specific_config = {}
    except yaml.YAMLError as e:
        print(f"Error parsing specific configuration file {config_path}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while reading {config_path}: {e}")
        raise

    # Merge configurations: specific config overrides base config
    # Use a deep merge strategy if necessary, but for now, a simple update might suffice
    # depending on the config structure. Let's start simple.
    merged_config = base_config.copy()
    merged_config.update(specific_config) # Simple top-level merge

    # TODO: Implement deep merging if nested dictionaries need careful merging later

    return merged_config

if __name__ == '__main__':
    # Example usage (assuming you have dummy files)
    # Create dummy files for testing:
    if not os.path.exists('configs'):
        os.makedirs('configs')
    with open('configs/base.yaml', 'w') as f:
        yaml.dump({'learning_rate': 0.01, 'optimizer': 'Adam', 'epochs': 50, 'seed': 42}, f)
    with open('configs/dummy_exp.yaml', 'w') as f:
        yaml.dump({'learning_rate': 0.005, 'dataset': 'CIFAR10', 'model': {'type': 'CNN'}}, f)

    try:
        config = load_config('configs/dummy_exp.yaml')
        print("Loaded and merged config:")
        print(yaml.dump(config, default_flow_style=False))

        # Test loading without base
        # os.remove('configs/base.yaml') # Temporarily remove base
        # config_no_base = load_config('configs/dummy_exp.yaml')
        # print("\nLoaded config without base:")
        # print(yaml.dump(config_no_base, default_flow_style=False))
        # # Recreate base for future tests
        # with open('configs/base.yaml', 'w') as f:
        #     yaml.dump({'learning_rate': 0.01, 'optimizer': 'Adam', 'epochs': 50, 'seed': 42}, f)

        # Test non-existent file
        # load_config('non_existent_config.yaml')

    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"\nError during example usage: {e}")
    finally:
        # Clean up dummy files
        # os.remove('configs/dummy_exp.yaml')
        # os.remove('configs/base.yaml')
        # if not os.listdir('configs'):
        #     os.rmdir('configs')
        pass # Keep dummy files for now as they might be useful

    # Example loading the actual base config (if it exists)
    try:
        base_cfg = load_config('configs/base.yaml', base_config_path='non_existent_base.yaml') # Load base as specific, ignore missing base
        print("\nLoaded actual base config:")
        print(yaml.dump(base_cfg, default_flow_style=False))
    except FileNotFoundError as e:
        print(f"\nCould not load actual base config: {e}")
    except yaml.YAMLError as e:
        print(f"\nError parsing actual base config: {e}")
