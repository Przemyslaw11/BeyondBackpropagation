# File: src/utils/config_parser.py
import yaml
import os
from typing import Dict, Any
import logging  # Use logging

# Get logger instance for this module
logger = logging.getLogger(__name__)


def load_config(
    config_path: str, base_config_path: str = "configs/base.yaml"
) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and merges it with a base configuration file.

    Args:
        config_path: Path to the specific experiment configuration file.
        base_config_path: Path to the base configuration file. Defaults to 'configs/base.yaml'.

    Returns:
        A dictionary containing the merged configuration.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        yaml.YAMLError: If there is an error parsing the YAML files.
        IOError: For other file reading issues.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    base_config = {}
    if os.path.exists(base_config_path):
        try:
            with open(base_config_path, "r") as f:
                base_config = yaml.safe_load(f)
                if base_config is None:  # Handle empty base file
                    base_config = {}
            logger.debug(f"Loaded base configuration from {base_config_path}")
        except yaml.YAMLError as e:
            logger.error(
                f"Error parsing base configuration file {base_config_path}: {e}"
            )
            raise
        except IOError as e:
            logger.error(
                f"Error reading base configuration file {base_config_path}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while reading {base_config_path}: {e}"
            )
            raise
    else:
        logger.warning(
            f"Base configuration file not found at {base_config_path}. Proceeding without it."
        )

    specific_config = {}
    try:
        with open(config_path, "r") as f:
            specific_config = yaml.safe_load(f)
            if specific_config is None:  # Handle empty specific config file
                specific_config = {}
        logger.debug(f"Loaded specific configuration from {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing specific configuration file {config_path}: {e}")
        raise
    except IOError as e:
        logger.error(f"Error reading specific configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {config_path}: {e}")
        raise

    # --- Deep Merge Implementation ---
    def deep_merge(source, destination):
        """Deeply merges source dict into destination dict."""
        for key, value in source.items():
            if isinstance(value, dict):
                # Get node or create one
                node = destination.setdefault(key, {})
                deep_merge(value, node)
            else:
                destination[key] = value
        return destination

    # Merge configurations: specific config overrides base config using deep merge
    merged_config = base_config.copy()  # Start with a copy of the base
    merged_config = deep_merge(specific_config, merged_config)

    logger.info(
        f"Successfully merged configuration from {config_path} and {base_config_path}"
    )
    return merged_config
