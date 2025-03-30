# File: src/utils/config_parser.py
import yaml
import os
from typing import Dict, Any
import logging  # Use logging

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

    # Merge configurations: specific config overrides base config
    # Using a simple top-level merge for now.
    # A deep merge function would be needed for nested dictionaries if required.
    merged_config = base_config.copy()
    merged_config.update(specific_config)  # Simple top-level merge

    logger.info(
        f"Successfully merged configuration from {config_path} and {base_config_path}"
    )
    return merged_config


# Removed the __main__ block with dummy file creation for cleaner utils file
# Testing should ideally be done via separate unit test files.
