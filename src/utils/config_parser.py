"""Utilities for loading and parsing YAML configuration files."""

import logging
import os
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def _load_one_yaml(path: str) -> Dict[str, Any]:
    """
    Loads a single YAML file.

    Returns an empty dictionary if the file is empty.
    Raises FileNotFoundError or YAMLError on failure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path) as f:
            config = yaml.safe_load(f)
        # Ensure that an empty file is treated as an empty dict
        return config if isinstance(config, dict) else {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {path}: {e}")
        raise
    except OSError as e:
        logger.error(f"Error reading configuration file {path}: {e}")
        raise


def _deep_merge(source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
    """Deeply merges the source dictionary into the destination dictionary."""
    for key, value in source.items():
        if (
            isinstance(value, dict)
            and key in destination
            and isinstance(destination[key], dict)
        ):
            # If the key exists in both and both values are dicts, recurse
            _deep_merge(value, destination[key])
        else:
            # Otherwise, overwrite the destination's value with the source's
            destination[key] = value
    return destination


def load_config(
    config_path: str, base_config_path: str = "configs/base.yaml"
) -> Dict[str, Any]:
    """Loads a YAML configuration file and merges it with a base configuration file.

    Args:
        config_path: Path to the specific experiment configuration file.
        base_config_path: Path to the base configuration file.
            Defaults to 'configs/base.yaml'.

    Returns:
        A dictionary containing the merged configuration.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        yaml.YAMLError: If there is an error parsing the YAML files.
        IOError: For other file reading issues.
    """
    base_config = {}
    if os.path.exists(base_config_path):
        logger.debug(f"Loading base configuration from {base_config_path}")
        base_config = _load_one_yaml(base_config_path)
    else:
        logger.warning(
            f"Base configuration file not found at {base_config_path}. Proceeding without it."
        )

    logger.debug(f"Loading specific configuration from {config_path}")
    specific_config = _load_one_yaml(config_path)

    # Start with a copy of the base config and merge the specific config into it
    merged_config = base_config.copy()
    _deep_merge(specific_config, merged_config)

    logger.info(
        f"Successfully merged configuration from {config_path} and {base_config_path}"
    )
    return merged_config
