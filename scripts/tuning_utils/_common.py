"""Shared implementation for the ``update_*_configs`` utilities.

The per-algorithm scripts in this package differ only in the
Optuna-parameter-to-YAML-key mapping, the YAML section they target, and two
edge-case policies (missing target section, empty best trial). The shared
scaffolding lives here so that each algorithm declares its mapping in one
place instead of duplicating the study-loading, backup, and YAML read/write
logic four times.
"""

import argparse
import logging
import os
import shutil
import sys
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def parse_args(description: str) -> argparse.Namespace:
    """Parse the CLI flags shared by every update script.

    Args:
        description: Human-readable description shown in ``--help``.

    Returns:
        Parsed namespace with ``db_path``, ``config_path``, ``study_name``
        and ``no_backup`` attributes.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-path",
        type=str,
        required=True,
        help="Path to the Optuna SQLite database file (.db).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the YAML configuration file to update.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help=(
            "Name of the Optuna study. If None, attempts to load the first/only study."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable automatic backup of the original config file.",
    )
    return parser.parse_args()


def load_best_params(db_path: str, study_name: Optional[str]) -> Optional[dict]:
    """Load the best trial's hyperparameters from an Optuna study.

    Args:
        db_path: Path to the Optuna SQLite database file.
        study_name: Study to load; auto-selects the only study when ``None``.

    Returns:
        The best trial's parameter dict (possibly empty when the trial
        recorded no parameters), or ``None`` on any validation or loading
        failure (errors are logged).
    """
    if not os.path.exists(db_path):
        logger.error(f"Optuna database file not found: {db_path}")
        return None

    try:
        import optuna
    except ImportError:
        logger.error("Optuna library not found. Install with `pip install optuna`")
        return None

    storage_url = f"sqlite:///{db_path}"
    try:
        logger.info(f"Loading Optuna study from storage: {storage_url}")
        loaded_studies = optuna.study.get_all_study_summaries(storage=storage_url)
        if not loaded_studies:
            logger.error(f"No studies found in the database: {db_path}")
            return None

        if study_name is None:
            if len(loaded_studies) > 1:
                logger.error(
                    f"Multiple studies found in {db_path}. Specify --study-name."
                )
                return None
            study_name = loaded_studies[0].study_name
            logger.info(f"Automatically selected study: '{study_name}'")
        elif study_name not in [s.study_name for s in loaded_studies]:
            logger.error(
                f"Specified study name '{study_name}' not found in {db_path}. "
                "Available studies:"
            )
            for summary in loaded_studies:
                logger.error(f"  - {summary.study_name}")
            return None

        study = optuna.load_study(study_name=study_name, storage=storage_url)
        logger.info(f"Successfully loaded study '{study.study_name}'")
    except Exception as e:
        logger.error(
            f"Failed to load Optuna study '{study_name}' from {db_path}: {e}",
            exc_info=True,
        )
        return None

    try:
        best_trial = study.best_trial
        logger.info(
            f"Best trial found: Number {best_trial.number}, "
            f"Value: {best_trial.value:.6f}"
        )
        best_params = dict(best_trial.params)
        for key, value in best_params.items():
            logger.info(f"  - {key}: {value}")
        return best_params
    except ValueError:
        logger.error(
            f"No completed trials found in study '{study.study_name}'. Cannot "
            "determine best parameters."
        )
        return None
    except Exception as e:
        logger.error(f"Error retrieving best trial: {e}", exc_info=True)
        return None


def update_config_with_best_params(
    db_path: str,
    config_path: str,
    study_name: Optional[str],
    key_map: Mapping[str, str],
    *,
    section: str = "algorithm_params",
    create_missing_section: bool = True,
    fail_on_empty_params: bool = False,
    create_backup: bool = True,
    label: str = "",
) -> bool:
    """Apply the best Optuna trial's parameters to a YAML config file.

    Args:
        db_path: Path to the Optuna SQLite database file.
        config_path: Path to the YAML configuration file to update in place.
        study_name: Study to load; auto-selects the only study when ``None``.
        key_map: Mapping from Optuna parameter names to YAML keys inside
            ``section``; unmapped trial parameters are ignored.
        section: YAML section to update.
        create_missing_section: When True (the ``algorithm_params``
            convention) a missing section is created with a warning; when
            False (the BP ``optimizer`` convention) a missing section is an
            error.
        fail_on_empty_params: Exit-code policy when the best trial recorded
            no parameters: True fails (legacy BP behavior), False succeeds
            without writing (legacy FF/MF/CaFo behavior).
        create_backup: Write a timestamped ``.bak_YYYYmmdd_HHMMSS`` copy
            next to the config before overwriting it.
        label: Algorithm label used in log messages (e.g. ``"FF"``).

    Returns:
        True when the config was updated (or nothing needed changing),
        False on any failure. Callers translate this into the process exit
        code.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return False

    best_params = load_best_params(db_path, study_name)
    if best_params is None:
        return False
    if not best_params:
        logger.warning("Best trial has no parameters recorded. Cannot update config.")
        return not fail_on_empty_params

    # --- Load YAML Config ---
    try:
        logger.info(f"Loading {label} YAML configuration file: {config_path}")
        with open(config_path) as f:
            config_data: Any = yaml.safe_load(f)
        if not isinstance(config_data, dict):
            logger.error(f"Failed to parse YAML or file not dict: {config_path}")
            return False
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}", exc_info=True)
        return False
    except OSError as e:
        logger.error(f"Error reading YAML file {config_path}: {e}", exc_info=True)
        return False

    # --- Update Target Section ---
    if section not in config_data or not isinstance(config_data.get(section), dict):
        if not create_missing_section:
            logger.error(
                f"YAML file {config_path} is missing the '{section}' dictionary "
                "section. Cannot update."
            )
            return False
        logger.warning(
            f"YAML file {config_path} missing '{section}' dict. Creating it."
        )
        config_data[section] = {}

    target_section = config_data[section]
    keys_updated = []

    logger.info(f"Updating {section} section with best {label} parameters...")
    for optuna_key, value in best_params.items():
        yaml_key = key_map.get(optuna_key)
        if yaml_key is None:
            continue  # Ignore unknown keys
        old_value = target_section.get(yaml_key, "NOT_PRESENT")
        target_section[yaml_key] = value
        keys_updated.append(yaml_key)
        logger.info(f"  Updated '{yaml_key}': {old_value} -> {value}")

    if not keys_updated:
        logger.warning(f"No relevant {label} parameters found in best trial to update.")
        return True

    # --- Create Backup ---
    if create_backup:
        backup_path = f"{config_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copyfile(config_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create backup {backup_path}: {e}", exc_info=True)
            logger.warning("Proceeding without backup.")

    # --- Write Updated YAML ---
    try:
        logger.info(f"Writing updated configuration back to: {config_path}")
        with open(config_path, "w") as f:
            yaml.dump(
                config_data, f, default_flow_style=False, sort_keys=False, indent=2
            )
        logger.info(
            f"{label} YAML file updated successfully. Keys updated: {keys_updated}"
        )
        return True
    except OSError as e:
        logger.error(
            f"Error writing updated YAML file {config_path}: {e}", exc_info=True
        )
        return False
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while writing YAML: {e}", exc_info=True
        )
        return False


def run_main(label: str, **kwargs: Any) -> None:
    """Standard ``main()`` body: run an update and translate it to an exit code.

    Args:
        label: Algorithm label used in the start/success/failure log lines.
        **kwargs: Forwarded to :func:`update_config_with_best_params`.
    """
    logger.info(f"Starting {label} configuration update process.")
    kwargs.setdefault("label", label)
    success = update_config_with_best_params(**kwargs)
    if success:
        logger.info(f"{label} Configuration update process completed successfully.")
        sys.exit(0)
    else:
        logger.error(f"{label} Configuration update process failed.")
        sys.exit(1)
