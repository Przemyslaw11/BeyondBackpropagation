"""Updates BP config files with the best hyperparameters from an Optuna study."""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import Optional

import optuna
import yaml

# --- Basic Logging Setup (before potentially loading project config) ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Update a baseline YAML config file with the best hyperparameters "
            "from an Optuna study."
        ),
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
        help="Path to the baseline YAML configuration file to update.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help=(
            "Name of the Optuna study within the database. If None, attempts to "
            "load the first/only study."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable automatic backup of the original config file.",
    )
    return parser.parse_args()


def update_config_from_optuna(
    db_path: str, config_path: str, study_name: Optional[str], create_backup: bool
) -> bool:
    """Loads an Optuna study and updates an optimizer config.

    This function finds the best trial from a study and updates the optimizer
    section of the specified YAML configuration file.

    Args:
        db_path: Path to the Optuna database (.db).
        config_path: Path to the YAML config file to update.
        study_name: Optional name of the study.
        create_backup: Whether to create a backup of the original config file.

    Returns:
        True if the update was successful, False otherwise.
    """
    # --- Validate Input Paths ---
    if not os.path.exists(db_path):
        logger.error(f"Optuna database file not found: {db_path}")
        return False
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return False

    # --- Load Optuna Study ---
    storage_url = f"sqlite:///{db_path}"
    try:
        logger.info(f"Loading Optuna study from storage: {storage_url}")
        loaded_studies = optuna.study.get_all_study_summaries(storage=storage_url)
        if not loaded_studies:
            logger.error(f"No studies found in the database: {db_path}")
            return False

        if study_name is None:
            if len(loaded_studies) > 1:
                logger.error(
                    f"Multiple studies found in {db_path}. Please specify one using --study-name:"
                )
                for study in loaded_studies:
                    logger.error(f"  - {study.study_name}")
                return False
            study_name = loaded_studies[0].study_name
            logger.info(f"Automatically selected study: '{study_name}'")
        elif study_name not in [s.study_name for s in loaded_studies]:
            logger.error(
                f"Specified study name '{study_name}' not found in the database: {db_path}"
            )
            logger.error("Available studies:")
            for study in loaded_studies:
                logger.error(f"  - {study.study_name}")
            return False

        study = optuna.load_study(study_name=study_name, storage=storage_url)
        logger.info(f"Successfully loaded study '{study.study_name}'")

    except ImportError:
        logger.error("Optuna library not found. Install with `pip install optuna`")
        return False
    except Exception as e:
        logger.error(
            f"Failed to load Optuna study '{study_name}' from {db_path}: {e}",
            exc_info=True,
        )
        return False

    # --- Get Best Trial ---
    try:
        best_trial = study.best_trial
        logger.info(
            f"Best trial found: Number {best_trial.number}, Value: {best_trial.value:.6f}"
        )
        logger.info("Best Hyperparameters:")
        best_params = best_trial.params
        if not best_params:
            logger.warning(
                f"Best trial {best_trial.number} has no parameters recorded. Cannot update config."
            )
            return False  # Nothing to update
        for key, value in best_params.items():
            logger.info(f"  - {key}: {value}")
    except ValueError:
        logger.error(
            f"No completed trials found in study '{study.study_name}'. Cannot "
            "determine best parameters."
        )
        return False
    except Exception as e:
        logger.error(
            f"Error retrieving best trial from study '{study.study_name}': {e}",
            exc_info=True,
        )
        return False

    # --- Load YAML Config ---
    try:
        logger.info(f"Loading YAML configuration file: {config_path}")
        with open(config_path) as f:
            # Use safe_load for security
            config_data = yaml.safe_load(f)
        if not isinstance(config_data, dict):
            logger.error(
                f"Failed to parse YAML or file is not a dictionary: {config_path}"
            )
            return False
        logger.info("YAML loaded successfully.")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}", exc_info=True)
        return False
    except OSError as e:
        logger.error(f"Error reading YAML file {config_path}: {e}", exc_info=True)
        return False

    # --- Update Optimizer Section ---
    if "optimizer" not in config_data or not isinstance(
        config_data.get("optimizer"), dict
    ):
        # Error out if 'optimizer' section is missing, as it's safer.
        logger.error(
            f"YAML file {config_path} is missing the 'optimizer' dictionary "
            "section. Cannot update."
        )
        return False

    optimizer_section = config_data["optimizer"]
    updated_values = {}
    keys_updated = []

    logger.info("Updating optimizer section with best parameters...")
    for optuna_key, value in best_params.items():
        # Map Optuna param names to YAML config keys
        if optuna_key == "lr":
            yaml_key = "lr"
        elif optuna_key == "wd":  # Optuna objective uses 'wd'
            yaml_key = "weight_decay"  # Config uses 'weight_decay'
        elif optuna_key == "momentum":  # Handle momentum if tuned
            yaml_key = "momentum"
        else:
            continue

        # Update the value in the loaded config dictionary
        old_value = optimizer_section.get(yaml_key, "NOT_PRESENT")
        optimizer_section[yaml_key] = value
        updated_values[yaml_key] = value
        keys_updated.append(yaml_key)
        logger.info(f"  Mapped '{optuna_key}' to '{yaml_key}': {old_value} -> {value}")

    if not updated_values:
        logger.warning(
            "No relevant optimizer parameters (lr, wd, momentum) were found "
            "in the best trial params to update the config."
        )
        # Don't treat as failure, maybe study didn't tune expected params
        return True

    # --- Create Backup (Optional) ---
    if create_backup:
        backup_path = f"{config_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copyfile(config_path, backup_path)
            logger.info(f"Created backup of original config: {backup_path}")
        except Exception as e:
            logger.error(
                f"Failed to create backup file {backup_path}: {e}", exc_info=True
            )
            logger.warning("Proceeding without backup.")

    # --- Write Updated YAML ---
    try:
        logger.info(f"Writing updated configuration back to: {config_path}")
        with open(config_path, "w") as f:
            # Dump with settings to preserve formatting better
            yaml.dump(
                config_data, f, default_flow_style=False, sort_keys=False, indent=2
            )
        logger.info(f"YAML file updated successfully. Keys updated: {keys_updated}")
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


def main() -> None:
    """Main execution function."""
    args = parse_args()

    logger.info("Starting configuration update process.")
    success = update_config_from_optuna(
        db_path=args.db_path,
        config_path=args.config_path,
        study_name=args.study_name,
        create_backup=not args.no_backup,
    )

    if success:
        logger.info("Configuration update process completed successfully.")
        sys.exit(0)
    else:
        logger.error("Configuration update process failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
