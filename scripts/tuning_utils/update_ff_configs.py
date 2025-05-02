#!/usr/bin/env python
import argparse
import logging
import optuna
import os
import yaml
import sys
import shutil
from datetime import datetime
from typing import Optional

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update a Forward-Forward (FF) YAML config file with the best hyperparameters from an Optuna study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db-path",type=str,required=True,help="Path to the Optuna SQLite database file (.db).")
    parser.add_argument("--config-path",type=str,required=True,help="Path to the FF YAML configuration file to update.")
    parser.add_argument("--study-name",type=str,default=None,help="Name of the Optuna study. If None, attempts to load the first/only study.")
    parser.add_argument("--no-backup",action="store_true",help="Disable automatic backup of the original config file.")
    return parser.parse_args()

def update_ff_config_from_optuna(
    db_path: str, config_path: str, study_name: Optional[str], create_backup: bool
) -> bool:
    """
    Loads an Optuna study, finds the best trial, and updates the
    algorithm_params section of the specified FF YAML configuration file.
    """
    # --- Validate Input Paths ---
    if not os.path.exists(db_path): logger.error(f"Optuna database file not found: {db_path}"); return False
    if not os.path.exists(config_path): logger.error(f"Configuration file not found: {config_path}"); return False

    # --- Load Optuna Study ---
    storage_url = f"sqlite:///{db_path}"
    try:
        logger.info(f"Loading Optuna study from storage: {storage_url}")
        loaded_studies = optuna.study.get_all_study_summaries(storage=storage_url)
        if not loaded_studies: logger.error(f"No studies found in the database: {db_path}"); return False
        if study_name is None:
            if len(loaded_studies) > 1: logger.error(f"Multiple studies found in {db_path}. Specify --study-name."); return False
            study_name = loaded_studies[0].study_name; logger.info(f"Auto-selected study: '{study_name}'")
        elif study_name not in [s.study_name for s in loaded_studies]: logger.error(f"Study '{study_name}' not found in {db_path}"); return False
        study = optuna.load_study(study_name=study_name, storage=storage_url); logger.info(f"Loaded study '{study.study_name}'")
    except ImportError: logger.error("Optuna library not found."); return False
    except Exception as e: logger.error(f"Failed to load Optuna study '{study_name}': {e}", exc_info=True); return False

    # --- Get Best Trial ---
    try:
        best_trial = study.best_trial; logger.info(f"Best trial: Nr {best_trial.number}, Val: {best_trial.value:.6f}")
        logger.info("Best Hyperparameters (for FF):")
        best_params = best_trial.params
        if not best_params: logger.warning("Best trial has no parameters. Cannot update."); return True
        for key, value in best_params.items(): logger.info(f"  - {key}: {value}")
    except ValueError: logger.error(f"No completed trials found in study '{study.study_name}'."); return False
    except Exception as e: logger.error(f"Error retrieving best trial: {e}", exc_info=True); return False

    # --- Load YAML Config ---
    try:
        logger.info(f"Loading FF YAML configuration file: {config_path}")
        with open(config_path, "r") as f: config_data = yaml.safe_load(f)
        if not isinstance(config_data, dict): logger.error(f"Failed to parse YAML or file not dict: {config_path}"); return False
        logger.info("YAML loaded successfully.")
    except yaml.YAMLError as e: logger.error(f"Error parsing YAML file {config_path}: {e}", exc_info=True); return False
    except IOError as e: logger.error(f"Error reading YAML file {config_path}: {e}", exc_info=True); return False

    # --- Update algorithm_params Section ---
    if "algorithm_params" not in config_data or not isinstance(config_data.get("algorithm_params"), dict):
        logger.warning(f"YAML file {config_path} missing 'algorithm_params' dict. Creating it.")
        config_data["algorithm_params"] = {}

    algo_params_section = config_data["algorithm_params"]
    updated_values = {}; keys_updated = []

    logger.info("Updating algorithm_params section with best FF parameters...")
    for optuna_key, value in best_params.items():
        # Map Optuna param names to YAML config keys
        if optuna_key == "ff_lr": yaml_key = "ff_learning_rate"
        elif optuna_key == "ff_wd": yaml_key = "ff_weight_decay"
        elif optuna_key == "ds_lr": yaml_key = "downstream_learning_rate"
        elif optuna_key == "ds_wd": yaml_key = "downstream_weight_decay"
        # Add mappings for other potential tunable FF hyperparameters here
        # elif optuna_key == "opt_type": yaml_key = "optimizer_type"
        else: continue # Ignore unknown keys

        old_value = algo_params_section.get(yaml_key, "NOT_PRESENT")
        # Convert to string if needed (original FF configs had LRs/WDs as strings)
        # Best practice is numbers, but let's match existing format if present
        # if isinstance(old_value, str) and not isinstance(value, str):
        #     # Format float to scientific notation string similar to original?
        #     # This can be tricky. Let's just save as number for now.
        #     pass
        algo_params_section[yaml_key] = value # Save as number (recommended)
        updated_values[yaml_key] = value; keys_updated.append(yaml_key)
        logger.info(f"  Updated '{yaml_key}': {old_value} -> {value}")

    if not updated_values: logger.warning("No relevant FF parameters found in best trial to update."); return True

    # --- Create Backup ---
    if create_backup:
        backup_path = f"{config_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try: shutil.copyfile(config_path, backup_path); logger.info(f"Created backup: {backup_path}")
        except Exception as e: logger.error(f"Failed to create backup {backup_path}: {e}", exc_info=True); logger.warning("Proceeding without backup.")

    # --- Write Updated YAML ---
    try:
        logger.info(f"Writing updated configuration back to: {config_path}")
        with open(config_path, "w") as f: yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, indent=2)
        logger.info(f"FF YAML file updated successfully. Keys updated: {keys_updated}")
        return True
    except IOError as e: logger.error(f"Error writing updated YAML file {config_path}: {e}", exc_info=True); return False
    except Exception as e: logger.error(f"Unexpected error writing YAML: {e}", exc_info=True); return False

def main():
    """Main execution function."""
    args = parse_args()
    logger.info("Starting FF configuration update process.")
    success = update_ff_config_from_optuna(
        db_path=args.db_path, config_path=args.config_path,
        study_name=args.study_name, create_backup=not args.no_backup,
    )
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()