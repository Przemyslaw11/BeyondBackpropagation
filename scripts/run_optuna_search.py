#!/usr/bin/env python
import argparse
import logging
import optuna
import os
import yaml
from datetime import datetime

# Assume these imports work based on your project structure
from src.utils.config_parser import load_config, CfgNode # Assuming CfgNode or similar for easy access
from src.utils.logging_utils import setup_logging
from src.tuning.optuna_objective import objective # Import the objective function

# Setup basic logger for the script itself
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search for BP baselines.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the baseline configuration YAML file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/optuna",
        help="Directory to save Optuna study results and logs."
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name for the Optuna study. Defaults to config filename + timestamp."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of trials to run. Overrides value in config if provided."
    )
    return parser.parse_args()

def main():
    """Main function to run the Optuna study."""
    args = parse_args()

    # --- Configuration Loading ---
    try:
        # Load the main config file specified by the user
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}", exc_info=True)
        return # Exit if config loading fails

    # --- Output Directory and Study Name ---
    os.makedirs(args.output_dir, exist_ok=True)
    if args.study_name is None:
        config_filename = os.path.splitext(os.path.basename(args.config))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{config_filename}_{timestamp}"
    else:
        study_name = args.study_name
    log_file = os.path.join(args.output_dir, f"{study_name}.log")

    # --- Logging Setup ---
    # Setup logging to file and console based on config/defaults
    log_level_str = config.get('logging', {}).get('level', 'INFO')
    setup_logging(log_level=log_level_str, log_file=log_file)
    logger.info(f"Optuna study name: {study_name}")
    logger.info(f"Saving logs and study database to: {args.output_dir}")

    # --- Optuna Study Setup ---
    n_trials = args.n_trials if args.n_trials is not None else config.get('tuning', {}).get('n_trials', 20)
    storage_path = f"sqlite:///{os.path.join(args.output_dir, f'{study_name}.db')}" # Use SQLite for persistence

    # Configure sampler and pruner (can be made configurable via YAML too)
    sampler_type = config.get('tuning', {}).get('sampler', 'TPE').upper()
    pruner_type = config.get('tuning', {}).get('pruner', 'Median').upper()

    if sampler_type == 'TPE':
        sampler = optuna.samplers.TPESampler(seed=config.get('training', {}).get('seed', 42))
    elif sampler_type == 'RANDOM':
        sampler = optuna.samplers.RandomSampler(seed=config.get('training', {}).get('seed', 42))
    else:
        logger.warning(f"Unsupported sampler type '{sampler_type}', defaulting to TPE.")
        sampler = optuna.samplers.TPESampler(seed=config.get('training', {}).get('seed', 42))

    if pruner_type == 'MEDIAN':
        pruner = optuna.pruners.MedianPruner()
    elif pruner_type == 'HYPERBAND':
        pruner = optuna.pruners.HyperbandPruner()
    elif pruner_type == 'NONE':
         pruner = optuna.pruners.NopPruner()
    else:
        logger.warning(f"Unsupported pruner type '{pruner_type}', defaulting to Median.")
        pruner = optuna.pruners.MedianPruner()

    logger.info(f"Using sampler: {sampler_type}, pruner: {pruner_type}")
    logger.info(f"Database storage: {storage_path}")

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction=config.get('tuning', {}).get('direction', 'maximize'), # 'maximize' for accuracy, 'minimize' for loss
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True # Allows resuming study
        )

        # --- Run Optimization ---
        logger.info(f"Starting Optuna optimization with {n_trials} trials...")
        # Pass the loaded config dictionary to the objective function
        study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)

        # --- Results ---
        logger.info("Optimization finished.")
        logger.info(f"Number of finished trials: {len(study.trials)}")

        best_trial = study.best_trial
        logger.info("Best trial:")
        logger.info(f"  Value (Validation Metric): {best_trial.value:.6f}")
        logger.info("  Params: ")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")

        # --- Save Best Parameters (Optional) ---
        # Option 1: Update the original config file (use with caution)
        # Option 2: Save to a new file
        best_params_file = os.path.join(args.output_dir, f"{study_name}_best_params.yaml")
        best_params_config = {
            'best_validation_metric': best_trial.value,
            'best_hyperparameters': best_trial.params
        }
        try:
            with open(best_params_file, 'w') as f:
                yaml.dump(best_params_config, f, default_flow_style=False)
            logger.info(f"Best parameters saved to: {best_params_file}")
        except Exception as e:
            logger.error(f"Failed to save best parameters: {e}")

    except Exception as e:
        logger.error(f"An error occurred during the Optuna study: {e}", exc_info=True)

if __name__ == "__main__":
    main()
