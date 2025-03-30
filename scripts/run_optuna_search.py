# File: scripts/run_optuna_search.py
#!/usr/bin/env python
import argparse
import logging
import optuna
import os
import yaml
from datetime import datetime
import sys  # For path manipulation

# Add project root to path to allow imports like src.utils
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config_parser import load_config  # Use the correct function name
from src.utils.logging_utils import setup_logging  # Use centralized logging setup
from src.tuning.optuna_objective import objective  # Import the objective function

# Setup basic logger for this script
logger = logging.getLogger(__name__)  # Get logger instance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for BP baselines."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the baseline configuration YAML file (should contain a 'tuning' section).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/optuna",
        help="Directory to save Optuna study results (DB) and logs.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name for the Optuna study. Defaults to config filename + timestamp.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of trials to run. Overrides value in config's 'tuning' section if provided.",
    )
    return parser.parse_args()


def main():
    """Main function to set up and run the Optuna study."""
    args = parse_args()

    # --- Configuration Loading ---
    try:
        # Load the main config file specified by the user
        # This config should contain the 'tuning' section and parameters for the baseline
        config = load_config(args.config)  # Use correct function name
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}", exc_info=True)
        return

    # --- Output Directory and Study Name ---
    os.makedirs(args.output_dir, exist_ok=True)
    if args.study_name is None:
        config_filename = os.path.splitext(os.path.basename(args.config))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{config_filename}_optuna_{timestamp}"
    else:
        study_name = args.study_name
    log_file = os.path.join(args.output_dir, f"{study_name}.log")

    # --- Logging Setup ---
    # Setup logging based on config or defaults, routing to file and console
    log_level_str = config.get("logging", {}).get("level", "INFO")
    setup_logging(
        log_level=log_level_str, log_file=log_file
    )  # Use central setup function
    logger.info(f"Optuna study name: {study_name}")
    logger.info(f"Saving logs and study database to: {args.output_dir}")

    # --- Optuna Study Setup ---
    tuning_config = config.get("tuning", {})
    if not tuning_config:
        logger.error("Config file must contain a 'tuning' section for Optuna search.")
        return

    n_trials = (
        args.n_trials
        if args.n_trials is not None
        else tuning_config.get("n_trials", 20)
    )
    storage_path = f"sqlite:///{os.path.join(args.output_dir, f'{study_name}.db')}"  # Use SQLite for persistence

    # Configure sampler and pruner
    sampler_type = tuning_config.get("sampler", "TPE").upper()
    pruner_type = tuning_config.get("pruner", "Median").upper()
    optuna_seed = config.get("general", {}).get(
        "seed", 42
    )  # Use general seed for sampler

    sampler_map = {
        "TPE": optuna.samplers.TPESampler(seed=optuna_seed),
        "RANDOM": optuna.samplers.RandomSampler(seed=optuna_seed),
        # Add other samplers if desired
    }
    pruner_map = {
        "MEDIAN": optuna.pruners.MedianPruner(),
        "HYPERBAND": optuna.pruners.HyperbandPruner(),
        "NONE": optuna.pruners.NopPruner(),
        # Add other pruners if desired
    }

    sampler = sampler_map.get(sampler_type)
    if sampler is None:
        logger.warning(f"Unsupported sampler type '{sampler_type}', defaulting to TPE.")
        sampler = optuna.samplers.TPESampler(seed=optuna_seed)

    pruner = pruner_map.get(pruner_type)
    if pruner is None:
        logger.warning(
            f"Unsupported pruner type '{pruner_type}', defaulting to Median."
        )
        pruner = optuna.pruners.MedianPruner()

    study_direction = tuning_config.get("direction", "maximize")
    logger.info(f"Using sampler: {sampler_type}, pruner: {pruner_type}")
    logger.info(f"Optimization direction: {study_direction}")
    logger.info(f"Database storage: {storage_path}")

    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction=study_direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,  # Allows resuming study
        )

        # --- Run Optimization ---
        logger.info(f"Starting Optuna optimization with {n_trials} trials...")
        # Pass the loaded config dictionary (contains baseline setup and tuning ranges)
        study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)

        # --- Results ---
        logger.info("Optimization finished.")
        logger.info(f"Number of finished trials: {len(study.trials)}")

        # Log best trial results
        best_trial = study.best_trial
        logger.info("=" * 30)
        logger.info("           Best Trial           ")
        logger.info("=" * 30)
        logger.info(f"  Trial Number: {best_trial.number}")
        metric_name = tuning_config.get("metric", "validation_metric")
        logger.info(f"  Value ({metric_name}): {best_trial.value:.6f}")
        logger.info("  Params: ")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")
        logger.info("=" * 30)

        # --- Save Best Parameters ---
        best_params_file = os.path.join(
            args.output_dir, f"{study_name}_best_params.yaml"
        )
        # Save parameters in a format that can be easily used to update a config
        best_config_update = {
            "optimizer": best_trial.params  # Assumes keys match optimizer config structure
        }
        # Add information about the result
        study_results_summary = {
            "study_name": study_name,
            "best_trial_number": best_trial.number,
            f"best_{metric_name}": best_trial.value,
            "best_hyperparameters": best_trial.params,
            "config_override": best_config_update,  # Parameters to use in final run
        }

        try:
            with open(best_params_file, "w") as f:
                yaml.dump(
                    study_results_summary, f, default_flow_style=False, sort_keys=False
                )
            logger.info(
                f"Best parameters and study summary saved to: {best_params_file}"
            )
        except Exception as e:
            logger.error(f"Failed to save best parameters: {e}")

    except ImportError:
        logger.error(
            "optuna library not found. Please install it (`pip install optuna`)"
        )
    except Exception as e:
        logger.error(f"An error occurred during the Optuna study: {e}", exc_info=True)


if __name__ == "__main__":
    # Basic logging setup in case configuration loading fails early
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
