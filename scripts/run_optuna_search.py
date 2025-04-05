# File: scripts/run_optuna_search.py
#!/usr/bin/env python
import argparse
import logging
import optuna
import os
import yaml
from datetime import datetime
import sys
import pprint

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config_parser import load_config
from src.utils.logging_utils import setup_logging, logger  # Use central logger
from src.tuning.optuna_objective import objective
from src.utils.helpers import create_directory_if_not_exists


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for BP baselines."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the baseline configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/optuna",
        help="Directory to save Optuna study results.",
    )
    parser.add_argument(
        "--study-name", type=str, default=None, help="Name for the Optuna study."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of trials to run (overrides config).",
    )
    return parser.parse_args()


def main():
    """Main function to set up and run the Optuna study."""
    args = parse_args()

    # --- Configuration Loading ---
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
        logger.info("Initial Config for Optuna:")
        # Log the config used for the study setup
        config_str = pprint.pformat(config)
        for line in config_str.split("\n"):
            logger.info(line)
    except Exception as e:
        logging.error(
            f"Error loading configuration: {e}", exc_info=True
        )  # Use basic logging if logger setup failed
        return

    # --- Output Directory and Study Name ---
    create_directory_if_not_exists(args.output_dir)  # Use helper
    if args.study_name is None:
        config_filename = os.path.splitext(os.path.basename(args.config))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{config_filename}_optuna_{timestamp}"
    else:
        study_name = args.study_name
    log_file = os.path.join(args.output_dir, f"{study_name}.log")

    # --- Logging Setup ---
    log_level_str = config.get("logging", {}).get("level", "INFO")
    # Ensure logging goes to the specific file for this study
    setup_logging(log_level=log_level_str, log_file=log_file)
    logger.info(f"Optuna study name: {study_name}")
    logger.info(f"Saving logs and study database to: {args.output_dir}")

    # --- Optuna Study Setup ---
    tuning_config = config.get("tuning", {})
    if not tuning_config:
        logger.error("Config file must contain a 'tuning' section.")
        return

    n_trials = (
        args.n_trials
        if args.n_trials is not None
        else tuning_config.get("n_trials", 20)
    )
    storage_path = f"sqlite:///{os.path.join(args.output_dir, f'{study_name}.db')}"

    sampler_type = tuning_config.get("sampler", "TPE").upper()
    pruner_type = tuning_config.get("pruner", "Median").upper()
    optuna_seed = config.get("general", {}).get("seed", 42)

    sampler_map = {
        "TPE": optuna.samplers.TPESampler(seed=optuna_seed),
        "RANDOM": optuna.samplers.RandomSampler(seed=optuna_seed),
    }
    pruner_map = {
        "MEDIAN": optuna.pruners.MedianPruner(),
        "HYPERBAND": optuna.pruners.HyperbandPruner(),
        "NONE": optuna.pruners.NopPruner(),
    }

    sampler = sampler_map.get(
        sampler_type, optuna.samplers.TPESampler(seed=optuna_seed)
    )
    pruner = pruner_map.get(pruner_type, optuna.pruners.MedianPruner())
    if sampler_type not in sampler_map:
        logger.warning(f"Unsupported sampler '{sampler_type}', using TPE.")
    if pruner_type not in pruner_map:
        logger.warning(f"Unsupported pruner '{pruner_type}', using Median.")

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
            load_if_exists=True,
        )

        # --- Run Optimization ---
        logger.info(f"Starting Optuna optimization with {n_trials} trials...")
        # Pass the base_config to the objective function
        study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)

        # --- Results ---
        logger.info("Optimization finished.")
        logger.info(f"Number of finished trials: {len(study.trials)}")
        best_trial = study.best_trial
        logger.info("=" * 30)
        logger.info("           Best Trial           ")
        logger.info("=" * 30)
        logger.info(f"  Trial Number: {best_trial.number}")
        metric_name = tuning_config.get("metric", "validation_metric")
        logger.info(f"  Value ({metric_name}): {best_trial.value:.6f}")
        logger.info("  Params (Use these to update baseline config): ")
        # Prepare params in a format suitable for the optimizer section
        best_config_update = {"optimizer": {}}
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")
            # Map Optuna trial param names (e.g., 'wd') back to config keys ('weight_decay') if needed
            if key == "wd":
                best_config_update["optimizer"]["weight_decay"] = value
            else:
                best_config_update["optimizer"][key] = value
        # Add the optimizer type back (it wasn't tuned)
        best_config_update["optimizer"]["type"] = config.get("optimizer", {}).get(
            "type", "AdamW"
        )
        logger.info("=" * 30)
        logger.info("YAML snippet to update baseline config's 'optimizer' section:")
        print(
            "\n"
            + yaml.dump(best_config_update, default_flow_style=False, sort_keys=False)
            + "\n"
        )

        # --- Save Best Parameters ---
        best_params_file = os.path.join(
            args.output_dir, f"{study_name}_best_params.yaml"
        )
        study_results_summary = {
            "study_name": study_name,
            "best_trial_number": best_trial.number,
            f"best_{metric_name}": best_trial.value,
            "best_hyperparameters_raw": best_trial.params,  # Original trial params
            "config_override": best_config_update,  # Formatted for config update
        }

        try:
            with open(best_params_file, "w") as f:
                yaml.dump(
                    study_results_summary, f, default_flow_style=False, sort_keys=False
                )
            logger.info(
                f"Best parameters and study summary saved to: {best_params_file}"
            )
            # --- IMPORTANT REMINDER ---
            logger.warning("=" * 50)
            logger.warning("REMINDER: Manually update the 'optimizer' section in your")
            logger.warning(f"baseline config file ('{args.config}') with the 'lr' and ")
            logger.warning("'weight_decay' values from 'config_override' above before")
            logger.warning("running the final baseline experiment!")
            logger.warning("=" * 50)
        except Exception as e:
            logger.error(f"Failed to save best parameters: {e}")

    except ImportError:
        logger.error("optuna library not found. Install with `pip install optuna`")
    except Exception as e:
        logger.error(f"An error occurred during the Optuna study: {e}", exc_info=True)


if __name__ == "__main__":
    # Basic logging if config loading fails very early
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
