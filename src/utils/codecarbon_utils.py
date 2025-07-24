"""Utilities for setting up the CodeCarbon emissions tracker."""

import logging
import os
from typing import Any, Dict, Optional

try:
    from codecarbon import OfflineEmissionsTracker
except ImportError:
    OfflineEmissionsTracker = None
    logging.warning("CodeCarbon library not found. Carbon tracking will be disabled.")

from .helpers import create_directory_if_not_exists

logger = logging.getLogger(__name__)


def setup_codecarbon_tracker(
    config: Dict[str, Any], results: Dict[str, Any]
) -> Optional[OfflineEmissionsTracker]:
    """Initializes and starts the CodeCarbon OfflineEmissionsTracker based on config.

    Stores the CSV path in the results dictionary.

    Args:
        config: The main experiment configuration dictionary.
        results: The dictionary to store run results (modified in place).

    Returns:
        The initialized and started tracker object, or None if disabled/failed.
    """
    if OfflineEmissionsTracker is None:
        logger.warning("CodeCarbon library not installed, skipping tracker setup.")
        results["codecarbon_enabled"] = False
        results["codecarbon_csv_path"] = None
        return None

    carbon_tracker_config = config.get("carbon_tracker", {})
    enabled = carbon_tracker_config.get("enabled", False)
    results["codecarbon_enabled"] = enabled

    if not enabled:
        logger.info("CodeCarbon tracking is disabled in config.")
        results["codecarbon_csv_path"] = None
        return None

    output_dir = carbon_tracker_config.get("output_dir", "results/carbon")
    experiment_name = config.get("experiment_name", "default_experiment")
    create_directory_if_not_exists(output_dir)

    csv_filename = f"{experiment_name}_carbon.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    results["codecarbon_csv_path"] = csv_path
    country_iso = carbon_tracker_config.get("country_iso_code", None)
    if not country_iso:
        logger.warning(
            "CodeCarbon country_iso_code not specified, attempting auto-detection."
        )
    results["codecarbon_country_iso"] = country_iso

    mode = carbon_tracker_config.get("mode", "offline").lower()
    results["codecarbon_mode"] = mode

    if mode != "offline":
        logger.warning(
            f"CodeCarbon mode is '{mode}', but only offline is supported. Using offline."
        )

    try:
        logger.info(
            f"Initializing CodeCarbon OfflineEmissionsTracker. Outputting to {csv_path}"
        )
        project_name = (
            config.get("logging", {})
            .get("wandb", {})
            .get("project", "BeyondBackpropagation")
        )
        tracker = OfflineEmissionsTracker(
            output_dir=output_dir,
            output_file=csv_filename,
            country_iso_code=country_iso,
            log_level="INFO",
            save_to_file=True,
            project_name=project_name,
        )
        tracker.start()
        logger.info("CodeCarbon tracker started.")
        return tracker
    except Exception as e:
        log_msg = f"Failed to initialize or start CodeCarbon tracker: {e}"
        logger.error(log_msg, exc_info=True)
        results["codecarbon_csv_path"] = None
        results["codecarbon_enabled"] = False
        return None
