"""Updates BP config files with the best hyperparameters from an Optuna study.

BP differs from FF/MF/CaFo here: it targets the required ``optimizer``
section and treats an empty best trial as a failure.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import LOG_FORMAT, parse_args, run_main  # noqa: E402

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Maps Optuna trial parameter names to keys inside the `optimizer` section;
# unmapped parameters are ignored.
KEY_MAP = {
    "lr": "lr",
    "wd": "weight_decay",  # Optuna objective uses 'wd', config 'weight_decay'
    "momentum": "momentum",
}


def main() -> None:
    """Update the BP optimizer settings at ``--config-path`` in place."""
    args = parse_args(
        "Update a Backpropagation (BP) YAML config file with the best "
        "hyperparameters from an Optuna study."
    )
    run_main(
        "BP",
        db_path=args.db_path,
        config_path=args.config_path,
        study_name=args.study_name,
        key_map=KEY_MAP,
        section="optimizer",
        create_missing_section=False,
        fail_on_empty_params=True,
        create_backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
