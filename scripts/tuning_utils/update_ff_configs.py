"""Updates FF config files with the best hyperparameters from an Optuna study."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import LOG_FORMAT, parse_args, run_main  # noqa: E402

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Maps Optuna trial parameter names to keys inside the `algorithm_params`
# section; unmapped parameters are ignored.
KEY_MAP = {
    "ff_lr": "ff_learning_rate",
    "ff_wd": "ff_weight_decay",
    "ds_lr": "downstream_learning_rate",
    "ds_wd": "downstream_weight_decay",
}


def main() -> None:
    """Update the FF config at ``--config-path`` in place."""
    args = parse_args(
        "Update a Forward-Forward (FF) YAML config file with the best "
        "hyperparameters from an Optuna study."
    )
    run_main(
        "FF",
        db_path=args.db_path,
        config_path=args.config_path,
        study_name=args.study_name,
        key_map=KEY_MAP,
        create_backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
