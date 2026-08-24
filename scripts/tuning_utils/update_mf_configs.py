"""Updates MF config files with the best hyperparameters from an Optuna study."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import LOG_FORMAT, parse_args, run_main  # noqa: E402

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Maps Optuna trial parameter names to keys inside the `algorithm_params`
# section; unmapped parameters are ignored.
KEY_MAP = {
    "lr": "lr",
    "epochs_per_layer": "epochs_per_layer",
    "wd": "weight_decay",
}


def main() -> None:
    """Update the MF config at ``--config-path`` in place."""
    args = parse_args(
        "Update a Mono-Forward (MF) YAML config file with the best "
        "hyperparameters from an Optuna study."
    )
    run_main(
        "MF",
        db_path=args.db_path,
        config_path=args.config_path,
        study_name=args.study_name,
        key_map=KEY_MAP,
        create_backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
