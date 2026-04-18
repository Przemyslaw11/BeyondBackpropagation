"""Guards the protocol invariants that make cross-algorithm comparisons fair.

Every experiment config in a configuration group must stop on the same metric with
the same patience, min_delta and epoch cap, use the same pruner policy, and see the
same data. Anything else turns an energy comparison into a budget comparison.
"""

import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from src.utils.config_parser import load_config
from src.utils.early_stopping import resolve_early_stopping

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
# The ladder rungs added in Phase 3 have no pre-Phase-2 protocol to preserve.
LEGACY_EXPERIMENT_DIRS = ("bp_baselines", "cafo", "ff", "mf")
EXPERIMENT_DIRS = LEGACY_EXPERIMENT_DIRS + ("bp_ds", "mf_joint")

TUNED_HYPERPARAMETERS = (
    ("optimizer", "lr"),
    ("optimizer", "weight_decay"),
    ("algorithm_params", "predictor_lr"),
    ("algorithm_params", "predictor_weight_decay"),
    ("algorithm_params", "block_lr"),
    ("algorithm_params", "block_weight_decay"),
    ("algorithm_params", "ff_learning_rate"),
    ("algorithm_params", "ff_weight_decay"),
    ("algorithm_params", "downstream_learning_rate"),
    ("algorithm_params", "downstream_weight_decay"),
    ("algorithm_params", "mf_lr"),
    ("algorithm_params", "mf_weight_decay"),
    ("algorithm_params", "aux_weight"),
)

# Config pairs known to carry an identical tuned value that was copied rather than
# searched. Phase 3 re-tunes them; until then the collision is acknowledged, not hidden.
SHARED_HYPERPARAMETER_OPT_OUT = {
    # Ladder rungs 4, 5 and 6 differ only by the activation cache. They MUST share
    # rung 4's hyperparameters, otherwise the comparison measures the search too.
    *(
        (section, name, owner, other)
        for section, name in (
            ("algorithm_params", "lr"),
            ("algorithm_params", "weight_decay"),
        )
        for owner, other in (
            ("mnist_mlp_2x1000.yaml", "mnist_mlp_2x1000_cache_device.yaml"),
            ("mnist_mlp_2x1000.yaml", "mnist_mlp_2x1000_cache_host.yaml"),
            (
                "mnist_mlp_2x1000_cache_device.yaml",
                "mnist_mlp_2x1000_cache_host.yaml",
            ),
            (
                "fashion_mnist_mlp_2x1000.yaml",
                "fashion_mnist_mlp_2x1000_cache_device.yaml",
            ),
            (
                "fashion_mnist_mlp_2x1000.yaml",
                "fashion_mnist_mlp_2x1000_cache_host.yaml",
            ),
            (
                "fashion_mnist_mlp_2x1000_cache_device.yaml",
                "fashion_mnist_mlp_2x1000_cache_host.yaml",
            ),
        )
    ),
    ("optimizer", "lr", "mnist_mlp_3x1000_bp.yaml", "mnist_mlp_4x2000_bp.yaml"),
    (
        "optimizer",
        "weight_decay",
        "mnist_mlp_3x1000_bp.yaml",
        "mnist_mlp_4x2000_bp.yaml",
    ),
    (
        "algorithm_params",
        "predictor_lr",
        "cafodfa_cifar100_cnn_3block.yaml",
        "cafodfa_cifar10_cnn_3block.yaml",
    ),
    (
        "algorithm_params",
        "predictor_weight_decay",
        "cafodfa_cifar100_cnn_3block.yaml",
        "cafodfa_cifar10_cnn_3block.yaml",
    ),
    (
        "algorithm_params",
        "block_lr",
        "cafodfa_cifar100_cnn_3block.yaml",
        "cafodfa_cifar10_cnn_3block.yaml",
    ),
    (
        "algorithm_params",
        "block_weight_decay",
        "cafodfa_cifar100_cnn_3block.yaml",
        "cafodfa_cifar10_cnn_3block.yaml",
    ),
    (
        "algorithm_params",
        "predictor_lr",
        "cifar100_cnn_3block.yaml",
        "cifar10_cnn_3block.yaml",
    ),
    (
        "algorithm_params",
        "predictor_weight_decay",
        "cifar100_cnn_3block.yaml",
        "cifar10_cnn_3block.yaml",
    ),
}


def _experiment_config_paths() -> List[Path]:
    paths: List[Path] = []
    for directory in EXPERIMENT_DIRS:
        paths.extend(sorted((REPO_ROOT / "configs" / directory).glob("*.yaml")))
    return paths


def _tuning_config_paths() -> List[Path]:
    return sorted((REPO_ROOT / "configs" / "tuning").glob("*.yaml"))


def _load(path: Path) -> Dict[str, Any]:
    return load_config(str(path), base_config_path=str(BASE_CONFIG))


def _load_raw(path: Path) -> Dict[str, Any]:
    """Reads a config without the base merge, so inherited defaults are not mistaken
    for values a config chose for itself."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _group_key(config: Dict[str, Any]) -> Tuple[str, str]:
    """Groups configs that must be directly comparable in a results table."""
    model_params = config.get("model", {}).get("params", {})
    shape = model_params.get("hidden_dims") or model_params.get("block_channels") or []
    return config.get("data", {}).get("name", "?").lower(), str(list(shape))


class FairnessInvariantTests(unittest.TestCase):
    """Asserts that protocol settings are uniform within each configuration group."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment_configs = {
            path: _load(path) for path in _experiment_config_paths()
        }
        cls.raw_configs = {
            path: _load_raw(path) for path in _experiment_config_paths()
        }
        cls.tuning_configs = {path: _load(path) for path in _tuning_config_paths()}
        cls.groups: Dict[Tuple[str, str], List[Path]] = {}
        for path, config in cls.experiment_configs.items():
            cls.groups.setdefault(_group_key(config), []).append(path)

    def test_configs_exist(self) -> None:
        self.assertTrue(self.experiment_configs)
        self.assertTrue(self.tuning_configs)

    def test_early_stopping_identical_within_group(self) -> None:
        for key, paths in self.groups.items():
            policies = {
                path.name: resolve_early_stopping(self.experiment_configs[path])
                for path in paths
            }
            distinct = {
                tuple(sorted(policy.items())) for policy in policies.values()
            }
            self.assertEqual(
                len(distinct),
                1,
                f"Group {key} has diverging early-stopping policies: {policies}",
            )

    def test_no_per_algorithm_early_stopping_keys_remain(self) -> None:
        forbidden = (
            "early_stopping_metric",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "early_stopping_mode",
            "mf_early_stopping_enabled",
            "predictor_early_stopping_enabled",
        )
        for path, config in {**self.experiment_configs, **self.tuning_configs}.items():
            for section in ("training", "algorithm_params"):
                keys = config.get(section, {}) or {}
                for name in forbidden:
                    self.assertNotIn(
                        name,
                        keys,
                        f"{path.name} still sets {section}.{name}; use the shared "
                        "early_stopping block instead.",
                    )

    def test_pruner_policy_is_uniform(self) -> None:
        pruners = {
            path.name: str(config.get("tuning", {}).get("pruner", "None")).lower()
            for path, config in {
                **self.experiment_configs,
                **self.tuning_configs,
            }.items()
        }
        self.assertEqual(
            set(pruners.values()),
            {"none"},
            f"Pruner policy is not uniform across algorithms: {pruners}",
        )

    def test_tuning_epoch_budget_is_shared(self) -> None:
        budgets = {
            path.name: config.get("tuning", {}).get("max_epochs")
            for path, config in self.tuning_configs.items()
        }
        self.assertEqual(
            len(set(budgets.values())),
            1,
            f"HPO epoch budget is not uniform across algorithms: {budgets}",
        )
        for path, config in self.tuning_configs.items():
            self.assertNotIn(
                "num_epochs",
                config.get("tuning", {}),
                f"{path.name} still overrides tuning.num_epochs.",
            )

    def test_data_pipeline_identical_within_group(self) -> None:
        for key, paths in self.groups.items():
            batch_sizes = {
                path.name: self.experiment_configs[path]
                .get("data_loader", {})
                .get("batch_size")
                for path in paths
            }
            val_splits = {
                path.name: self.experiment_configs[path]
                .get("data", {})
                .get("val_split")
                for path in paths
            }
            self.assertEqual(
                len(set(batch_sizes.values())),
                1,
                f"Group {key} has diverging batch sizes: {batch_sizes}",
            )
            self.assertEqual(
                len(set(val_splits.values())),
                1,
                f"Group {key} has diverging validation splits: {val_splits}",
            )

    def test_tuned_hyperparameters_are_not_silently_shared(self) -> None:
        seen: Dict[Tuple[str, str, Any], str] = {}
        collisions: List[str] = []
        for path in sorted(self.raw_configs):
            config = self.raw_configs[path]
            for section, name in TUNED_HYPERPARAMETERS:
                value = (config.get(section) or {}).get(name)
                if value is None:
                    continue
                marker = (section, name, value)
                owner = seen.setdefault(marker, path.name)
                if owner == path.name:
                    continue
                if (section, name, owner, path.name) in SHARED_HYPERPARAMETER_OPT_OUT:
                    continue
                collisions.append(
                    f"{section}.{name}={value} shared by {owner} and {path.name}"
                )
        self.assertEqual(
            collisions,
            [],
            "Tuned values duplicated across configs without an opt-out entry: "
            + "; ".join(collisions),
        )

    def test_legacy_hyperparameters_are_preserved(self) -> None:
        for path, config in self.experiment_configs.items():
            if path.parent.name not in LEGACY_EXPERIMENT_DIRS:
                continue
            self.assertIn(
                "legacy_hyperparameters",
                config,
                f"{path.name} dropped its pre-Phase-2 settings; Phase 6 needs them.",
            )


if __name__ == "__main__":
    unittest.main()
