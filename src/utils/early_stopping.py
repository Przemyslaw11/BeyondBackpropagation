"""Single early-stopping policy shared by every algorithm.

Fairness requires that BP, FF, CaFo and MF stop on the same metric with the same
patience, min_delta and epoch cap. All four read this module instead of their own
per-algorithm config keys.
"""

from typing import Any, Dict

DEFAULT_POLICY: Dict[str, Any] = {
    "enabled": True,
    "metric": "val_loss",
    "mode": "min",
    "min_delta": 0.0,
    "patience": 20,
    "max_epochs": 100,
}


def resolve_early_stopping(config: Dict[str, Any]) -> Dict[str, Any]:
    """Returns the shared early-stopping policy declared under ``early_stopping:``."""
    policy = dict(DEFAULT_POLICY)
    declared = config.get("early_stopping", {})
    if not isinstance(declared, dict):
        raise ValueError("'early_stopping' must be a mapping.")

    unknown = set(declared) - set(DEFAULT_POLICY)
    if unknown:
        raise ValueError(
            f"Unknown early_stopping keys {sorted(unknown)}; the policy is shared "
            f"across algorithms and only accepts {sorted(DEFAULT_POLICY)}."
        )
    policy.update(declared)

    policy["enabled"] = bool(policy["enabled"])
    policy["metric"] = str(policy["metric"]).lower()
    policy["mode"] = str(policy["mode"]).lower()
    policy["min_delta"] = float(policy["min_delta"])
    policy["patience"] = int(policy["patience"])
    policy["max_epochs"] = int(policy["max_epochs"])

    if policy["mode"] not in ("min", "max"):
        raise ValueError(f"early_stopping.mode must be 'min' or 'max', got {policy['mode']}.")
    if policy["patience"] < 1:
        raise ValueError("early_stopping.patience must be >= 1.")
    if policy["max_epochs"] < 1:
        raise ValueError("early_stopping.max_epochs must be >= 1.")
    return policy


def resolve_tuning_max_epochs(config: Dict[str, Any]) -> int:
    """Returns the reduced HPO epoch budget, shared by every algorithm's objective."""
    tuning_cfg = config.get("tuning", {})
    max_epochs = tuning_cfg.get("max_epochs")
    if max_epochs is None:
        max_epochs = resolve_early_stopping(config)["max_epochs"]
    return int(max_epochs)
