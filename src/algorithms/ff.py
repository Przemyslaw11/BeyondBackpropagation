"""Compatibility shim: the canonical FF implementation lives in ``beyond_backprop``."""

from beyond_backprop.algorithms.ff import (  # noqa: F401
    evaluate_ff_model,
    generate_ff_hinton_inputs,
    get_linear_cooldown_lr,
    train_ff_model,
)

__all__ = [
    "evaluate_ff_model",
    "generate_ff_hinton_inputs",
    "get_linear_cooldown_lr",
    "train_ff_model",
]
