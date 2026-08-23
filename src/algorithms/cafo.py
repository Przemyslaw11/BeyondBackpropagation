"""Compatibility shim: the canonical CaFo implementation lives in beyond_backprop."""

from beyond_backprop.algorithms.cafo import (  # noqa: F401
    evaluate_cafo_model,
    train_cafo_model,
)

__all__ = ["evaluate_cafo_model", "train_cafo_model"]
