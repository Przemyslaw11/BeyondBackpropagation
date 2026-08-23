"""Compatibility shim: the canonical MF implementation lives in beyond_backprop."""

from beyond_backprop.algorithms.mf import (  # noqa: F401
    evaluate_mf_local_loss,
    evaluate_mf_model,
    mf_local_loss_fn,
    train_mf_matrix_only,
    train_mf_model,
)

__all__ = [
    "evaluate_mf_local_loss",
    "evaluate_mf_model",
    "mf_local_loss_fn",
    "train_mf_matrix_only",
    "train_mf_model",
]
