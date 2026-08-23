"""Compatibility shim: the canonical MF architecture lives in ``beyond_backprop``."""

from beyond_backprop.architectures.mf_mlp import MF_MLP  # noqa: F401

__all__ = ["MF_MLP"]
