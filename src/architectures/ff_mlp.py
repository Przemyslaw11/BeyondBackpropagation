"""Compatibility shim: the canonical FF architecture lives in ``beyond_backprop``."""

from beyond_backprop.architectures.ff_mlp import (  # noqa: F401
    FF_MLP,
    ReLU_full_grad,
)

__all__ = ["FF_MLP", "ReLU_full_grad"]
