"""Compatibility shim: the canonical CaFo CNN lives in ``beyond_backprop``."""

from beyond_backprop.architectures.cafo_cnn import (  # noqa: F401
    CaFoBlock,
    CaFoPredictor,
    CaFo_CNN,
)

__all__ = ["CaFo_CNN", "CaFoBlock", "CaFoPredictor"]
