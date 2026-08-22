"""Compatibility alias for the canonical data loader implementation."""

import sys

from beyond_backprop.data import loaders as _canonical

sys.modules[__name__] = _canonical
