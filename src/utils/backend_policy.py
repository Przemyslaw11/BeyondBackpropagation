"""Compatibility alias for :mod:`beyond_backprop.runtime.backend_policy`.

The module object itself is aliased so legacy tests and callers that patch the
historical ``torch`` symbol continue to affect the canonical implementation.
"""

import sys

from beyond_backprop.runtime import backend_policy as _canonical

sys.modules[__name__] = _canonical
