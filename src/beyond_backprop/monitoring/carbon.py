"""Optional CodeCarbon resource monitor."""

from __future__ import annotations

import time
from typing import Any

from ..contracts import ResourceSnapshot


class CodeCarbonResourceMonitor:
    """Measure emissions only when the optional CodeCarbon package is installed."""

    def __init__(self, **config: Any) -> None:
        self.config = config
        self._tracker: Any | None = None
        self._started_at: float | None = None

    def start(self) -> None:
        self._started_at = time.monotonic()
        try:
            from codecarbon import OfflineEmissionsTracker

            self._tracker = OfflineEmissionsTracker(**self.config)
            self._tracker.start()
        except (ImportError, OSError, RuntimeError):
            self._tracker = None

    def stop(self) -> ResourceSnapshot:
        if self._started_at is None:
            return ResourceSnapshot(measured=False, source="codecarbon-not-started")
        duration = time.monotonic() - self._started_at
        if self._tracker is None:
            return ResourceSnapshot(
                duration_sec=duration,
                measured=False,
                source="codecarbon-unavailable",
            )
        emissions = self._tracker.stop()
        return ResourceSnapshot(
            duration_sec=duration,
            co2e_g=float(emissions) if emissions is not None else None,
            measured=emissions is not None,
            source="codecarbon",
        )


__all__ = ["CodeCarbonResourceMonitor"]
