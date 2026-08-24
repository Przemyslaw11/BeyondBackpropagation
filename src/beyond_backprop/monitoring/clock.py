"""Always-available wall-clock measurement."""

from __future__ import annotations

import time

from ..contracts import ResourceSnapshot


class WallClockResourceMonitor:
    """Measure the canonical training duration without optional dependencies."""

    def __init__(self) -> None:
        self._started_at: float | None = None

    def start(self) -> None:
        self._started_at = time.monotonic()

    def stop(self) -> ResourceSnapshot:
        if self._started_at is None:
            return ResourceSnapshot(measured=False, source="clock-not-started")
        return ResourceSnapshot(
            duration_sec=time.monotonic() - self._started_at,
            measured=True,
            source="monotonic-clock",
        )


__all__ = ["WallClockResourceMonitor"]
