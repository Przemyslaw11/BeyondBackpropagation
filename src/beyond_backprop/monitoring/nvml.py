"""Optional NVML monitoring preserving the legacy energy sampler semantics."""

from __future__ import annotations

import time
from typing import Any

from ..contracts import ResourceSnapshot


class NvmlResourceMonitor:
    """Adapt the existing NVML sampler behind a lazy optional-import boundary."""

    def __init__(self, device_index: int = 0, interval_sec: float = 0.2) -> None:
        if interval_sec <= 0:
            raise ValueError("Sampling interval must be positive")
        self.device_index = device_index
        self.interval_sec = interval_sec
        self._legacy_monitor: Any | None = None
        self._started_at: float | None = None

    def start(self) -> None:
        self._started_at = time.monotonic()
        try:
            from src.utils.monitoring import GPUEnergyMonitor

            self._legacy_monitor = GPUEnergyMonitor(
                device_index=self.device_index, interval_sec=self.interval_sec
            )
            self._legacy_monitor.start()
        except (ImportError, OSError, RuntimeError):
            self._legacy_monitor = None

    def stop(self) -> ResourceSnapshot:
        if self._started_at is None:
            return ResourceSnapshot(measured=False, source="nvml-not-started")

        duration = time.monotonic() - self._started_at
        if self._legacy_monitor is None:
            return ResourceSnapshot(
                duration_sec=duration,
                measured=False,
                source="nvml-unavailable",
            )

        energy_joules = self._legacy_monitor.stop()
        return ResourceSnapshot(
            duration_sec=duration,
            energy_wh=(float(energy_joules) / 3600.0)
            if energy_joules is not None
            else None,
            measured=energy_joules is not None,
            source="nvml",
        )


__all__ = ["NvmlResourceMonitor"]
