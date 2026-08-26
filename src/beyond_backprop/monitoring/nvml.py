"""Optional NVML monitoring preserving the legacy energy sampler semantics."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any

from ..contracts import ResourceSnapshot

logger = logging.getLogger(__name__)


class NvmlResourceMonitor:
    """Adapt the existing NVML sampler behind a lazy optional-import boundary."""

    def __init__(self, device_index: int = 0, interval_sec: float = 0.2) -> None:
        if interval_sec <= 0:
            raise ValueError("Sampling interval must be positive")
        self.device_index = device_index
        self.interval_sec = interval_sec
        self._started_at: float | None = None
        self._nvml: Any | None = None
        self._handle: Any | None = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, float, float]] = []

    def start(self) -> None:
        self._started_at = time.monotonic()
        try:
            import pynvml

            pynvml.nvmlInit()
            count = int(pynvml.nvmlDeviceGetCount())
            if self.device_index < 0 or self.device_index >= count:
                raise RuntimeError(
                    f"NVML device index {self.device_index} is unavailable"
                )
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._samples = []
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        except Exception as exc:
            logger.warning(
                "NVML monitoring unavailable (%s: %s); resource snapshot will "
                "report measured=False with source 'nvml-unavailable'.",
                type(exc).__name__,
                exc,
            )
            self._nvml = None
            self._handle = None

    def _sample_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self.interval_sec)
        self._sample()

    def _sample(self) -> None:
        if self._nvml is None or self._handle is None:
            return
        try:
            now = time.monotonic()
            power_w = float(self._nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
            memory_mib = float(
                self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used
            ) / (1024.0 * 1024.0)
            self._samples.append((now, power_w, memory_mib))
        except (AttributeError, OSError, RuntimeError):
            return

    def stop(self) -> ResourceSnapshot:
        if self._started_at is None:
            return ResourceSnapshot(measured=False, source="nvml-not-started")

        duration = time.monotonic() - self._started_at
        if self._nvml is None or self._handle is None:
            return ResourceSnapshot(
                duration_sec=duration,
                measured=False,
                source="nvml-unavailable",
            )

        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 2.0))
        samples = sorted(self._samples)
        with contextlib.suppress(AttributeError, OSError, RuntimeError):
            self._nvml.nvmlShutdown()
        energy_joules = 0.0
        for first, second in zip(samples, samples[1:], strict=False):
            energy_joules += (second[0] - first[0]) * (first[1] + second[1]) / 2.0
        if not samples:
            return ResourceSnapshot(
                duration_sec=duration,
                measured=False,
                source="nvml-no-samples",
            )
        return ResourceSnapshot(
            duration_sec=duration,
            energy_wh=energy_joules / 3600.0,
            peak_memory_mib=max(sample[2] for sample in samples),
            measured=True,
            source="nvml",
        )


__all__ = ["NvmlResourceMonitor"]
