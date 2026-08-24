"""No-op monitor used for CPU smoke tests and disabled measurements."""

from ..contracts import ResourceSnapshot


class NoOpResourceMonitor:
    def start(self) -> None:
        return None

    def stop(self) -> ResourceSnapshot:
        return ResourceSnapshot(measured=False, source="disabled")
