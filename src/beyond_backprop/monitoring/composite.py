"""Composition of independent resource monitors."""

from __future__ import annotations

from collections.abc import Iterable

from ..contracts import ResourceMonitor, ResourceSnapshot


class CompositeResourceMonitor:
    """Start and stop several monitors over the same training region."""

    def __init__(self, monitors: Iterable[ResourceMonitor]) -> None:
        self.monitors = tuple(monitors)

    def start(self) -> None:
        started: list[ResourceMonitor] = []
        try:
            for monitor in self.monitors:
                monitor.start()
                started.append(monitor)
        except Exception:
            for monitor in reversed(started):
                try:
                    monitor.stop()
                except Exception:
                    pass
            raise

    def stop(self) -> ResourceSnapshot:
        snapshots: list[ResourceSnapshot] = []
        errors: list[Exception] = []
        for monitor in self.monitors:
            try:
                snapshots.append(monitor.stop())
            except Exception as exc:
                errors.append(exc)
        if errors and not snapshots:
            raise errors[0]
        if not snapshots:
            return ResourceSnapshot(source="none", measured=False)

        duration = next(
            (item.duration_sec for item in snapshots if item.duration_sec is not None),
            None,
        )
        energy = next(
            (item.energy_wh for item in snapshots if item.energy_wh is not None), None
        )
        peak_memory = next(
            (
                item.peak_memory_mib
                for item in snapshots
                if item.peak_memory_mib is not None
            ),
            None,
        )
        co2e = next(
            (item.co2e_g for item in snapshots if item.co2e_g is not None), None
        )
        sources = "+".join(item.source for item in snapshots)
        return ResourceSnapshot(
            duration_sec=duration,
            energy_wh=energy,
            peak_memory_mib=peak_memory,
            co2e_g=co2e,
            measured=any(item.measured for item in snapshots),
            source=sources,
        )


__all__ = ["CompositeResourceMonitor"]
