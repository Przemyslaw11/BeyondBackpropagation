"""Shared lifecycle primitives.

The runner is loaded lazily so algorithm adapters can import the standalone
early-stopping primitive without creating an algorithms↔training cycle.
"""

from typing import TYPE_CHECKING, Any

from .early_stopping import EarlyStopping

if TYPE_CHECKING:
    from .runner import ExperimentResult, ExperimentRunner


def __getattr__(name: str) -> Any:
    if name in {"ExperimentResult", "ExperimentRunner", "run_experiment"}:
        from .runner import ExperimentResult, ExperimentRunner, run_experiment

        return {
            "ExperimentResult": ExperimentResult,
            "ExperimentRunner": ExperimentRunner,
            "run_experiment": run_experiment,
        }[name]
    raise AttributeError(name)


__all__ = ["EarlyStopping", "ExperimentResult", "ExperimentRunner", "run_experiment"]
