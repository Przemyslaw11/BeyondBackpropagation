"""Structured tuning result contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrialResult:
    number: int
    seed: int
    value: float | None
    params: dict[str, float | int]
    state: str = "complete"
    error: str | None = None


@dataclass
class StudyResult:
    study_name: str
    algorithm: str
    direction: str
    metric: str
    trials: list[TrialResult] = field(default_factory=list)
    best_trial_number: int | None = None
    best_value: float | None = None
    best_params: dict[str, float | int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_name": self.study_name,
            "algorithm": self.algorithm,
            "direction": self.direction,
            "metric": self.metric,
            "trials": [asdict(trial) for trial in self.trials],
            "best_trial_number": self.best_trial_number,
            "best_value": self.best_value,
            "best_params": self.best_params,
        }


__all__ = ["StudyResult", "TrialResult"]
