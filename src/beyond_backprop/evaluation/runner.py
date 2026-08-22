"""Small evaluation facade shared by runners and tuning objectives."""

from __future__ import annotations

from typing import Any

from ..algorithms.base import AlgorithmAdapter
from ..contracts import EvaluationResult, TrainingContext


def evaluate_algorithm(
    algorithm: AlgorithmAdapter,
    model: Any,
    loader: Any,
    context: TrainingContext,
) -> EvaluationResult:
    """Evaluate only after training has completed and resources are stopped."""

    return algorithm.evaluate(model, loader, context)


__all__ = ["evaluate_algorithm"]
