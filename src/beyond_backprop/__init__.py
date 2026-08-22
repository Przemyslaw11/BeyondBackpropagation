"""Canonical package for the Beyond Backpropagation experiments.

The legacy ``src.*`` namespace remains available while the migration is in
progress. New code should import from this package.
"""

from .contracts import (
    EvaluationResult,
    Evaluator,
    ExperimentTracker,
    MetricProvenance,
    MetricValue,
    ResourceMonitor,
    ResourceSnapshot,
    TrainingAlgorithm,
    TrainingContext,
    TrainingResult,
)

__all__ = [
    "EvaluationResult",
    "Evaluator",
    "ExperimentTracker",
    "MetricProvenance",
    "MetricValue",
    "ResourceMonitor",
    "ResourceSnapshot",
    "TrainingAlgorithm",
    "TrainingContext",
    "TrainingResult",
]
