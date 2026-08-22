"""Canonical Forward-Forward algorithm adapter."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from ..contracts import EvaluationResult, TrainingContext, TrainingResult
from .base import (
    AlgorithmAdapter,
    context_mapping,
    evaluation_result,
    flatten_if_needed,
    result_from_peak_memory,
)


def _legacy_module() -> Any:
    return importlib.import_module("src.algorithms.ff")


class FFAdapter(AlgorithmAdapter):
    """Preserve local goodness updates, downstream training, and FF inference."""

    name = "ff"

    def fit(self, context: TrainingContext) -> TrainingResult:
        module = _legacy_module()
        self.lifecycle.extend(("local_goodness_updates", "downstream_classifier"))
        peak_memory = module.train_ff_model(
            model=context.model,
            train_loader=context.train_loader,
            val_loader=context.val_loader,
            config=context_mapping(context),
            device=torch.device(context.device),
            wandb_run=None,
            input_adapter=flatten_if_needed(context),
            step_ref=[-1],
            gpu_handle=None,
            nvml_active=False,
        )
        return result_from_peak_memory(self.name, peak_memory)

    def evaluate(
        self, model: Any, loader: Any, context: TrainingContext
    ) -> EvaluationResult:
        module = _legacy_module()
        values = module.evaluate_ff_model(model, loader, torch.device(context.device))
        self.lifecycle.append("algorithm_specific_inference")
        return evaluation_result(values)


__all__ = ["FFAdapter"]
