"""Canonical Mono-Forward adapter preserving its layer-wise lifecycle."""

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
    return importlib.import_module("src.algorithms.mf")


class MFAdapter(AlgorithmAdapter):
    """Preserve M0/layer isolation, detached activations, and MF inference."""

    name = "mf"

    def fit(self, context: TrainingContext) -> TrainingResult:
        module = _legacy_module()
        model = context.model
        if hasattr(model, "num_hidden_layers"):
            self.lifecycle.append("M0")
            self.lifecycle.extend(
                f"W{i}_M{i}" for i in range(1, int(model.num_hidden_layers) + 1)
            )
        peak_memory = module.train_mf_model(
            model=model,
            train_loader=context.train_loader,
            config=context_mapping(context),
            device=torch.device(context.device),
            input_adapter=flatten_if_needed(context),
            val_loader=context.val_loader,
        )
        return result_from_peak_memory(self.name, peak_memory)

    def evaluate(
        self, model: Any, loader: Any, context: TrainingContext
    ) -> EvaluationResult:
        module = _legacy_module()
        values = module.evaluate_mf_model(
            model,
            loader,
            torch.device(context.device),
            flatten_if_needed(context),
        )
        self.lifecycle.append("last_activation_projection_inference")
        return evaluation_result(values)


__all__ = ["MFAdapter"]
