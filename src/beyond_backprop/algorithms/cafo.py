"""Canonical CaFo adapter preserving block and predictor lifecycle stages."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from ..contracts import EvaluationResult, TrainingContext, TrainingResult
from .base import (
    AlgorithmAdapter,
    context_mapping,
    evaluation_result,
    result_from_peak_memory,
)


def _legacy_module() -> Any:
    return importlib.import_module("src.algorithms.cafo")


class CaFoAdapter(AlgorithmAdapter):
    """Preserve frozen blocks, predictor stopping, aggregation, and inference."""

    name = "cafo"

    def fit(self, context: TrainingContext) -> TrainingResult:
        module = _legacy_module()
        config = context_mapping(context)
        algorithm_params = config.get("algorithm_params", {})
        if bool(algorithm_params.get("train_blocks", False)):
            self.lifecycle.append("block_training")
        else:
            self.lifecycle.append("blocks_frozen")
        self.lifecycle.append("predictor_stages")
        peak_memory = module.train_cafo_model(
            model=context.model,
            train_loader=context.train_loader,
            val_loader=context.val_loader,
            config=config,
            device=torch.device(context.device),
            input_adapter=None,
        )
        return result_from_peak_memory(self.name, peak_memory)

    def evaluate(
        self, model: Any, loader: Any, context: TrainingContext
    ) -> EvaluationResult:
        module = _legacy_module()
        params = context_mapping(context).get("algorithm_params", {})
        values = module.evaluate_cafo_model(
            model,
            loader,
            torch.device(context.device),
            criterion=torch.nn.CrossEntropyLoss(),
            aggregation_method=str(params.get("aggregation_method", "sum")),
        )
        self.lifecycle.append("aggregation_inference")
        return evaluation_result(values)


__all__ = ["CaFoAdapter"]
