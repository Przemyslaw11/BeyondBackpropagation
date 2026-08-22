"""Canonical BP adapter over the repository's established BP routines."""

from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import nn

from ..contracts import EvaluationResult, TrainingContext, TrainingResult
from .base import (
    AlgorithmAdapter,
    context_mapping,
    evaluation_result,
    flatten_if_needed,
    result_from_peak_memory,
)


def _legacy_module() -> Any:
    """Load legacy numerical code only when the BP lifecycle is invoked."""
    return importlib.import_module("src.baselines.bp")


class BPAdapter(AlgorithmAdapter):
    """Adapter preserving global cross-entropy and AdamW BP semantics."""

    name = "bp"

    def fit(self, context: TrainingContext) -> TrainingResult:
        config = context_mapping(context)
        optimizer_config = dict(config.get("optimizer", {}))
        optimizer_config.setdefault("type", "AdamW")
        config["optimizer"] = optimizer_config
        training_config = dict(config.get("training", {}))
        training_config.setdefault("criterion", "CrossEntropyLoss")
        config["training"] = training_config
        module = _legacy_module()
        peak_memory = module.train_bp_model(
            model=context.model,
            train_loader=context.train_loader,
            val_loader=context.val_loader,
            config=config,
            device=torch.device(context.device),
            input_adapter=flatten_if_needed(context),
        )
        self.lifecycle.append("global_cross_entropy")
        self.lifecycle.append(
            "adamw"
            if optimizer_config["type"].lower() == "adamw"
            else "configured_optimizer"
        )
        return result_from_peak_memory(self.name, peak_memory)

    def evaluate(
        self, model: Any, loader: Any, context: TrainingContext
    ) -> EvaluationResult:
        module = _legacy_module()
        criterion = nn.CrossEntropyLoss()
        loss, accuracy = module.evaluate_bp_model(
            model,
            loader,
            criterion,
            torch.device(context.device),
            flatten_if_needed(context),
        )
        self.lifecycle.append("global_classifier_inference")
        return evaluation_result({"eval_loss": loss, "eval_accuracy": accuracy})


__all__ = ["BPAdapter"]
