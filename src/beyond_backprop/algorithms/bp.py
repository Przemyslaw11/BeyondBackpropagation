"""Canonical BP adapter over the repository's established BP routines."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..contracts import EvaluationResult, RunStatus, TrainingContext, TrainingResult
from ..training.early_stopping import EarlyStopping
from .base import (
    AlgorithmAdapter,
    context_mapping,
    evaluation_result,
    flatten_if_needed,
    metric,
)


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
        if (
            str(training_config.get("criterion", "CrossEntropyLoss")).lower()
            != "crossentropyloss"
        ):
            raise ValueError("BP supports only CrossEntropyLoss")
        model = context.model.to(context.device)
        input_adapter = flatten_if_needed(context)
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("BP model has no trainable parameters")
        optimizer_name = str(optimizer_config.get("type", "AdamW")).lower()
        optimizer_cls = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }.get(optimizer_name)
        if optimizer_cls is None:
            raise ValueError(f"Unsupported BP optimizer: {optimizer_name}")
        optimizer_kwargs: dict[str, Any] = {
            "lr": optimizer_config.get("lr", 0.001),
            "weight_decay": optimizer_config.get("weight_decay", 0.0),
        }
        if optimizer_name == "sgd":
            optimizer_kwargs["momentum"] = optimizer_config.get("momentum", 0.0)
        if "betas" in optimizer_config:
            optimizer_kwargs["betas"] = tuple(optimizer_config["betas"])
        optimizer = optimizer_cls(parameters, **optimizer_kwargs)
        criterion = nn.CrossEntropyLoss()
        epochs = int(training_config.get("epochs", 1))
        monitor_metric = str(
            training_config.get("early_stopping_metric", "bp_val_loss")
        ).lower()
        mode = str(training_config.get("early_stopping_mode", "min")).lower()
        stopping: EarlyStopping | None = None
        if (
            training_config.get("early_stopping_enabled", False)
            and context.val_loader is not None
        ):
            stopping = EarlyStopping(
                patience=int(training_config.get("early_stopping_patience", 0)),
                mode=mode,
                min_delta=float(training_config.get("early_stopping_min_delta", 0.0)),
            )
        best_value = float("inf") if mode == "min" else -float("inf")
        best_epoch: int | None = None
        checkpoint_path: str | None = None
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            total_samples = 0
            for images, labels in context.train_loader:
                images, labels = images.to(context.device), labels.to(context.device)
                if input_adapter is not None:
                    images = input_adapter(images)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * labels.shape[0]
                total_samples += labels.shape[0]

            val_loss, val_accuracy = (
                self._evaluate_values(
                    model, context.val_loader, context.device, criterion, input_adapter
                )
                if context.val_loader is not None
                else (float("nan"), float("nan"))
            )
            current_value = val_accuracy if "accuracy" in monitor_metric else val_loss
            improved = bool(torch.isfinite(torch.tensor(current_value))) and (
                current_value > best_value
                if mode == "max"
                else current_value < best_value
            )
            if improved or best_epoch is None:
                best_value = current_value
                best_epoch = epoch + 1
                self._snapshot(model)
                if context.checkpoint_manager is not None:
                    name = str(config.get("experiment_name", "model"))
                    epoch_name = f"bp_checkpoint_epoch_{epoch + 1}.pth"
                    best_name = f"bp_{name}_best.pth"
                    context.checkpoint_manager.save(
                        epoch_name,
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        epoch=epoch + 1,
                        algorithm=self.name,
                        best_metric_name=monitor_metric,
                        best_metric_value=float(current_value)
                        if torch.isfinite(torch.tensor(current_value))
                        else None,
                        config_hash=getattr(context.config, "config_hash", None),
                    )
                    context.checkpoint_manager.save(
                        best_name,
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        epoch=epoch + 1,
                        algorithm=self.name,
                        best_metric_name=monitor_metric,
                        best_metric_value=float(current_value)
                        if torch.isfinite(torch.tensor(current_value))
                        else None,
                        config_hash=getattr(context.config, "config_hash", None),
                    )
                    checkpoint_path = str(
                        context.checkpoint_manager.directory / best_name
                    )
            elif context.checkpoint_manager is not None:
                context.checkpoint_manager.save(
                    f"bp_checkpoint_epoch_{epoch + 1}.pth",
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch + 1,
                    algorithm=self.name,
                    best_metric_name=monitor_metric,
                    best_metric_value=float(best_value)
                    if torch.isfinite(torch.tensor(best_value))
                    else None,
                    config_hash=getattr(context.config, "config_hash", None),
                )
            if (
                stopping is not None
                and torch.isfinite(torch.tensor(current_value))
                and stopping.update(current_value, epoch + 1)
            ):
                break
        self.lifecycle.append("global_cross_entropy")
        self.lifecycle.append(
            "adamw"
            if optimizer_config["type"].lower() == "adamw"
            else "configured_optimizer"
        )
        return TrainingResult(
            status=RunStatus.SUCCEEDED,
            best_epoch=best_epoch,
            best_metric=metric(
                best_value,
                unit="percentage_points" if "accuracy" in monitor_metric else "loss",
                source="validation",
            )
            if best_epoch is not None
            else None,
            metrics={
                "training_loss": metric(
                    total_loss / total_samples if total_samples else 0.0, unit="loss"
                ),
                "peak_memory_mib": metric(
                    0.0, unit="MiB", source="device-unavailable", measured=False
                ),
            },
            checkpoint_path=checkpoint_path,
        )

    def evaluate(
        self, model: Any, loader: Any, context: TrainingContext
    ) -> EvaluationResult:
        criterion = nn.CrossEntropyLoss()
        loss, accuracy = self._evaluate_values(
            model, loader, context.device, criterion, flatten_if_needed(context)
        )
        self.lifecycle.append("global_classifier_inference")
        return evaluation_result({"eval_loss": loss, "eval_accuracy": accuracy})

    @staticmethod
    def _evaluate_values(
        model: Any, loader: Any, device: Any, criterion: nn.Module, input_adapter: Any
    ) -> tuple[float, float]:
        if loader is None:
            return float("nan"), float("nan")
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                if input_adapter is not None:
                    images = input_adapter(images)
                logits = model(images)
                loss = criterion(logits, labels)
                total_loss += float(loss) * labels.shape[0]
                total_correct += int((logits.argmax(dim=1) == labels).sum())
                total_samples += labels.shape[0]
        return (
            total_loss / total_samples if total_samples else float("nan"),
            100.0 * total_correct / total_samples if total_samples else float("nan"),
        )


__all__ = ["BPAdapter"]
