# src/algorithms/mf.py
"""Implementation of the Mono-Forward (MF) algorithm for training MLPs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..architectures.mf_mlp import MF_MLP
from ..contracts import EvaluationResult, TrainingContext, TrainingResult
from ..training.early_stopping import (
    DEFAULT_MIN_DELTA,
    DEFAULT_PATIENCE,
    EarlyStopping,
)
from ..training.loop_support import (
    EpochContext,
    NanLossGuard,
    build_optimizer,
    run_epochs,
)
from ..utils.training_support import (
    create_directory_if_not_exists,
    log_metrics,
    save_checkpoint,
)
from .base import (
    AlgorithmAdapter,
    context_mapping,
    evaluation_result,
    flatten_if_needed,
    result_from_peak_memory,
)
from .mf_math import local_cross_entropy, projection_logits

if TYPE_CHECKING:
    import wandb.sdk.wandb_run


logger = logging.getLogger(__name__)


def mf_local_loss_fn(
    activation_i: torch.Tensor,
    projection_matrix_i: nn.Parameter,
    targets: torch.Tensor,
    criterion: nn.Module | None = None,
) -> torch.Tensor:
    """Compatibility wrapper around the canonical MF local-loss function."""
    return local_cross_entropy(activation_i, projection_matrix_i, targets, criterion)


@torch.no_grad()
def evaluate_mf_local_loss(
    model: MF_MLP,
    matrix_index: int,
    criterion: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    input_adapter: Callable[[torch.Tensor], torch.Tensor],
    log_prefix: str = "Layer",
) -> float:
    """Evaluates local loss for a specific MF layer (M_i) on a validation set."""
    if matrix_index < 0 or matrix_index >= len(model.projection_matrices):
        logger.error(f"{log_prefix} Eval: Matrix index {matrix_index} out of bounds.")
        return float("nan")

    model.eval()
    model.to(device)
    projection_matrix = model.get_projection_matrix(matrix_index).to(device)

    total_loss = 0.0
    total_samples = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        adapted_input = input_adapter(images)
        all_activations = model.forward_with_intermediate_activations(adapted_input)
        if len(all_activations) <= matrix_index:
            logger.error(
                f"{log_prefix} Eval: Activation list too short "
                f"({len(all_activations)}) for index {matrix_index}."
            )
            continue
        activation_a_i = all_activations[matrix_index]

        batch_loss = mf_local_loss_fn(
            activation_a_i, projection_matrix, labels, criterion
        )
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else float("nan")


def train_mf_matrix_only(
    model: MF_MLP,
    matrix_index: int,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    input_adapter: Callable[[torch.Tensor], torch.Tensor],
    early_stopping_config: dict[str, Any],
    val_loader: DataLoader | None = None,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
    log_interval: int = 100,
    step_ref: list[int] | None = None,
    diagnostics: dict[str, float] | None = None,
) -> tuple[float, float, int]:
    """Trains a single projection matrix (M_i) using local loss for an MF_MLP."""
    if step_ref is None:
        step_ref = [-1]
    log_prefix = f"Layer_M{matrix_index}"
    if not 0 <= matrix_index < len(model.projection_matrices):
        raise IndexError(f"Matrix index {matrix_index} out of bounds.")

    projection_matrix = model.get_projection_matrix(matrix_index)
    if not projection_matrix.requires_grad:
        logger.error(f"{log_prefix} requires_grad is False.")
        return float("nan"), 0.0, 0

    model.to(device)
    model.eval()  # Keep feedforward layers frozen
    logger.info(f"--- Starting MF training for {log_prefix} ---")

    es_enabled = early_stopping_config.get("mf_early_stopping_enabled", False)
    if es_enabled and val_loader is None:
        logger.warning(f"{log_prefix}: ES enabled but no val_loader. Disabling.")
        es_enabled = False

    if es_enabled:
        es_patience = early_stopping_config.get(
            "mf_early_stopping_patience", DEFAULT_PATIENCE
        )
        es_min_delta = early_stopping_config.get(
            "mf_early_stopping_min_delta", DEFAULT_MIN_DELTA
        )
        # D1: patience-1 emulates the verbatim legacy "bad epochs >= patience"
        # boundary on EarlyStopping's strict "bad epochs > patience".
        matrix_stopping = EarlyStopping(
            patience=max(int(es_patience) - 1, 0),
            mode="min",
            min_delta=float(es_min_delta),
        )
        logger.info(
            f"{log_prefix}: Early Stopping Enabled - Patience: {es_patience}, "
            f"MinDelta: {es_min_delta}"
        )
    else:
        logger.info(f"{log_prefix}: Early Stopping Disabled.")

    # RUN-004: NaN/Inf loss still breaks the batch loop (legacy stop timing),
    # but a second epoch aborting on NaN/Inf raises instead of looping.
    ctx = EpochContext(
        step_ref=step_ref, log_interval=log_interval, wandb_run=wandb_run
    )
    guard = NanLossGuard()

    def _batch_loss(
        batch_idx: int, images: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor | None:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            adapted_input = input_adapter(images)
            all_activations = model.forward_with_intermediate_activations(adapted_input)
            if len(all_activations) <= matrix_index:
                logger.error(f"{log_prefix} Batch {batch_idx}: Act list too short.")
                return None
            activation_a_i = all_activations[matrix_index]
        return mf_local_loss_fn(activation_a_i, projection_matrix, labels, criterion)

    def _validate(epoch: int) -> bool:
        if not (es_enabled and val_loader is not None):
            return False
        projection_matrix.requires_grad_(False)
        val_loss = evaluate_mf_local_loss(
            model=model,
            matrix_index=matrix_index,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            input_adapter=input_adapter,
            log_prefix=log_prefix,
        )
        logger.info(
            f"{log_prefix} Epoch {epoch + 1}/{epochs} - Val Local Loss: {val_loss:.6f}"
        )
        log_metrics(
            {
                "global_step": step_ref[0],
                f"{log_prefix}/Val_LocalLoss_Epoch": val_loss,
            },
            wandb_run=wandb_run,
            commit=True,
        )
        if matrix_stopping.update(val_loss, epoch + 1):
            logger.info(f"--- {log_prefix}: Early Stopping at Epoch {epoch + 1}! ---")
            return True
        return False

    def _epoch_summary(epoch: int, avg_loss: float) -> None:
        peak_mem_epoch = 0.0  # legacy MF never sampled per-epoch memory here
        logger.info(
            f"{log_prefix} Epoch {epoch + 1}/{epochs} - Train Loss: "
            f"{avg_loss:.6f}, Peak Mem: {peak_mem_epoch:.1f} MiB"
        )
        log_metrics(
            {
                "global_step": step_ref[0],
                f"{log_prefix}/Train_Loss_EpochAvg": avg_loss,
                f"{log_prefix}/Peak_GPU_Mem_Epoch_MiB": peak_mem_epoch,
            },
            wandb_run=wandb_run,
            commit=True,
        )

    outcome = run_epochs(
        ctx,
        train_loader,
        epochs=epochs,
        log_prefix=log_prefix,
        logger=logger,
        optimizer=optimizer,
        guard=guard,
        batch_loss=_batch_loss,
        on_epoch_start=lambda _epoch: projection_matrix.requires_grad_(True),
        on_epoch_summary=_epoch_summary,
        validate=_validate,
        log_batch_loss=True,
    )

    projection_matrix.requires_grad_(False)
    if diagnostics is not None:
        diagnostics["nan_loss_breaks"] = float(guard.strikes)
    logger.info(
        f"--- Finished training for {log_prefix} "
        f"after {outcome.epochs_trained} epochs. ---"
    )
    return outcome.final_avg_epoch_loss, outcome.peak_mem, outcome.epochs_trained


def train_mf_model(
    model: MF_MLP,
    train_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    input_adapter: Callable[[torch.Tensor], torch.Tensor],
    val_loader: DataLoader | None = None,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
    step_ref: list[int] | None = None,
    diagnostics: dict[str, float] | None = None,
) -> float:
    """Orchestrates layer-wise training of MF_MLP: M0, then (W1,M1), (W2,M2), etc."""
    if step_ref is None:
        step_ref = [-1]
    model.to(device)
    num_w_layers = model.num_hidden_layers
    num_m_matrices = len(model.projection_matrices)

    logger.info(
        f"Starting layer-wise MF training for MLP with {num_w_layers} W-layers"
        f" and {num_m_matrices} M-matrices."
    )

    algo_config = config.get("algorithm_params", config.get("training", {}))
    # RUN-004: NaN/Inf loss still breaks the batch loop (legacy stop timing),
    # but a second epoch aborting on NaN/Inf raises instead of looping.
    nan_guard = NanLossGuard()
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    lr = algo_config.get("lr", 0.001)
    weight_decay = algo_config.get("weight_decay", 0.0)
    optimizer_extra_kwargs: dict[str, Any] = {}
    epochs_per_layer = algo_config.get("epochs_per_layer", 5)
    log_interval = algo_config.get("log_interval", 100)
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None)
    mf_criterion = nn.CrossEntropyLoss()

    es_enabled = algo_config.get("mf_early_stopping_enabled", False)
    es_patience = algo_config.get("mf_early_stopping_patience", DEFAULT_PATIENCE)
    es_min_delta = algo_config.get("mf_early_stopping_min_delta", DEFAULT_MIN_DELTA)
    mf_early_stopping_config = {
        "mf_early_stopping_enabled": es_enabled,
        "mf_early_stopping_patience": es_patience,
        "mf_early_stopping_min_delta": es_min_delta,
    }

    def _build_layer_hooks(
        i: int,
        m_idx: int,
        prefix: str,
        projection_matrix: nn.Parameter,
        stopping: EarlyStopping,
    ) -> tuple[
        Callable[[int, Any, Any], torch.Tensor | None],
        Callable[[int], None],
        Callable[[int], bool],
    ]:
        """Bind one layer's epoch-loop hooks (no late-bound loop variables)."""

        def _batch_loss(
            batch_idx: int, images: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor | None:
            images, labels = images.to(device), labels.to(device)
            # Get input for the current layer W_i+1, which is activation a_i
            with torch.no_grad():
                prev_activation = input_adapter(images)
                for k in range(i):  # Recompute forward pass up to layer i-1
                    temp_linear = model.layers[k * 2]
                    temp_act_fn = model.layers[k * 2 + 1]
                    prev_activation = temp_act_fn(temp_linear(prev_activation))

            # Forward through W_i+1 to get a_i+1, with grads for W_i+1
            pre_act_z = model.layers[i * 2](prev_activation.detach())
            activation_a_next = model.layers[i * 2 + 1](pre_act_z)
            return mf_local_loss_fn(
                activation_a_next, projection_matrix, labels, mf_criterion
            )

        def _epoch_start(_epoch: int) -> None:
            model.layers[i * 2].train()
            model.layers[i * 2 + 1].train()
            projection_matrix.requires_grad_(True)

        def _validate(epoch: int) -> bool:
            if not (es_enabled and val_loader is not None):
                return False
            model.eval()  # Set all layers to eval for consistent validation
            val_loss = evaluate_mf_local_loss(
                model, m_idx, mf_criterion, val_loader, device, input_adapter
            )
            if stopping.update(val_loss, epoch + 1):
                logger.info(f"--- {prefix}: Early Stopping at Epoch {epoch + 1}! ---")
                return True
            return False

        return _batch_loss, _epoch_start, _validate

    peak_mem_train = 0.0
    total_epochs_trained_all_layers = 0

    logger.debug("Freezing all model parameters initially.")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # --- Phase 1: Train M0 (on input a0) ---
    if num_m_matrices > 0:
        m0_params = [model.get_projection_matrix(0)]
        model.get_projection_matrix(0).requires_grad_(True)
        m0_optimizer = build_optimizer(
            optimizer_name,
            m0_params,
            lr=lr,
            weight_decay=weight_decay,
            extra_kwargs=optimizer_extra_kwargs,
        )
        _, m0_peak_mem, epochs_trained_m0 = train_mf_matrix_only(
            model=model,
            matrix_index=0,
            optimizer=m0_optimizer,
            criterion=mf_criterion,
            train_loader=train_loader,
            epochs=epochs_per_layer,
            device=device,
            input_adapter=input_adapter,
            early_stopping_config=mf_early_stopping_config,
            val_loader=val_loader,
            wandb_run=wandb_run,
            log_interval=log_interval,
            step_ref=step_ref,
            diagnostics=diagnostics,
        )
        total_epochs_trained_all_layers += epochs_trained_m0
        peak_mem_train = max(peak_mem_train, m0_peak_mem)
        model.get_projection_matrix(0).requires_grad_(False)
        if checkpoint_dir:
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained_index": -1},
                is_best=False,
                filename="mf_matrix_M0_complete.pth",
                checkpoint_dir=checkpoint_dir,
            )

    # --- Phase 2: Train W_i+1 and M_i+1 together ---
    for i in range(num_w_layers):
        w_idx, m_idx = i + 1, i + 1
        log_prefix = f"Layer_W{w_idx}_M{m_idx}"
        logger.info(f"--- Starting MF training for {log_prefix} ---")

        params_to_optimize = []
        if i * 2 < len(model.layers):
            linear_layer = model.layers[i * 2]
            params_to_optimize.extend(list(linear_layer.parameters()))
            for p in linear_layer.parameters():
                p.requires_grad_(True)
            linear_layer.train()
            model.layers[i * 2 + 1].train()  # Associated activation
        else:
            logger.error(f"{log_prefix}: Linear layer index {i * 2} out of range.")
            continue

        projection_matrix = model.get_projection_matrix(m_idx)
        projection_matrix.requires_grad_(True)
        params_to_optimize.append(projection_matrix)

        if not params_to_optimize:
            logger.error(f"{log_prefix}: No parameters to optimize.")
            continue

        optimizer = build_optimizer(
            optimizer_name,
            params_to_optimize,
            lr=lr,
            weight_decay=weight_decay,
            extra_kwargs=optimizer_extra_kwargs,
        )

        peak_mem_layer_train = 0.0
        # D1: patience-1 emulates the verbatim legacy "bad epochs >= patience"
        # boundary on EarlyStopping's strict "bad epochs > patience".
        layer_stopping = EarlyStopping(
            patience=max(int(es_patience) - 1, 0),
            mode="min",
            min_delta=float(es_min_delta),
        )
        layer_batch_loss, layer_epoch_start, layer_validate = _build_layer_hooks(
            i, m_idx, log_prefix, projection_matrix, layer_stopping
        )
        outcome = run_epochs(
            EpochContext(
                step_ref=step_ref, log_interval=log_interval, wandb_run=wandb_run
            ),
            train_loader,
            epochs=epochs_per_layer,
            log_prefix=log_prefix,
            logger=logger,
            optimizer=optimizer,
            guard=nan_guard,
            batch_loss=layer_batch_loss,
            on_epoch_start=layer_epoch_start,
            validate=layer_validate,
        )
        epochs_trained_this_layer = outcome.epochs_trained
        peak_mem_layer_train = max(peak_mem_layer_train, outcome.peak_mem)

        total_epochs_trained_all_layers += epochs_trained_this_layer
        peak_mem_train = max(peak_mem_train, peak_mem_layer_train)
        for p in params_to_optimize:
            p.requires_grad_(False)
        model.eval()

        if checkpoint_dir:
            create_directory_if_not_exists(checkpoint_dir)
            chkpt_filename = f"mf_layer_{w_idx}_complete.pth"
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained_index": i},
                is_best=False,
                filename=chkpt_filename,
                checkpoint_dir=checkpoint_dir,
            )

    logger.info(
        f"Finished all layer-wise MF training. Total Epochs (Sum): "
        f"{total_epochs_trained_all_layers}"
    )
    if diagnostics is not None:
        diagnostics["nan_loss_breaks"] = float(nan_guard.strikes)
    model.eval()
    return peak_mem_train


@torch.no_grad()
def evaluate_mf_model(
    model: MF_MLP,
    data_loader: DataLoader,
    device: torch.device,
    input_adapter: Callable[[torch.Tensor], torch.Tensor],
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """Evaluates the trained MF_MLP using the last activation and projection matrix.

    This uses the activation from the last layer (a_L) and the last projection
    matrix (M_L), where L is the number of hidden layers.
    """
    model.eval()
    model.to(device)
    total_correct, total_samples = 0, 0

    num_layers = model.num_hidden_layers
    last_activation_index = num_layers
    last_projection_matrix_index = num_layers

    logger.info(
        f"Evaluating MF (MLP) using activation a_{last_activation_index} and "
        f"matrix M_{last_projection_matrix_index}."
    )
    if last_projection_matrix_index >= len(model.projection_matrices):
        # EVAL-001: domain guard aborts evaluation instead of returning NaN.
        raise ValueError(
            f"Projection matrix index M_{last_projection_matrix_index} out of "
            f"bounds ({len(model.projection_matrices)} matrices)."
        )
    last_projection_matrix = model.get_projection_matrix(last_projection_matrix_index)

    pbar = tqdm(data_loader, desc="Evaluating MF MLP", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        eval_input = input_adapter(images)

        all_activations = model.forward_with_intermediate_activations(eval_input)

        if len(all_activations) <= last_activation_index:
            # EVAL-001: domain guard aborts evaluation instead of skipping.
            raise ValueError(
                f"Activation list length ({len(all_activations)}) too short "
                f"for a_{last_activation_index}."
            )

        last_activation = all_activations[last_activation_index].to(device)
        last_projection_matrix = last_projection_matrix.to(device)
        goodness_scores = projection_logits(last_activation, last_projection_matrix)
        predicted_labels = torch.argmax(goodness_scores, dim=1)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"MF Evaluation Results (MLP): Accuracy: {accuracy:.2f}%")
    return {"eval_accuracy": accuracy, "eval_loss": float("nan")}


class MFAdapter(AlgorithmAdapter):
    """Preserve M0/layer isolation, detached activations, and MF inference."""

    name = "mf"

    def fit(self, context: TrainingContext) -> TrainingResult:
        model = context.model
        if hasattr(model, "num_hidden_layers"):
            self.lifecycle.append("M0")
            self.lifecycle.extend(
                f"W{i}_M{i}" for i in range(1, int(model.num_hidden_layers) + 1)
            )
        diagnostics: dict[str, float] = {}
        peak_memory = train_mf_model(
            model=model,
            train_loader=context.train_loader,
            config=context_mapping(context),
            device=torch.device(context.device),
            input_adapter=flatten_if_needed(context),  # type: ignore[arg-type]
            val_loader=context.val_loader,
            diagnostics=diagnostics,
        )
        return result_from_peak_memory(self.name, peak_memory, diagnostics)

    def evaluate(
        self, model: Any, loader: Any, context: TrainingContext
    ) -> EvaluationResult:
        values = evaluate_mf_model(
            model,
            loader,
            torch.device(context.device),
            flatten_if_needed(context),
        )
        self.lifecycle.append("last_activation_projection_inference")
        return evaluation_result(values)
