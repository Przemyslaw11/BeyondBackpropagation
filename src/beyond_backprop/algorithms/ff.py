"""Implementation of Hinton's Forward-Forward (FF) training and evaluation logic."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..architectures.ff_mlp import FF_MLP
from ..contracts import EvaluationResult, TrainingContext, TrainingResult
from ..training.early_stopping import EarlyStopping
from ..utils.training_support import (
    create_directory_if_not_exists,
    format_time,
    get_gpu_memory_usage,
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
from .ff_math import (
    aggregate_goodness,
    generate_hinton_inputs,
    linear_cooldown_lr,
)

if TYPE_CHECKING:
    import wandb.sdk.wandb_run


logger = logging.getLogger(__name__)


def train_ff_model(
    model: FF_MLP,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: dict[str, Any],
    device: torch.device,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
    input_adapter: Callable[[torch.Tensor], torch.Tensor] | None = None,
    step_ref: list[int] | None = None,
    gpu_handle: Any | None = None,
    nvml_active: bool = False,
    on_best_epoch: Callable[[], None] | None = None,
    diagnostics: dict[str, float] | None = None,
) -> float:
    """Orchestrates end-to-end training of a model using the Forward-Forward algorithm.

    MODIFIED: Added Early Stopping logic based on validation accuracy.
    Returns the peak GPU memory observed during training.
    """
    if step_ref is None:
        step_ref = [-1]
    model.to(device)
    logger.info(
        "Starting Forward-Forward (Hinton style) training using modified FF_MLP."
    )
    if input_adapter is not None:
        logger.warning(
            "FF Training: 'input_adapter' provided but FF_MLP uses internal "
            "logic. Adapter ignored."
        )

    train_config = config.get("training", {})
    algo_config = config.get("algorithm_params", {})
    data_config = config.get("data", {})
    checkpoint_config = config.get("checkpointing", {})

    try:
        initial_ff_lr = float(algo_config.get("ff_learning_rate", 1e-3))
        initial_ds_lr = float(algo_config.get("downstream_learning_rate", 1e-2))
        ff_wd = float(algo_config.get("ff_weight_decay", 3e-4))
        ds_wd = float(algo_config.get("downstream_weight_decay", 3e-3))
        ff_momentum = float(algo_config.get("ff_momentum", 0.9))
        ds_momentum = float(algo_config.get("downstream_momentum", 0.9))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid LR/WD/Momentum format: {e}") from e

    optimizer_type = algo_config.get("optimizer_type", "SGD")
    epochs = train_config.get("epochs", 100)
    log_interval = train_config.get("log_interval", 100)
    num_classes = data_config.get("num_classes", 10)
    checkpoint_dir = checkpoint_config.get("checkpoint_dir", None)

    # --- Early Stopping Setup ---
    es_enabled = train_config.get("early_stopping_enabled", True)
    es_metric_key = train_config.get(
        "early_stopping_metric", "FF_Hinton/Val_Acc_Epoch"
    ).lower()
    es_patience = train_config.get("early_stopping_patience", 10)
    es_mode = train_config.get("early_stopping_mode", "max").lower()
    es_min_delta = train_config.get("early_stopping_min_delta", 0.0)
    best_checkpoint_metric_value = -float("inf") if es_mode == "max" else float("inf")

    if es_enabled:
        if val_loader is None:
            logger.warning(
                "Early stopping enabled but no validation loader provided. "
                "Disabling early stopping."
            )
            es_enabled = False
        else:
            if (es_mode == "min" and "acc" in es_metric_key) or (
                es_mode == "max" and "loss" in es_metric_key
            ):
                logger.error(
                    f"Early stopping mode '{es_mode}' incompatible with metric "
                    f"key '{es_metric_key}'. Disabling."
                )
                es_enabled = False
            else:
                logger.info(
                    f"Early stopping enabled: Metric Key='{es_metric_key}', "
                    f"Patience={es_patience}, Mode='{es_mode}', MinDelta={es_min_delta}"
                )
    else:
        logger.info("Early stopping disabled.")

    if es_enabled:
        # D1: patience-1 emulates the verbatim legacy "bad epochs >= patience"
        # boundary on EarlyStopping's strict "bad epochs > patience", keeping
        # published stop timing byte-identical on finite/NaN streams.
        ff_stopping = EarlyStopping(
            patience=max(int(es_patience) - 1, 0),
            mode=es_mode,
            min_delta=float(es_min_delta),
        )

    # --- Optimizer Setup ---
    ff_layer_params = [
        p for layer in model.layers for p in layer.parameters() if p.requires_grad
    ]
    classifier_params = [
        p for p in model.linear_classifier.parameters() if p.requires_grad
    ]
    optimizer_groups = []
    if ff_layer_params:
        group_ff = {
            "params": ff_layer_params,
            "lr": initial_ff_lr,
            "weight_decay": ff_wd,
        }
        if optimizer_type.lower() == "sgd":
            group_ff["momentum"] = ff_momentum
        optimizer_groups.append(group_ff)
        log_msg = f"Opt Group 0 (FF): Initial LR={initial_ff_lr}, WD={ff_wd}"
        if "momentum" in group_ff:
            log_msg += f", Mom={ff_momentum}"
        logger.info(log_msg)

    if classifier_params:
        group_ds = {
            "params": classifier_params,
            "lr": initial_ds_lr,
            "weight_decay": ds_wd,
        }
        if optimizer_type.lower() == "sgd":
            group_ds["momentum"] = ds_momentum
        optimizer_groups.append(group_ds)
        log_msg = f"Opt Group 1 (DS): Initial LR={initial_ds_lr}, WD={ds_wd}"
        if "momentum" in group_ds:
            log_msg += f", Mom={ds_momentum}"
        logger.info(log_msg)

    if not optimizer_groups:
        logger.error("FF_MLP: No trainable parameters found.")
        return 0.0

    try:
        if optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(optimizer_groups)
        elif optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(optimizer_groups)
        elif optimizer_type.lower() == "adam":
            optimizer = optim.Adam(optimizer_groups)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        logger.info(f"Using optimizer: {optimizer_type}")
    except Exception as e:
        logger.error(f"Failed to create optimizer: {e}", exc_info=True)
        return 0.0

    # --- Training Loop Initialization ---
    peak_mem_train = 0.0
    run_start_time = time.time()

    # --- Epoch Loop ---
    skipped_batches = 0
    # RUN-002: exception-driven batch skips are tolerated only up to ~1% of
    # the epoch; beyond that something is broken and the run must fail.
    max_skipped_batches = max(1, len(train_loader) // 100)

    for epoch in range(epochs):
        # --- LR Schedule Update ---
        current_lr_ff = linear_cooldown_lr(initial_ff_lr, epoch, epochs)
        current_lr_ds = linear_cooldown_lr(initial_ds_lr, epoch, epochs)
        if len(optimizer.param_groups) > 0:
            optimizer.param_groups[0]["lr"] = current_lr_ff
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]["lr"] = current_lr_ds
        logger.debug(
            f"Epoch {epoch + 1}/{epochs}: LR Update - "
            f"FF={current_lr_ff:.6f}, DS={current_lr_ds:.6f}"
        )

        # --- Training Phase ---
        model.train()
        epoch_start_time = time.time()
        epoch_total_loss, epoch_samples = 0.0, 0
        epoch_ff_loss_total, epoch_peer_loss_total = 0.0, 0.0
        epoch_cls_loss_total, epoch_cls_acc_total = 0.0, 0.0
        epoch_layer_ff_acc_sum = {
            f"Layer_{i + 1}": 0.0 for i in range(model.num_layers)
        }
        peak_mem_epoch = 0.0

        pbar = tqdm(train_loader, desc=f"FF Epoch {epoch + 1}/{epochs}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1
            current_global_step = step_ref[0]
            current_batch_size = images.size(0)
            images, labels = images.to(device), labels.to(device)

            try:
                pos_images_flat, neg_images_flat, _ = generate_hinton_inputs(
                    images, labels, num_classes, device
                )
            except Exception as e_gen:
                skipped_batches += 1
                logger.error(f"Input gen error: {e_gen}", exc_info=True)
                if skipped_batches > max_skipped_batches:
                    raise RuntimeError(
                        f"FF training skipped {skipped_batches} batches "
                        f"(threshold {max_skipped_batches}); aborting run."
                    ) from e_gen
                continue

            stacked_z = torch.cat([pos_images_flat, neg_images_flat], dim=0)
            posneg_labels = torch.zeros(stacked_z.shape[0], device=device)
            posneg_labels[:current_batch_size] = 1

            try:
                ff_combined_loss, ff_metrics_dict = model.forward_ff_train(
                    stacked_z, posneg_labels, current_batch_size
                )
                cls_loss, cls_accuracy = model.forward_downstream_only(labels)

                if torch.isnan(ff_combined_loss) or torch.isinf(ff_combined_loss):
                    logger.warning(
                        f"NaN/Inf FF loss ({ff_combined_loss.item()}) "
                        f"encountered at step {current_global_step}. Skipping batch."
                    )
                    continue
                if torch.isnan(cls_loss) or torch.isinf(cls_loss):
                    logger.warning(
                        f"NaN/Inf Cls loss ({cls_loss.item()}) "
                        f"encountered at step {current_global_step}. Skipping batch."
                    )
                    continue

                if not cls_loss.requires_grad and any(
                    p.requires_grad for p in model.linear_classifier.parameters()
                ):
                    cls_loss = cls_loss.clone().requires_grad_(True)

                total_batch_loss = ff_combined_loss + cls_loss
            except Exception as e_fwd:
                skipped_batches += 1
                logger.error(
                    f"Forward/loss error at step {current_global_step}: {e_fwd}",
                    exc_info=True,
                )
                if skipped_batches > max_skipped_batches:
                    raise RuntimeError(
                        f"FF training skipped {skipped_batches} batches "
                        f"(threshold {max_skipped_batches}); aborting run."
                    ) from e_fwd
                continue

            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                skipped_batches += 1
                logger.error(
                    f"NaN/Inf total loss before backward ({total_batch_loss.item()}) "
                    f"at step {current_global_step}. Skipping batch update."
                )
                continue

            optimizer.zero_grad()
            try:
                total_batch_loss.backward()
            except Exception as e_bwd:
                logger.error(
                    f"Backward pass error at step {current_global_step}: {e_bwd}",
                    exc_info=True,
                )
                continue

            optimizer.step()

            # --- Accumulate Epoch Metrics ---
            epoch_total_loss += total_batch_loss.item() * current_batch_size
            epoch_ff_loss_total += (
                ff_metrics_dict.get("FF_Loss_Total", torch.tensor(0.0)).item()
                * current_batch_size
            )
            epoch_peer_loss_total += (
                ff_metrics_dict.get(
                    "Peer_Normalization_Loss_Total", torch.tensor(0.0)
                ).item()
                * current_batch_size
            )
            epoch_cls_loss_total += cls_loss.item() * current_batch_size
            epoch_cls_acc_total += cls_accuracy * current_batch_size
            epoch_samples += current_batch_size
            for i in range(model.num_layers):
                key = f"Layer_{i + 1}/FF_Accuracy"
                epoch_layer_ff_acc_sum[f"Layer_{i + 1}"] += (
                    ff_metrics_dict.get(key, 0.0) * current_batch_size
                )

            # --- Memory Monitoring & Logging ---
            # P2: sample NVML only at logging boundaries instead of every
            # batch (cadence parity with MF); peak tracking therefore samples
            # at log_interval cadence -- documented measurement note.
            is_log_time = (batch_idx + 1) % log_interval == 0 or (
                batch_idx == len(train_loader) - 1
            )
            current_mem_used = float("nan")
            if nvml_active and gpu_handle and is_log_time:
                mem_info = get_gpu_memory_usage(gpu_handle)
                current_mem_used = mem_info[0] if mem_info else float("nan")
                if not math.isnan(current_mem_used):
                    peak_mem_epoch = max(peak_mem_epoch, current_mem_used)

            if is_log_time:
                metrics_to_log = {
                    "global_step": current_global_step,
                    "FF_Hinton/Train_Loss_Batch": total_batch_loss.item(),
                    "FF_Hinton/FF_Loss_Batch": ff_metrics_dict.get(
                        "FF_Loss_Total", torch.tensor(0.0)
                    ).item(),
                    "FF_Hinton/PeerNorm_Loss_Batch": ff_metrics_dict.get(
                        "Peer_Normalization_Loss_Total", torch.tensor(0.0)
                    ).item(),
                    "FF_Hinton/Cls_Loss_Batch": cls_loss.item(),
                    "FF_Hinton/Cls_Acc_Batch": cls_accuracy,
                }
                for i in range(model.num_layers):
                    key = f"Layer_{i + 1}/FF_Accuracy"
                    metrics_to_log[f"Layer_{i + 1}/FF_Acc_Batch"] = ff_metrics_dict.get(
                        key, 0.0
                    )
                if not math.isnan(current_mem_used):
                    metrics_to_log["FF_Hinton/GPU_Mem_Used_MiB_Batch"] = (
                        current_mem_used
                    )
                log_metrics(metrics_to_log, wandb_run=wandb_run, commit=True)
                pbar.set_postfix(
                    loss=f"{total_batch_loss.item():.4f}",
                    cls_acc=f"{cls_accuracy:.2f}%",
                )

        peak_mem_train = max(peak_mem_train, peak_mem_epoch)

        if epoch_samples == 0:
            logger.warning(
                f"Epoch {epoch + 1} completed with 0 samples processed. "
                "Skipping evaluation and logging."
            )
            continue

        avg_epoch_loss = epoch_total_loss / epoch_samples
        avg_ff_loss = epoch_ff_loss_total / epoch_samples
        avg_peer_loss = epoch_peer_loss_total / epoch_samples
        avg_cls_loss = epoch_cls_loss_total / epoch_samples
        avg_cls_acc = epoch_cls_acc_total / epoch_samples
        epoch_layer_ff_acc_avg = {
            f"Layer_{i + 1}": epoch_layer_ff_acc_sum[f"Layer_{i + 1}"] / epoch_samples
            for i in range(model.num_layers)
        }
        epoch_duration = time.time() - epoch_start_time

        val_results = {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
        if val_loader:
            val_results = evaluate_ff_model(model, val_loader, device)
            logger.info(
                f"FF Validation Epoch {epoch + 1}/{epochs} - Accuracy: "
                f"{val_results.get('eval_accuracy', 'N/A'):.2f}%"
            )
        else:
            logger.warning("No validation loader provided. Skipping validation.")

        current_global_step = step_ref[0]
        epoch_summary_metrics = {
            "global_step": current_global_step,
            "FF_Hinton/Train_Loss_Epoch": avg_epoch_loss,
            "FF_Hinton/FF_Loss_Epoch": avg_ff_loss,
            "FF_Hinton/PeerNorm_Loss_Epoch": avg_peer_loss,
            "FF_Hinton/Cls_Loss_Epoch": avg_cls_loss,
            "FF_Hinton/Cls_Acc_Epoch": avg_cls_acc,
            "FF_Hinton/Val_Acc_Epoch": val_results.get("eval_accuracy", float("nan")),
            "FF_Hinton/Epoch_Duration_Sec": epoch_duration,
            "FF_Hinton/LR_FF_Layers": current_lr_ff,
            "FF_Hinton/LR_Downstream": current_lr_ds,
            "FF_Hinton/Epoch": epoch + 1,
            "FF_Hinton/Peak_GPU_Mem_Epoch_MiB": peak_mem_epoch,
        }
        for i in range(model.num_layers):
            key = f"Layer_{i + 1}/FF_Acc_EpochAvg"
            epoch_summary_metrics[key] = epoch_layer_ff_acc_avg[f"Layer_{i + 1}"]
        log_metrics(epoch_summary_metrics, wandb_run=wandb_run, commit=True)
        log_msg = (
            f"FF Epoch {epoch + 1}/{epochs} | Train Loss: {avg_epoch_loss:.4f}, "
            f"Cls Acc: {avg_cls_acc:.2f}% | "
            f"Val Acc: {val_results.get('eval_accuracy', 'N/A'):.2f}% | "
            f"Peak Mem: {peak_mem_epoch:.1f} MiB | "
            f"Duration: {format_time(epoch_duration)}"
        )
        logger.info(log_msg)

        current_metric_value = val_results.get("eval_accuracy", float("nan"))
        is_best_for_checkpointing = False

        if es_enabled:
            if math.isnan(current_metric_value):
                logger.warning(
                    f"Epoch {epoch + 1}: Early stopping metric '{es_metric_key}' is "
                    "NaN. Treating as no improvement."
                )
            should_stop = ff_stopping.update(current_metric_value, epoch + 1)
            new_best = ff_stopping.best_value
            if ff_stopping.best_epoch == epoch + 1 and new_best is not None:
                is_best_for_checkpointing = True
                best_checkpoint_metric_value = float(new_best)
                if on_best_epoch is not None:
                    # BEST-001: capture best-validation weights in memory so
                    # final evaluation is independent of checkpoint_dir.
                    on_best_epoch()
                logger.info(
                    f"Epoch {epoch + 1}: Early stopping metric improved to "
                    f"{new_best:.4f}. Reset patience."
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}: Early stopping metric did not improve. "
                    f"Patience: {ff_stopping.bad_epochs}/{es_patience}."
                )

            if should_stop:
                logger.info("--- Early Stopping Triggered ---")
                logger.info(
                    f"Metric '{es_metric_key}' did not improve for {es_patience} "
                    f"epochs (Best: {ff_stopping.best_value:.4f})."
                )
                logger.info(f"Stopping training at epoch {epoch + 1}.")
                break
        else:
            if not math.isnan(current_metric_value):  # noqa: SIM102 - early-stopping structure kept verbatim
                if (
                    es_mode == "max"
                    and (current_metric_value > best_checkpoint_metric_value)
                ) or (
                    es_mode == "min"
                    and (current_metric_value < best_checkpoint_metric_value)
                ):
                    best_checkpoint_metric_value = current_metric_value
                    is_best_for_checkpointing = True
            if is_best_for_checkpointing:
                logger.info(
                    f"Epoch {epoch + 1}: New best checkpoint metric: "
                    f"{best_checkpoint_metric_value:.4f}"
                )
                if on_best_epoch is not None:
                    on_best_epoch()

        if checkpoint_dir:
            create_directory_if_not_exists(checkpoint_dir)
            exp_name = config.get("experiment_name", "model")
            save_checkpoint(
                state={
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_metric_value": best_checkpoint_metric_value,
                    "val_accuracy": current_metric_value,
                },
                is_best=is_best_for_checkpointing,
                checkpoint_dir=checkpoint_dir,
                filename=f"ff_checkpoint_epoch_{epoch + 1}.pth",
                best_filename=f"ff_{exp_name}_best.pth",
            )

    total_training_time = time.time() - run_start_time
    logger.info(
        "Finished Forward-Forward (Hinton) training loop. Total time: "
        f"{format_time(total_training_time)}"
    )

    # BEST-001: best-validation weights are captured in memory via the
    # on_best_epoch callback and restored by AlgorithmAdapter before final
    # evaluation. The legacy ff_<exp>_best.pth disk round-trip is gone;
    # checkpoint files themselves are still written unchanged.

    # RUN-002: surface batch accounting without polluting measurements.
    if diagnostics is not None:
        diagnostics["skipped_batches"] = float(skipped_batches)
    if skipped_batches:
        logger.info(
            f"FF training diagnostics: {skipped_batches} batches skipped "
            f"(threshold {max_skipped_batches})."
        )

    logger.info(
        "NOTE: Reference implementation used PyTorch 1.11. Your environment uses "
        f"{torch.__version__}. Small differences in final accuracy might arise "
        "from library versions or hardware."
    )

    return peak_mem_train


def evaluate_ff_model(
    model: FF_MLP,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluates the FF model using the multi-pass inference method."""
    model.eval()
    model.to(device)
    num_classes = model.num_classes
    logger.info("Evaluating FF (Hinton style) model using multi-pass inference.")
    total_correct, total_samples = 0, 0
    failed_batches = 0
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating FF (Hinton) Model", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]
            try:
                # P4: all label candidates evaluated in one stacked forward
                # pass instead of num_classes sequential passes. Rows are
                # independent through every layer, so results are identical
                # to the candidate loop (pinned by test_ff_eval_batching.py).
                # ponytail: peak activation memory grows ~num_classes x vs
                # the loop; chunk stacked rows if that ever becomes a problem.
                stacked_images = images.repeat_interleave(num_classes, dim=0)
                candidate_labels = torch.arange(num_classes, device=device).repeat(
                    batch_size
                )
                ff_input_candidates, _, _ = generate_hinton_inputs(
                    stacked_images, candidate_labels, num_classes, device
                )
                layer_goodness_list = model.forward_goodness_per_layer(
                    ff_input_candidates
                )
            except Exception:
                # EVAL-001: zero tolerance - a single failed batch aborts
                # evaluation instead of substituting predictions.
                failed_batches += 1
                logger.exception(f"Eval fwd error batch {batch_idx + 1}.")
                raise
            # Domain guards are explicit raises, not exception substitutions.
            if not layer_goodness_list:
                raise ValueError("Eval produced no goodness layers.")
            total_goodness = aggregate_goodness(layer_goodness_list)
            if total_goodness.shape != (batch_size * num_classes,):
                raise ValueError(f"Bad goodness shape: {total_goodness.shape}")
            batch_total_goodness = total_goodness.reshape(batch_size, num_classes)

            all_inf_mask = torch.all(
                torch.isinf(batch_total_goodness) & (batch_total_goodness < 0),
                dim=1,
            )
            predicted_labels = torch.zeros_like(labels)
            if torch.any(~all_inf_mask):
                valid_indices = ~all_inf_mask
                predicted_labels[valid_indices] = torch.argmax(
                    batch_total_goodness[valid_indices], dim=1
                )
            if torch.any(all_inf_mask):
                logger.warning(
                    f"Eval Batch {batch_idx + 1}: {all_inf_mask.sum().item()}/"
                    f"{batch_size} samples failed all candidates."
                )
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size
    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    logger.info(f"FF Evaluation Accuracy (Hinton Multi-Pass): {accuracy:.2f}%")
    return {"eval_accuracy": accuracy, "eval_loss": float("nan")}


class FFAdapter(AlgorithmAdapter):
    """Preserve local goodness updates, downstream training, and FF inference."""

    name = "ff"

    def fit(self, context: TrainingContext) -> TrainingResult:
        self.lifecycle.extend(("local_goodness_updates", "downstream_classifier"))
        diagnostics: dict[str, float] = {}
        peak_memory = train_ff_model(
            model=context.model,
            train_loader=context.train_loader,
            val_loader=context.val_loader,
            config=context_mapping(context),
            device=torch.device(context.device),
            input_adapter=flatten_if_needed(context),
            on_best_epoch=lambda: self._snapshot(context.model),
            diagnostics=diagnostics,
        )
        return result_from_peak_memory(self.name, peak_memory, diagnostics)

    def evaluate(
        self, model: Any, loader: Any, context: TrainingContext
    ) -> EvaluationResult:
        values = evaluate_ff_model(model, loader, torch.device(context.device))
        self.lifecycle.append("algorithm_specific_inference")
        return evaluation_result(values)
