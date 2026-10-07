# File: ./src/baselines/bp_ds.py
"""The joint-gradient rungs of the ablation ladder: BP-DS and MF-Joint.

Reviewers asked whether Mono-Forward's gains come from its forward-only mechanism
or merely from placing a loss at every layer. Answering that needs two
intermediate baselines, and both attach MF's local loss to every layer and
propagate joint gradients. They differ only in which head makes the prediction,
so they share one training loop:

    BP-DS     reads out through ``output_layer``; M_0..M_L are auxiliary heads.
    MF-Joint  reads out through M_L; M_0..M_{L-1} are auxiliary heads.

At ``aux_weight`` 1.0 MF-Joint optimises exactly MF's sum of local losses, so it
differs from MF by detachment alone.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import pynvml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.algorithms.mf import assert_activation_cache_allowed, mf_local_loss_fn
from src.architectures.mf_mlp import MF_MLP
from src.utils.early_stopping import resolve_early_stopping
from src.utils.helpers import format_time
from src.utils.logging_utils import log_metrics
from src.utils.monitoring import get_gpu_memory_usage

if TYPE_CHECKING:
    import wandb.sdk.wandb_run

logger = logging.getLogger(__name__)

READOUT_OUTPUT_LAYER = "output_layer"
READOUT_PROJECTION = "projection"


def _auxiliary_indices(model: MF_MLP, readout: str) -> List[int]:
    """Returns the M_i that act as auxiliary heads rather than as the readout."""
    last = model.num_hidden_layers
    if readout == READOUT_OUTPUT_LAYER:
        return list(range(last + 1))
    return list(range(last))


def _readout_logits(
    model: MF_MLP, activations: List[torch.Tensor], readout: str
) -> torch.Tensor:
    """Scores a_L through the head this rung predicts with."""
    last = model.num_hidden_layers
    if readout == READOUT_OUTPUT_LAYER:
        return model.output_layer(activations[last])
    # mf_local_loss_fn's goodness step, expanded so the logits can also score accuracy.
    return torch.matmul(activations[last], model.get_projection_matrix(last).t())


def _select_trainable_parameters(model: MF_MLP, readout: str) -> List[nn.Parameter]:
    """Freezes any head this rung never reads, so it collects no optimiser state."""
    if readout == READOUT_OUTPUT_LAYER:
        for param in model.parameters():
            param.requires_grad_(True)
        return [p for p in model.parameters() if p.requires_grad]

    for name, param in model.named_parameters():
        param.requires_grad_(not name.startswith("output_layer"))
    return [p for p in model.parameters() if p.requires_grad]


def _deep_supervised_losses(
    model: MF_MLP,
    activations: List[torch.Tensor],
    targets: torch.Tensor,
    criterion: nn.Module,
    readout: str,
    aux_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Returns (total loss, readout logits, auxiliary loss sum)."""
    logits = _readout_logits(model, activations, readout)
    total = criterion(logits, targets)
    readout_only = float(total.item())

    for index in _auxiliary_indices(model, readout):
        total = total + aux_weight * mf_local_loss_fn(
            activations[index], model.get_projection_matrix(index), targets, criterion
        )
    return total, logits, float(total.item()) - readout_only


def _evaluate(
    model: MF_MLP,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]],
    readout: str,
) -> Tuple[float, float]:
    """Scores the readout head only; auxiliary heads never make a prediction."""
    model.eval()
    model.to(device)
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            adapted = input_adapter(images) if input_adapter else images
            activations = model.forward_with_intermediate_activations(adapted)
            logits = _readout_logits(model, activations, readout)

            batch_size = labels.size(0)
            total_loss += criterion(logits, labels).item() * batch_size
            total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        return float("nan"), float("nan")
    return total_loss / total_samples, (total_correct / total_samples) * 100.0


def _train_epoch(
    model: MF_MLP,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    readout: str,
    aux_weight: float,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]],
    log_prefix: str,
    wandb_run: "Optional[wandb.sdk.wandb_run.Run]",
    log_interval: int,
    step_ref: List[int],
    gpu_handle: Optional[pynvml.c_nvmlDevice_t],
    nvml_active: bool,
) -> Tuple[float, float, float]:
    """One joint-gradient epoch over the readout plus every auxiliary head."""
    model.train()
    epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
    peak_mem_epoch = 0.0

    pbar = tqdm(
        train_loader, desc=f"{log_prefix} Epoch {epoch + 1}/{total_epochs}", leave=False
    )
    for batch_idx, (images, labels) in enumerate(pbar):
        step_ref[0] += 1
        images, labels = images.to(device), labels.to(device)
        adapted = input_adapter(images) if input_adapter else images

        activations = model.forward_with_intermediate_activations(adapted)
        loss, logits, aux_component = _deep_supervised_losses(
            model, activations, labels, criterion, readout, aux_weight
        )

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"{log_prefix}: NaN/Inf loss at epoch {epoch + 1}, batch {batch_idx}.")
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        with torch.no_grad():
            batch_correct = (torch.argmax(logits, dim=1) == labels).sum().item()
        epoch_loss += loss.item() * batch_size
        epoch_correct += batch_correct
        epoch_samples += batch_size

        is_log_time = (batch_idx + 1) % log_interval == 0
        is_last_batch = batch_idx == len(train_loader) - 1
        if nvml_active and gpu_handle and (is_log_time or is_last_batch):
            mem_info = get_gpu_memory_usage(gpu_handle)
            if mem_info:
                peak_mem_epoch = max(peak_mem_epoch, mem_info[0])

        if is_log_time or is_last_batch:
            batch_accuracy = (batch_correct / batch_size) * 100.0 if batch_size else 0.0
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_accuracy:.2f}%")
            log_metrics(
                {
                    "global_step": step_ref[0],
                    f"{log_prefix}/Train_Loss_Batch": loss.item(),
                    f"{log_prefix}/Train_Acc_Batch": batch_accuracy,
                    f"{log_prefix}/Aux_Loss_Batch": aux_component,
                },
                wandb_run=wandb_run,
                commit=True,
            )

    if epoch_samples == 0:
        return float("nan"), float("nan"), peak_mem_epoch
    return (
        epoch_loss / epoch_samples,
        (epoch_correct / epoch_samples) * 100.0,
        peak_mem_epoch,
    )


def train_deep_supervised_model(
    model: MF_MLP,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
    device: torch.device,
    readout: str,
    default_optimizer: str,
    log_prefix: str,
    wandb_run: "Optional[wandb.sdk.wandb_run.Run]" = None,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    step_ref: Optional[List[int]] = None,
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float:
    """Trains every layer jointly against its own head. Returns peak GPU memory."""
    if not isinstance(model, MF_MLP):
        raise TypeError(
            f"{log_prefix} needs the projection matrices of MF_MLP, got "
            f"{type(model).__name__}."
        )
    if step_ref is None:
        step_ref = [-1]

    model.to(device)
    start_time = time.time()

    optimizer_config = config.get("optimizer", {})
    algo_params = config.get("algorithm_params", {})
    train_config = config.get("training", {})
    aux_weight = float(algo_params.get("aux_weight", 1.0))
    log_interval = train_config.get("log_interval", 100)

    cache_strategy = str(algo_params.get("activation_cache", "recompute")).lower()
    if cache_strategy != "recompute":
        # Joint gradients restale a cached activation within one step; this always raises.
        assert_activation_cache_allowed(config, train_loader, cache_strategy)

    if config.get("checkpointing", {}).get("checkpoint_dir"):
        logger.warning(
            "%s ignores checkpoint_dir: the ladder compares wall time and energy, "
            "and per-epoch checkpoint I/O differs per rung.",
            log_prefix,
        )

    es_policy = resolve_early_stopping(config)
    epochs = es_policy["max_epochs"]
    es_enabled = es_policy["enabled"] and val_loader is not None
    es_metric, es_mode = es_policy["metric"], es_policy["mode"]
    es_patience, es_min_delta = es_policy["patience"], es_policy["min_delta"]
    epochs_no_improve = 0
    best_es_metric_value = float("inf") if es_mode == "min" else -float("inf")

    criterion = nn.CrossEntropyLoss()
    params_to_optimize = _select_trainable_parameters(model, readout)
    if not params_to_optimize:
        logger.error(f"{log_prefix}: no trainable parameters.")
        return 0.0

    optimizer_name = optimizer_config.get("type", default_optimizer)
    optimizer = getattr(optim, optimizer_name)(
        params_to_optimize,
        lr=optimizer_config.get("lr", 0.001),
        weight_decay=optimizer_config.get("weight_decay", 0.0),
        **optimizer_config.get("params", {}),
    )
    logger.info(
        "%s: %d auxiliary heads, aux_weight %.4g, readout '%s', optimizer %s.",
        log_prefix,
        len(_auxiliary_indices(model, readout)),
        aux_weight,
        readout,
        optimizer_name,
    )

    peak_mem_train = 0.0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss, train_acc, peak_mem_epoch = _train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            epochs,
            readout,
            aux_weight,
            input_adapter,
            log_prefix,
            wandb_run,
            log_interval,
            step_ref,
            gpu_handle,
            nvml_active,
        )
        peak_mem_train = max(peak_mem_train, peak_mem_epoch)
        if torch.isnan(torch.tensor(train_loss)):
            logger.error(f"{log_prefix}: aborting after a non-finite epoch loss.")
            break

        val_loss, val_acc = float("nan"), float("nan")
        if val_loader is not None:
            val_loss, val_acc = _evaluate(
                model, val_loader, criterion, device, input_adapter, readout
            )

        epoch_duration = time.time() - epoch_start_time
        log_metrics(
            {
                "global_step": step_ref[0],
                f"{log_prefix}/Train_Loss_Epoch": train_loss,
                f"{log_prefix}/Train_Acc_Epoch": train_acc,
                f"{log_prefix}/Val_Loss_Epoch": val_loss,
                f"{log_prefix}/Val_Acc_Epoch": val_acc,
                f"{log_prefix}/Epoch_Duration_Sec": epoch_duration,
                f"{log_prefix}/Learning_Rate": optimizer.param_groups[0]["lr"],
                f"{log_prefix}/Epoch": epoch + 1,
                f"{log_prefix}/Peak_GPU_Mem_Epoch_MiB": peak_mem_epoch,
            },
            wandb_run=wandb_run,
            commit=True,
        )
        logger.info(
            f"{log_prefix} Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
            f"Duration: {format_time(epoch_duration)}"
        )

        if not es_enabled:
            continue

        current = val_acc if "acc" in es_metric else val_loss
        if torch.isnan(torch.tensor(current)):
            epochs_no_improve += 1
        elif es_mode == "min" and current < best_es_metric_value - es_min_delta:
            best_es_metric_value, epochs_no_improve = current, 0
        elif es_mode == "max" and current > best_es_metric_value + es_min_delta:
            best_es_metric_value, epochs_no_improve = current, 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= es_patience:
            logger.info(
                f"--- {log_prefix}: early stopping at epoch {epoch + 1} "
                f"('{es_metric}' flat for {es_patience} epochs) ---"
            )
            break

    config.setdefault("_run_stats", {})["epochs_completed"] = epoch + 1
    logger.info(
        f"{log_prefix} training finished in {format_time(time.time() - start_time)}."
    )
    model.eval()
    return peak_mem_train


def train_bp_ds_model(
    model: MF_MLP,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: "Optional[wandb.sdk.wandb_run.Run]" = None,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    step_ref: Optional[List[int]] = None,
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float:
    """Rung 2: backpropagation with a matching layer-wise auxiliary loss."""
    return train_deep_supervised_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        readout=READOUT_OUTPUT_LAYER,
        default_optimizer="AdamW",
        log_prefix="BP_DS",
        wandb_run=wandb_run,
        input_adapter=input_adapter,
        step_ref=step_ref,
        gpu_handle=gpu_handle,
        nvml_active=nvml_active,
    )


def evaluate_bp_ds_model(
    model: MF_MLP,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[float, float]:
    """Evaluates rung 2 through ``output_layer``, as BP does."""
    return _evaluate(
        model, data_loader, criterion, device, input_adapter, READOUT_OUTPUT_LAYER
    )


def train_mf_joint_model(
    model: MF_MLP,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict[str, Any],
    device: torch.device,
    wandb_run: "Optional[wandb.sdk.wandb_run.Run]" = None,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    step_ref: Optional[List[int]] = None,
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float:
    """Rung 3: MF's objective and readout, but joint gradients and no detachment."""
    return train_deep_supervised_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        readout=READOUT_PROJECTION,
        default_optimizer="Adam",
        log_prefix="MF_Joint",
        wandb_run=wandb_run,
        input_adapter=input_adapter,
        step_ref=step_ref,
        gpu_handle=gpu_handle,
        nvml_active=nvml_active,
    )


def evaluate_mf_joint_model(
    model: MF_MLP,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[float, float]:
    """Evaluates rung 3 through M_L, as MF does."""
    return _evaluate(
        model, data_loader, criterion, device, input_adapter, READOUT_PROJECTION
    )
