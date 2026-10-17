# src/algorithms/mf_global.py
"""Seq-Global: the MF-Seq schedule with backprop through the trainable prefix.

Stage s of Seq-Global trains ``W_1..W_s`` jointly with ``M_s`` on ``M_s``'s local
cross-entropy, exactly like MF-Seq stages and with the same early stopping, but the
earlier layers are *unfrozen*: gradients flow from the local loss back through the
whole trainable prefix instead of stopping at the current layer's input. Frozen
layers still stop the graph, so at stage ``s`` the parameters that move are exactly
``{W_1..W_s, M_s}``.

If MF-Seq's advantage over plain BP were "backprop is local per stage", Seq-Global
should keep the advantage (same stopping schedule, same local losses, same readout);
if it were "each stage sees a frozen prefix", Seq-Global should lose it. That makes
it the schedule-vs-locality separator in the attribution ladder, and the reason it
lives beside ``mf.py`` rather than inside a flag branch of it.

Deviation table vs ``train_mf_model`` (everything else is shared code):

| aspect            | MF-Seq                       | Seq-Global                    |
|-------------------|------------------------------|-------------------------------|
| stage input       | frozen prefix a_i (no grad)  | live prefix a_s (grad)        |
| trainable stage s | W_s, M_s                     | W_1..W_s, M_s                 |
| activation cache  | recompute/device/host        | recompute only (prefix moves) |
| M0 stage          | optional (``mf_train_m0``)   | same option, same code path   |
| evaluation        | ``evaluate_mf_model`` (a_L, M_L) | identical                 |
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import pynvml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.architectures.mf_mlp import MF_MLP
from src.utils.early_stopping import resolve_early_stopping
from src.utils.helpers import (
    create_directory_if_not_exists,
    save_checkpoint,
)
from src.utils.logging_utils import log_metrics
from src.utils.monitoring import get_gpu_memory_usage

from .mf import evaluate_mf_local_loss, mf_local_loss_fn, train_mf_matrix_only

if TYPE_CHECKING:
    import wandb.sdk.wandb_run


logger = logging.getLogger(__name__)


def train_mf_global_model(
    model: MF_MLP,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    input_adapter: Callable[[torch.Tensor], torch.Tensor],
    val_loader: Optional[DataLoader] = None,
    wandb_run: "Optional[wandb.sdk.wandb_run.Run]" = None,
    step_ref: Optional[List[int]] = None,
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> float:
    """Trains an MF_MLP with the Seq-Global schedule (unfrozen trainable prefix)."""
    if step_ref is None:
        step_ref = [-1]
    model.to(device)
    num_w_layers = model.num_hidden_layers
    num_m_matrices = len(model.projection_matrices)

    logger.info(
        f"Starting Seq-Global training for MLP with {num_w_layers} W-layers "
        f"and {num_m_matrices} M-matrices."
    )

    algo_config = config.get("algorithm_params", config.get("training", {}))
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    lr = algo_config.get("lr", 0.001)
    weight_decay = algo_config.get("weight_decay", 0.0)
    es_policy = resolve_early_stopping(config)
    epochs_per_layer = es_policy["max_epochs"]
    log_interval = algo_config.get("log_interval", 100)
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None)
    mf_criterion = nn.CrossEntropyLoss()

    cache_strategy = str(algo_config.get("activation_cache", "recompute")).lower()
    if cache_strategy != "recompute":
        raise ValueError(
            "Seq-Global cannot use an activation cache: the trainable prefix "
            "changes every step, so cached activations would be stale. "
            f"Got activation_cache={cache_strategy!r}."
        )

    es_enabled = es_policy["enabled"]
    es_patience = es_policy["patience"]
    es_min_delta = es_policy["min_delta"]
    mf_early_stopping_config = {
        "enabled": es_enabled,
        "patience": es_patience,
        "min_delta": es_min_delta,
    }

    train_m0 = bool(algo_config.get("mf_train_m0", True))
    if not train_m0:
        logger.info("Skipping the M0 stage; training W1..WL against M1..ML only.")

    peak_mem_train = 0.0
    total_epochs_trained_all_layers = 0

    logger.debug("Freezing all model parameters initially.")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # --- Phase 1: M0 (dead compute; kept for schedule parity with MF-Seq) ---
    if train_m0 and num_m_matrices > 0:
        m0_params = [model.get_projection_matrix(0)]
        model.get_projection_matrix(0).requires_grad_(True)
        m0_optimizer = getattr(optim, optimizer_name)(
            m0_params, lr=lr, weight_decay=weight_decay
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
            gpu_handle=gpu_handle,
            nvml_active=nvml_active,
        )
        total_epochs_trained_all_layers += epochs_trained_m0
        peak_mem_train = max(peak_mem_train, m0_peak_mem)
        model.get_projection_matrix(0).requires_grad_(False)
        if checkpoint_dir:
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained_index": -1},
                is_best=False,
                filename="mf_global_matrix_M0_complete.pth",
                checkpoint_dir=checkpoint_dir,
            )

    # --- Phase 2: stage s trains W_1..W_s and M_s on M_s's local loss ---
    for s in range(1, num_w_layers + 1):
        log_prefix = f"Layer_W{s}_M{s}"
        logger.info(f"--- Starting Seq-Global training for {log_prefix} ---")

        params_to_optimize = []
        for k in range(s):  # W_1 .. W_s
            linear_layer = model.layers[k * 2]
            for p in linear_layer.parameters():
                p.requires_grad_(True)
            params_to_optimize.extend(list(linear_layer.parameters()))

        projection_matrix = model.get_projection_matrix(s)
        projection_matrix.requires_grad_(True)
        params_to_optimize.append(projection_matrix)

        optimizer = getattr(optim, optimizer_name)(
            params_to_optimize, lr=lr, weight_decay=weight_decay
        )

        peak_mem_layer_train = 0.0
        epochs_trained_this_layer = 0
        epochs_no_improve = 0
        best_es_val_loss = float("inf")

        for epoch in range(epochs_per_layer):
            epochs_trained_this_layer = epoch + 1
            epoch_loss, epoch_samples = 0.0, 0
            peak_mem_layer_epoch = 0.0

            for k in range(s):
                model.layers[k * 2].train()
                model.layers[k * 2 + 1].train()
            projection_matrix.requires_grad_(True)

            pbar_desc = f"{log_prefix} Epoch {epoch + 1}/{epochs_per_layer}"
            pbar = tqdm(train_loader, desc=pbar_desc, leave=False)

            loss = None
            for batch_idx, (images, labels) in enumerate(pbar):
                step_ref[0] += 1
                current_global_step = step_ref[0]
                images, labels = images.to(device), labels.to(device)

                # Live forward through the trainable prefix: the graph reaches
                # every unfrozen W_k and stops at the frozen ones.
                activations = model.forward_with_intermediate_activations(
                    input_adapter(images)
                )
                activation_a_s = activations[s]

                loss = mf_local_loss_fn(
                    activation_a_s, projection_matrix, labels, mf_criterion
                )
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(
                        f"NaN/Inf loss at {log_prefix}, Epoch {epoch + 1}, "
                        f"Batch {batch_idx}."
                    )
                    break

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)
                epoch_samples += images.size(0)

                is_log_time = (batch_idx + 1) % log_interval == 0 or (
                    batch_idx == len(train_loader) - 1
                )
                if nvml_active and gpu_handle and is_log_time:
                    mem_info = get_gpu_memory_usage(gpu_handle)
                    if mem_info:
                        peak_mem_layer_epoch = max(peak_mem_layer_epoch, mem_info[0])

                if is_log_time:
                    metrics = {"global_step": current_global_step}
                    metrics[f"{log_prefix}/Train_Loss_Batch"] = loss.item()
                    log_metrics(metrics, wandb_run=wandb_run, commit=True)
                    pbar.set_postfix(loss=f"{loss.item():.6f}")

            if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
                logger.error(f"Terminating {log_prefix} training due to invalid loss.")
                break

            final_avg_epoch_loss = (
                epoch_loss / epoch_samples if epoch_samples > 0 else float("nan")
            )
            peak_mem_layer_train = max(peak_mem_layer_train, peak_mem_layer_epoch)
            logger.info(
                f"{log_prefix} Epoch {epoch + 1}/{epochs_per_layer} - Train Loss: "
                f"{final_avg_epoch_loss:.6f}"
            )
            log_metrics(
                {
                    "global_step": step_ref[0],
                    f"{log_prefix}/Train_Loss_EpochAvg": final_avg_epoch_loss,
                },
                wandb_run=wandb_run,
                commit=True,
            )

            if es_enabled and val_loader is not None:
                val_loss = evaluate_mf_local_loss(
                    model, s, mf_criterion, val_loader, device, input_adapter
                )
                logger.info(
                    f"{log_prefix} Epoch {epoch + 1}/{epochs_per_layer} - "
                    f"Val Local Loss: {val_loss:.6f}"
                )
                log_metrics(
                    {
                        "global_step": step_ref[0],
                        f"{log_prefix}/Val_LocalLoss_Epoch": val_loss,
                    },
                    wandb_run=wandb_run,
                    commit=True,
                )
                if torch.isnan(torch.tensor(val_loss)):
                    epochs_no_improve += 1
                elif val_loss < best_es_val_loss - es_min_delta:
                    best_es_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= es_patience:
                    logger.info(
                        f"--- {log_prefix}: Early Stopping at Epoch {epoch + 1}! ---"
                    )
                    break

        total_epochs_trained_all_layers += epochs_trained_this_layer
        peak_mem_train = max(peak_mem_train, peak_mem_layer_train)
        for p in params_to_optimize:
            p.requires_grad_(False)
        model.eval()

        if checkpoint_dir:
            create_directory_if_not_exists(checkpoint_dir)
            save_checkpoint(
                state={"state_dict": model.state_dict(), "layer_trained_index": s - 1},
                is_best=False,
                filename=f"mf_global_layer_{s}_complete.pth",
                checkpoint_dir=checkpoint_dir,
            )

    config.setdefault("_run_stats", {})[
        "epochs_completed"
    ] = total_epochs_trained_all_layers
    logger.info(
        f"Finished all Seq-Global stages. Total Epochs (Sum): "
        f"{total_epochs_trained_all_layers}"
    )
    model.eval()
    return peak_mem_train
