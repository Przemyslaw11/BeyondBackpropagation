# src/algorithms/mf.py
"""Implementation of the Mono-Forward (MF) algorithm for training MLPs."""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import pynvml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

from src.architectures.mf_mlp import MF_MLP
from src.utils.early_stopping import resolve_early_stopping
from src.utils.helpers import (
    create_directory_if_not_exists,
    save_checkpoint,
)
from src.utils.logging_utils import log_metrics
from src.utils.monitoring import get_gpu_memory_usage

if TYPE_CHECKING:
    import wandb.sdk.wandb_run


logger = logging.getLogger(__name__)

ACTIVATION_CACHE_STRATEGIES = ("recompute", "cache_device", "cache_host")

# Caching a_i across epochs is only sound where the frozen prefix makes a_i a pure
# function of the sample, which needs layer-sequential training AND no input randomness.
LAYER_SEQUENTIAL_ALGORITHMS = frozenset({"mf"})


class _IndexedDataset(Dataset):
    """Wraps a dataset so every item carries its index, keying the activation cache."""

    def __init__(self, base: Dataset) -> None:
        """Stores the wrapped dataset."""
        self.base = base

    def __len__(self) -> int:
        """Returns the length of the wrapped dataset."""
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        """Returns the wrapped item followed by its index."""
        image, label = self.base[index]
        return image, label, index


def _with_sample_indices(loader: DataLoader) -> DataLoader:
    """Rebuilds a loader over the same dataset so batches also yield sample indices.

    Sampler type and length are preserved, so the shuffling sequence drawn from the
    global RNG is identical to the original loader's.
    """
    return DataLoader(
        dataset=_IndexedDataset(loader.dataset),
        batch_size=loader.batch_size,
        shuffle=isinstance(loader.sampler, RandomSampler),
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        persistent_workers=loader.num_workers > 0,
    )


def _stochastic_transform_names(dataset: Dataset) -> List[str]:
    """Names the random transforms reachable from a dataset, if any."""
    transform = getattr(dataset, "transform", None)
    candidates = getattr(transform, "transforms", [transform] if transform else [])
    return [
        type(op).__name__
        for op in candidates
        if op is not None and type(op).__name__.lower().startswith("random")
    ]


def assert_activation_cache_allowed(
    config: Dict[str, Any], train_loader: DataLoader, strategy: str
) -> None:
    """Raises unless caching frozen activations is mathematically sound here."""
    if strategy not in ACTIVATION_CACHE_STRATEGIES:
        raise ValueError(
            f"Unknown activation_cache '{strategy}'. "
            f"Expected one of {list(ACTIVATION_CACHE_STRATEGIES)}."
        )

    algorithm = config.get("algorithm", {}).get("name", "").lower()
    if algorithm not in LAYER_SEQUENTIAL_ALGORITHMS:
        raise ValueError(
            f"activation_cache='{strategy}' requires a layer-sequential algorithm "
            f"that detaches its inputs; '{algorithm or 'unknown'}' propagates joint "
            "gradients, so cached activations would go stale."
        )

    stochastic = _stochastic_transform_names(train_loader.dataset)
    if stochastic:
        raise ValueError(
            f"activation_cache='{strategy}' requires a deterministic input pipeline; "
            f"the training transform applies {stochastic}, so a_i differs per epoch."
        )


class _ActivationCache:
    """Holds a_i for every training sample so the frozen prefix runs once per layer."""

    def __init__(self, num_samples: int, storage: torch.device, device: torch.device):
        """Allocates lazily; ``storage`` is where activations live between epochs."""
        self._storage = storage
        self._device = device
        self._buffer: Optional[torch.Tensor] = None
        self._filled = torch.zeros(num_samples, dtype=torch.bool)

    def get(self, indices: torch.Tensor) -> Optional[torch.Tensor]:
        """Returns the cached batch, or None if any sample is still missing."""
        if self._buffer is None or not bool(self._filled[indices].all()):
            return None
        # Blocking transfers only: indexing yields a pageable temporary either way, so an
        # async copy would buy nothing and could be read before the device had written it.
        return self._buffer[indices].to(self._device)

    def put(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        """Stores a freshly computed batch of activations."""
        if self._buffer is None:
            self._buffer = torch.empty(
                (self._filled.numel(), *values.shape[1:]),
                dtype=values.dtype,
                device=self._storage,
            )
        self._buffer[indices] = values.to(self._storage)
        self._filled[indices] = True

    def release(self) -> None:
        """Drops the buffer so the next layer starts from an empty cache."""
        self._buffer = None
        self._filled.fill_(False)


def _make_activation_cache(
    strategy: str, num_samples: int, device: torch.device
) -> Optional[_ActivationCache]:
    """Builds the cache for a strategy, or None when activations are recomputed."""
    if strategy == "cache_device":
        return _ActivationCache(num_samples, device, device)
    if strategy == "cache_host":
        return _ActivationCache(num_samples, torch.device("cpu"), device)
    return None


def mf_local_loss_fn(
    activation_i: torch.Tensor,
    projection_matrix_i: nn.Parameter,
    targets: torch.Tensor,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Calculates the MF local cross-entropy loss for activation a_i using M_i."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if activation_i.dim() != 2:
        raise ValueError(
            f"Activation must be flattened (2D) for local loss. Got shape: {activation_i.shape}"
        )
    goodness_scores_i = torch.matmul(activation_i, projection_matrix_i.t())
    loss = criterion(goodness_scores_i, targets)
    return loss


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
    early_stopping_config: Dict[str, Any],
    val_loader: Optional[DataLoader] = None,
    wandb_run: "Optional[wandb.sdk.wandb_run.Run]" = None,
    log_interval: int = 100,
    step_ref: Optional[List[int]] = None,
    gpu_handle: Optional[pynvml.c_nvmlDevice_t] = None,
    nvml_active: bool = False,
) -> Tuple[float, float, int]:
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

    peak_mem_matrix_train = 0.0
    final_avg_epoch_loss = float("nan")
    epochs_trained = 0

    es_enabled = early_stopping_config.get("enabled", False)
    if es_enabled and val_loader is None:
        logger.warning(f"{log_prefix}: ES enabled but no val_loader. Disabling.")
        es_enabled = False

    if es_enabled:
        es_patience = early_stopping_config.get("patience", 10)
        es_min_delta = early_stopping_config.get("min_delta", 0.0)
        epochs_no_improve = 0
        best_es_metric_value = float("inf")
        logger.info(
            f"{log_prefix}: Early Stopping Enabled - Patience: {es_patience}, "
            f"MinDelta: {es_min_delta}"
        )
    else:
        logger.info(f"{log_prefix}: Early Stopping Disabled.")

    for epoch in range(epochs):
        epochs_trained = epoch + 1
        epoch_loss, epoch_samples = 0.0, 0
        peak_mem_matrix_epoch = 0.0
        projection_matrix.requires_grad_(True)

        pbar_desc = f"{log_prefix} Epoch {epoch + 1}/{epochs}"
        pbar = tqdm(train_loader, desc=pbar_desc, leave=False)

        for batch_idx, (images, labels) in enumerate(pbar):
            step_ref[0] += 1
            current_global_step = step_ref[0]
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                adapted_input = input_adapter(images)
                all_activations = model.forward_with_intermediate_activations(
                    adapted_input
                )
                if len(all_activations) <= matrix_index:
                    logger.error(f"{log_prefix} Batch {batch_idx}: Act list too short.")
                    continue
                activation_a_i = all_activations[matrix_index]

            loss = mf_local_loss_fn(
                activation_a_i, projection_matrix, labels, criterion
            )
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"NaN/Inf loss at {log_prefix}, Epoch {epoch + 1}, Batch {batch_idx}."
                )
                break  # Break from batch loop

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size

            is_log_time = (batch_idx + 1) % log_interval == 0 or (
                batch_idx == len(train_loader) - 1
            )
            current_mem_used = float("nan")
            if nvml_active and gpu_handle and is_log_time:
                mem_info = get_gpu_memory_usage(gpu_handle)
                if mem_info:
                    current_mem_used = mem_info[0]
                    peak_mem_matrix_epoch = max(peak_mem_matrix_epoch, current_mem_used)

            if is_log_time:
                metrics = {"global_step": current_global_step}
                metrics[f"{log_prefix}/Train_Loss_Batch"] = loss.item()
                if not torch.isnan(torch.tensor(current_mem_used)):
                    metrics[f"{log_prefix}/GPU_Mem_Used_MiB_Batch"] = current_mem_used
                log_metrics(metrics, wandb_run=wandb_run, commit=True)
                pbar.set_postfix(loss=f"{loss.item():.6f}")

        if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)):
            logger.error(f"Terminating {log_prefix} training due to invalid loss.")
            break  # Break from epoch loop

        final_avg_epoch_loss = (
            epoch_loss / epoch_samples if epoch_samples > 0 else float("nan")
        )
        peak_mem_matrix_train = max(peak_mem_matrix_train, peak_mem_matrix_epoch)
        logger.info(
            f"{log_prefix} Epoch {epoch + 1}/{epochs} - Train Loss: "
            f"{final_avg_epoch_loss:.6f}, Peak Mem: {peak_mem_matrix_epoch:.1f} MiB"
        )
        epoch_metrics = {
            "global_step": step_ref[0],
            f"{log_prefix}/Train_Loss_EpochAvg": final_avg_epoch_loss,
            f"{log_prefix}/Peak_GPU_Mem_Epoch_MiB": peak_mem_matrix_epoch,
        }
        log_metrics(epoch_metrics, wandb_run=wandb_run, commit=True)

        if es_enabled and val_loader is not None:
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
            log_msg = f"{log_prefix} Epoch {epoch + 1}/{epochs} - Val Local Loss: {val_loss:.6f}"
            logger.info(log_msg)
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
            elif val_loss < best_es_metric_value - es_min_delta:
                best_es_metric_value = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= es_patience:
                logger.info(
                    f"--- {log_prefix}: Early Stopping at Epoch {epoch + 1}! ---"
                )
                break

    if nvml_active and gpu_handle:
        mem_info = get_gpu_memory_usage(gpu_handle)
        peak_mem_matrix_train = max(
            peak_mem_matrix_train, mem_info[0] if mem_info else 0.0
        )

    projection_matrix.requires_grad_(False)
    logger.info(
        f"--- Finished training for {log_prefix} after {epochs_trained} epochs. ---"
    )
    return final_avg_epoch_loss, peak_mem_matrix_train, epochs_trained


def train_mf_model(
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
    optimizer_name = algo_config.get("optimizer_type", "Adam")
    lr = algo_config.get("lr", 0.001)
    weight_decay = algo_config.get("weight_decay", 0.0)
    optimizer_extra_kwargs = {}
    es_policy = resolve_early_stopping(config)
    epochs_per_layer = es_policy["max_epochs"]
    log_interval = algo_config.get("log_interval", 100)
    checkpoint_dir = config.get("checkpointing", {}).get("checkpoint_dir", None)
    mf_criterion = nn.CrossEntropyLoss()

    # MF stops each layer independently; the policy values are the shared ones.
    es_enabled = es_policy["enabled"]
    es_patience = es_policy["patience"]
    es_min_delta = es_policy["min_delta"]
    mf_early_stopping_config = {
        "enabled": es_enabled,
        "patience": es_patience,
        "min_delta": es_min_delta,
    }

    cache_strategy = str(algo_config.get("activation_cache", "recompute")).lower()
    layer_loader = train_loader
    if cache_strategy != "recompute":
        assert_activation_cache_allowed(config, train_loader, cache_strategy)
        layer_loader = _with_sample_indices(train_loader)

    # M0 postdates the published runs and feeds neither phase 2 nor evaluation.
    train_m0 = bool(algo_config.get("mf_train_m0", True))
    if not train_m0:
        logger.info("Skipping the M0 stage; training W1..WL against M1..ML only.")

    peak_mem_train = 0.0
    total_epochs_trained_all_layers = 0

    logger.debug("Freezing all model parameters initially.")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # --- Phase 1: Train M0 (on input a0) ---
    if train_m0 and num_m_matrices > 0:
        m0_params = [model.get_projection_matrix(0)]
        model.get_projection_matrix(0).requires_grad_(True)
        m0_optimizer = getattr(optim, optimizer_name)(
            m0_params, lr=lr, weight_decay=weight_decay, **optimizer_extra_kwargs
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

        optimizer = getattr(optim, optimizer_name)(
            params_to_optimize,
            lr=lr,
            weight_decay=weight_decay,
            **optimizer_extra_kwargs,
        )

        peak_mem_layer_train = 0.0
        epochs_trained_this_layer = 0
        epochs_no_improve = 0
        best_es_val_loss = float("inf")
        activation_cache = _make_activation_cache(
            cache_strategy, len(layer_loader.dataset), device
        )

        for epoch in range(epochs_per_layer):
            epochs_trained_this_layer = epoch + 1
            epoch_loss, epoch_samples = 0.0, 0
            peak_mem_layer_epoch = 0.0
            model.layers[i * 2].train()
            model.layers[i * 2 + 1].train()
            projection_matrix.requires_grad_(True)

            pbar_desc = f"{log_prefix} Epoch {epoch + 1}/{epochs_per_layer}"
            pbar = tqdm(layer_loader, desc=pbar_desc, leave=False)

            for batch_idx, batch in enumerate(pbar):
                step_ref[0] += 1
                if activation_cache is None:
                    images, labels = batch
                    indices = None
                else:
                    images, labels, indices = batch
                images, labels = images.to(device), labels.to(device)

                # Get input for the current layer W_i+1, which is activation a_i
                prev_activation = (
                    None if indices is None else activation_cache.get(indices)
                )
                if prev_activation is None:
                    with torch.no_grad():
                        prev_activation = input_adapter(images)
                        for k in range(i):  # Recompute forward pass up to layer i-1
                            temp_linear = model.layers[k * 2]
                            temp_act_fn = model.layers[k * 2 + 1]
                            prev_activation = temp_act_fn(temp_linear(prev_activation))
                    if activation_cache is not None:
                        activation_cache.put(indices, prev_activation)

                # Forward through W_i+1 to get a_i+1, with grads for W_i+1
                pre_act_z = model.layers[i * 2](prev_activation.detach())
                activation_a_next = model.layers[i * 2 + 1](pre_act_z)

                loss = mf_local_loss_fn(
                    activation_a_next, projection_matrix, labels, mf_criterion
                )
                if torch.isnan(loss) or torch.isinf(loss):
                    log_msg = f"NaN/Inf loss at {log_prefix}, Epoch {epoch + 1}, Batch {batch_idx}."
                    logger.error(log_msg)
                    break

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)
                epoch_samples += images.size(0)

                is_log_time = (batch_idx + 1) % log_interval == 0 or (
                    batch_idx == len(layer_loader) - 1
                )
                if nvml_active and gpu_handle and is_log_time:
                    mem_info = get_gpu_memory_usage(gpu_handle)
                    if mem_info:
                        peak_mem_layer_epoch = max(peak_mem_layer_epoch, mem_info[0])

            if "loss" in locals() and (torch.isnan(loss) or torch.isinf(loss)):
                break

            final_avg_epoch_loss = (
                epoch_loss / epoch_samples if epoch_samples > 0 else float("nan")
            )
            peak_mem_layer_train = max(peak_mem_layer_train, peak_mem_layer_epoch)
            # ... (epoch logging) ...

            if es_enabled and val_loader is not None:
                model.eval()  # Set all layers to eval for consistent validation
                val_loss = evaluate_mf_local_loss(
                    model, m_idx, mf_criterion, val_loader, device, input_adapter
                )
                # ... (early stopping logic as in train_mf_matrix_only) ...
                if val_loss < best_es_val_loss - es_min_delta:
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
        if activation_cache is not None:
            activation_cache.release()
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

    config.setdefault("_run_stats", {})[
        "epochs_completed"
    ] = total_epochs_trained_all_layers
    logger.info(
        f"Finished all layer-wise MF training. Total Epochs (Sum): "
        f"{total_epochs_trained_all_layers}"
    )
    model.eval()
    return peak_mem_train


@torch.no_grad()
def evaluate_mf_model(
    model: MF_MLP,
    data_loader: DataLoader,
    device: torch.device,
    input_adapter: Callable[[torch.Tensor], torch.Tensor],
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Evaluates the trained MF_MLP using the last activation and projection matrix.

    This uses the activation from the last layer (a_L) and the last projection
    matrix (M_L), where L is the number of hidden layers.
    """
    model.eval()
    model.to(device)
    total_correct, total_samples = 0, 0
    loss_sum = 0.0

    num_layers = model.num_hidden_layers
    last_activation_index = num_layers
    last_projection_matrix_index = num_layers

    logger.info(
        f"Evaluating MF (MLP) using activation a_{last_activation_index} and "
        f"matrix M_{last_projection_matrix_index}."
    )
    if last_projection_matrix_index >= len(model.projection_matrices):
        logger.error(
            f"Index M_{last_projection_matrix_index} out of bounds "
            f"({len(model.projection_matrices)} matrices)."
        )
        return {"eval_accuracy": float("nan"), "eval_loss": float("nan")}
    last_projection_matrix = model.get_projection_matrix(last_projection_matrix_index)

    pbar = tqdm(data_loader, desc="Evaluating MF MLP", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        eval_input = input_adapter(images)

        all_activations = model.forward_with_intermediate_activations(eval_input)

        if len(all_activations) <= last_activation_index:
            logger.error(
                f"Activation list len ({len(all_activations)}) too short "
                f"for a_{last_activation_index}."
            )
            continue

        last_activation = all_activations[last_activation_index].to(device)
        last_projection_matrix = last_projection_matrix.to(device)
        goodness_scores = torch.matmul(last_activation, last_projection_matrix.t())
        predicted_labels = torch.argmax(goodness_scores, dim=1)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)

        # Goodness scores are the only class-conditional score MF produces, and are
        # the same tensor the prediction argmax reads, so the reported test loss is
        # cross-entropy over them. This mirrors evaluate_ff_model.
        loss_sum += F.cross_entropy(goodness_scores, labels, reduction="sum").item()

    accuracy = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    eval_loss = loss_sum / total_samples if total_samples > 0 else float("nan")
    logger.info(
        f"MF Evaluation Results (MLP): Accuracy: {accuracy:.2f}%, Loss: {eval_loss:.4f}"
    )
    return {"eval_accuracy": accuracy, "eval_loss": eval_loss}
