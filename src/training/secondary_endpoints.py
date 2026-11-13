"""Post-training secondary endpoints for W1 (PREREG-phase2.md §9, 2026-10-22).

These endpoints are deliberately evaluated **after** training and after the NVML
energy window has closed.  They are diagnostic: they describe what the final,
frozen network contains, but they never participate in W1's primary decision
rules and they never influence an optimiser step.

The fixed probe set is the validation split (5,000 samples for the canonical
CIFAR protocol, no random augmentation), capped at 5,000.  A ridge linear probe
is fitted on those frozen features and scored on the first 5,000 test samples.
The same routine is used for every arm: BP, MF-Joint, MF-Simul, MF-Seq,
Seq-Global and MF-Seq-nom0.  Effective rank is the entropy effective rank of the
centred validation features, ``exp(-sum p_i log p_i)`` with ``p_i`` the
normalised singular values.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.algorithms import get_evaluation_function
from src.architectures.mf_mlp import MF_MLP
from src.data_utils.preprocessing import get_transforms
from src.utils.logging_utils import logger

PROBE_MAX_SAMPLES = 5_000
PROBE_RIDGE_LAMBDA = 1e-2
PROBE_BATCH_SIZE = 512
ACTIVATION_TYPES = (
    nn.ReLU,
    nn.Tanh,
    nn.GELU,
    nn.SiLU,
    nn.LeakyReLU,
    nn.ELU,
)


def _flatten_features(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1)


def _sequential_activations(
    model: nn.Module, inputs: torch.Tensor
) -> List[torch.Tensor]:
    """Return post-activation hidden tensors for an ``nn.Sequential`` BP model."""
    activations: List[torch.Tensor] = []
    current = inputs
    for module in model:
        current = module(current)
        if isinstance(module, ACTIVATION_TYPES):
            activations.append(current)
    return activations


def _forward_activations(
    model: nn.Module, inputs: torch.Tensor, include_input: bool
) -> List[torch.Tensor]:
    if isinstance(model, MF_MLP):
        activations = model.forward_with_intermediate_activations(inputs)
        return activations if include_input else activations[1:]
    if isinstance(model, nn.Sequential):
        return _sequential_activations(model, inputs)
    raise TypeError(
        "Secondary endpoints support MF_MLP and nn.Sequential BP models; "
        f"got {type(model).__name__}."
    )


def _collect_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]],
    max_samples: int,
    include_input: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Collect frozen activations and labels from at most ``max_samples`` rows."""
    model.eval()
    per_layer: List[List[torch.Tensor]] = []
    labels: List[torch.Tensor] = []
    seen = 0
    with torch.no_grad():
        for images, targets in loader:
            if seen >= max_samples:
                break
            remaining = max_samples - seen
            if remaining < images.shape[0]:
                images = images[:remaining]
                targets = targets[:remaining]
            images = images.to(device)
            adapted = input_adapter(images) if input_adapter else images
            activations = _forward_activations(model, adapted, include_input)
            if not per_layer:
                per_layer = [[] for _ in activations]
            if len(activations) != len(per_layer):
                raise RuntimeError(
                    "Activation count changed between batches: "
                    f"{len(per_layer)} vs {len(activations)}."
                )
            for idx, activation in enumerate(activations):
                per_layer[idx].append(_flatten_features(activation).detach().float())
            labels.append(targets.detach().to("cpu"))
            seen += images.shape[0]
    if not per_layer:
        raise RuntimeError("No activations were collected.")
    features = [torch.cat(layer, dim=0) for layer in per_layer]
    return features, torch.cat(labels, dim=0)


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    algo_name: str,
    device: torch.device,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]],
    config: Dict[str, Any],
) -> Tuple[float, float]:
    evaluation_fn = get_evaluation_function(algo_name)
    criterion = nn.CrossEntropyLoss()
    if algo_name == "ff":
        output = evaluation_fn(model=model, data_loader=loader, device=device)
    else:
        output = evaluation_fn(
            model=model,
            data_loader=loader,
            device=device,
            criterion=criterion,
            input_adapter=input_adapter,
        )
        if algo_name == "cafo":
            # CaFo's evaluation signature has no aggregation argument in the
            # current engine; keep this branch explicit for future changes.
            pass
    if isinstance(output, dict):
        return (
            float(output.get("eval_loss", float("nan"))),
            float(output.get("eval_accuracy", float("nan"))),
        )
    if isinstance(output, tuple) and len(output) == 2:
        return float(output[0]), float(output[1])
    raise TypeError(f"Unsupported evaluation output for {algo_name}: {type(output)}")


@contextmanager
def _deterministic_train_loader(
    train_loader: DataLoader, config: Dict[str, Any]
) -> Iterator[DataLoader]:
    """Temporarily replace random train augmentation with the eval transform.

    The underlying dataset is mutated only after training has finished and is
    restored in ``finally``.  A fresh single-process loader avoids worker copies
    seeing different transforms.
    """
    dataset = getattr(train_loader, "dataset", None)
    sentinel = object()
    old_transform = getattr(dataset, "transform", sentinel)
    if old_transform is sentinel:
        yield train_loader
        return
    dataset_name = config.get("data", {}).get("name", "")
    try:
        eval_transform = get_transforms(dataset_name, train=False)
    except ValueError:
        yield train_loader
        return
    dataset.transform = eval_transform
    try:
        yield DataLoader(
            dataset,
            batch_size=PROBE_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
    finally:
        dataset.transform = old_transform


def _effective_rank(features: torch.Tensor) -> float:
    x = features.float()
    x = x - x.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(x)
    singular_values = singular_values[singular_values > 0]
    if singular_values.numel() == 0:
        return float("nan")
    probabilities = singular_values / singular_values.sum()
    entropy = -(probabilities * torch.log(probabilities)).sum()
    return float(torch.exp(entropy).item())


def _linear_probe_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
) -> float:
    x_train = train_features.float()
    x_test = test_features.float()
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    x_train = torch.cat(
        [x_train, torch.ones(x_train.shape[0], 1, device=x_train.device)], dim=1
    )
    x_test = torch.cat(
        [x_test, torch.ones(x_test.shape[0], 1, device=x_test.device)], dim=1
    )
    targets = F.one_hot(train_labels.to(x_train.device), num_classes=num_classes).float()
    gram = x_train.transpose(0, 1) @ x_train
    gram.diagonal().add_(PROBE_RIDGE_LAMBDA)
    rhs = x_train.transpose(0, 1) @ targets
    weights = torch.linalg.solve(gram, rhs)
    predictions = (x_test @ weights).argmax(dim=1).to("cpu")
    return float((predictions == test_labels).float().mean().item() * 100.0)


def _local_head_accuracies(
    model: nn.Module,
    all_test_activations: Sequence[torch.Tensor],
    test_labels: torch.Tensor,
) -> Dict[str, float]:
    if not isinstance(model, MF_MLP):
        return {}
    accuracies: Dict[str, float] = {}
    for index, matrix in enumerate(model.projection_matrices):
        if index >= len(all_test_activations):
            break
        logits = all_test_activations[index] @ matrix.detach().t().float()
        predictions = logits.argmax(dim=1).to("cpu")
        accuracies[f"M_{index}"] = float(
            (predictions == test_labels).float().mean().item() * 100.0
        )
    return accuracies


def compute_secondary_endpoints(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    algo_name: str,
    input_adapter: Optional[Callable[[torch.Tensor], torch.Tensor]],
    test_loss: Optional[float] = None,
    test_accuracy: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute diagnostic endpoints without touching model parameters.

    Returns a JSON-safe dictionary.  All probe and rank work happens *after*
    ``GPUEnergyMonitor.stop()`` (the engine calls this after the test pass), so
    it is outside the NVML energy window.  The elapsed seconds are reported so
    the overhead is auditable.
    """
    started = time.perf_counter()
    num_classes = int(config.get("data", {}).get("num_classes", 10))
    result: Dict[str, Any] = {
        "secondary_eval_scope": "post_training; outside NVML energy window",
        "secondary_eval_probe_max_samples": PROBE_MAX_SAMPLES,
        "secondary_eval_probe_ridge_lambda": PROBE_RIDGE_LAMBDA,
        "secondary_eval_probe_source": "validation split; first 5,000 rows",
        "secondary_eval_probe_test_source": "test split; first 5,000 rows",
        "secondary_eval_effective_rank_definition": (
            "exp(entropy(centred activation singular values))"
        ),
        "secondary_eval_test_loss_from_primary_pass": test_loss,
        "secondary_eval_test_accuracy_from_primary_pass": test_accuracy,
    }
    if val_loader is None:
        result["secondary_eval_error"] = "no validation loader; probes skipped"
        result["secondary_eval_seconds"] = time.perf_counter() - started
        return result

    # Full train/validation pass.  The train pass uses the eval transform so
    # random crop/flip cannot make this endpoint seed-dependent.
    with _deterministic_train_loader(train_loader, config) as deterministic_loader:
        train_loss, train_accuracy = _evaluate_loader(
            model, deterministic_loader, algo_name, device, input_adapter, config
        )
    val_loss, val_accuracy = _evaluate_loader(
        model, val_loader, algo_name, device, input_adapter, config
    )
    result.update(
        {
            "final_train_loss": train_loss,
            "final_train_accuracy": train_accuracy,
            "final_val_loss": val_loss,
            "final_val_accuracy": val_accuracy,
            "final_test_loss": test_loss,
            "final_test_accuracy": test_accuracy,
        }
    )

    # Frozen features on the fixed probe set and the first 5,000 test rows.
    probe_features, probe_labels = _collect_features(
        model, val_loader, device, input_adapter, PROBE_MAX_SAMPLES
    )
    test_features, test_labels = _collect_features(
        model, test_loader, device, input_adapter, PROBE_MAX_SAMPLES
    )
    all_test_features, all_test_labels = _collect_features(
        model,
        test_loader,
        device,
        input_adapter,
        PROBE_MAX_SAMPLES,
        include_input=True,
    )
    if len(probe_features) != len(test_features):
        raise RuntimeError("Probe and test activation counts differ.")
    if len(probe_features) == 0:
        raise RuntimeError("No hidden activations found for probe endpoints.")

    linear_probe: Dict[str, float] = {}
    effective_rank: Dict[str, float] = {}
    for index, (probe, test) in enumerate(zip(probe_features, test_features), start=1):
        linear_probe[f"a_{index}"] = _linear_probe_accuracy(
            probe, probe_labels, test, test_labels, num_classes
        )
        effective_rank[f"a_{index}"] = _effective_rank(probe)

    result["linear_probe_test_accuracy"] = linear_probe
    result["effective_rank"] = effective_rank
    result["local_head_test_accuracy"] = _local_head_accuracies(
        model, all_test_features, all_test_labels
    )
    result["secondary_eval_probe_n_train"] = int(probe_labels.numel())
    result["secondary_eval_probe_n_test"] = int(test_labels.numel())
    result["secondary_eval_seconds"] = time.perf_counter() - started
    logger.info(
        "Secondary endpoints (post-training, outside energy window): "
        "train acc %.2f%%, val loss %.4f, probe layers %d, %.2f s.",
        train_accuracy,
        val_loss,
        len(linear_probe),
        result["secondary_eval_seconds"],
    )
    return result
