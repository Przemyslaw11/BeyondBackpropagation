"""Pure Forward-Forward math shared by training and evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def generate_hinton_inputs(
    base_images: torch.Tensor,
    base_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    replace_value_on: float = 1.0,
    replace_value_off: float = 0.0,
    neutral_value: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create positive, negative, and neutral label-embedded FF inputs."""
    batch_size = base_images.shape[0]
    base_flat_view = base_images.view(batch_size, -1)
    if num_classes > base_flat_view.shape[1]:
        raise ValueError(
            f"num_classes ({num_classes}) > total pixels ({base_flat_view.shape[1]}). "
            "Cannot embed label."
        )

    def embed(labels: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(labels, num_classes=num_classes).to(
            device=device, dtype=torch.float
        )
        return torch.where(one_hot == 1, replace_value_on, replace_value_off)

    positive = base_flat_view.clone()
    positive[:, :num_classes] = embed(base_labels)

    offsets = torch.randint(
        1, num_classes, (batch_size,), device=device, dtype=torch.long
    )
    negative_labels = (base_labels + offsets) % num_classes
    collision = negative_labels == base_labels
    retries = 0
    while torch.any(collision) and retries < 5:
        replacement = torch.randint(
            1,
            num_classes,
            (int(collision.sum().item()),),
            device=device,
            dtype=torch.long,
        )
        negative_labels[collision] = (
            base_labels[collision] + replacement
        ) % num_classes
        collision = negative_labels == base_labels
        retries += 1
    if torch.any(collision):
        negative_labels[collision] = (negative_labels[collision] + 1) % num_classes

    negative = base_flat_view.clone()
    negative[:, :num_classes] = embed(negative_labels)
    neutral = base_flat_view.clone()
    neutral[:, :num_classes] = torch.full(
        (batch_size, num_classes),
        neutral_value,
        device=device,
        dtype=torch.float,
    )
    return positive.detach(), negative.detach(), neutral.detach()


def linear_cooldown_lr(initial_lr: float, epoch: int, total_epochs: int) -> float:
    """Apply the repository's second-half linear learning-rate cooldown."""
    current_epoch = epoch + 1
    if current_epoch <= total_epochs // 2:
        return initial_lr
    factor = 2.0 * (1 + total_epochs - current_epoch) / float(total_epochs)
    return max(initial_lr * factor, 1e-9)


def aggregate_goodness(
    layer_goodness: list[torch.Tensor], *, start_layer: int = 1
) -> torch.Tensor:
    """Sum per-layer goodness using the historical FF inference convention."""
    selected = layer_goodness[start_layer:]
    if not selected:
        if not layer_goodness:
            raise ValueError("At least one goodness tensor is required")
        selected = layer_goodness
    return torch.stack(selected, dim=0).sum(dim=0)


__all__ = ["aggregate_goodness", "generate_hinton_inputs", "linear_cooldown_lr"]
