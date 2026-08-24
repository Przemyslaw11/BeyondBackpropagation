"""Pure Mono-Forward projection and local-loss functions."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def projection_logits(
    activation: torch.Tensor, projection_matrix: torch.Tensor
) -> torch.Tensor:
    """Project one detached activation into class goodness logits."""
    if activation.dim() != 2:
        raise ValueError(
            f"Activation must be flattened (2D) for local loss. Got shape: {activation.shape}"
        )
    return torch.matmul(activation, projection_matrix.t())


def local_cross_entropy(
    activation: torch.Tensor,
    projection_matrix: torch.Tensor,
    targets: torch.Tensor,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Calculate MF local cross-entropy from activation and projection matrix."""
    loss_fn = criterion or nn.CrossEntropyLoss()
    return loss_fn(projection_logits(activation, projection_matrix), targets)


__all__ = ["local_cross_entropy", "projection_logits"]
