"""Pure CaFo predictor loss and aggregation functions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812 - PyTorch convention


def predictor_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate the local predictor cross-entropy loss."""
    return F.cross_entropy(logits, labels)


def aggregate_predictor_outputs(
    predictor_outputs: Sequence[torch.Tensor], method: str = "sum"
) -> torch.Tensor:
    """Aggregate CaFo predictor logits using the configured inference rule."""
    if not predictor_outputs:
        raise ValueError("predictor_outputs must not be empty")
    normalized_method = method.lower()
    if normalized_method == "last":
        return predictor_outputs[-1]
    stacked = torch.stack(tuple(predictor_outputs), dim=0)
    if normalized_method == "sum":
        return stacked.sum(dim=0)
    if normalized_method == "average":
        return stacked.mean(dim=0)
    raise ValueError(f"Unsupported aggregation method: {method}")


__all__ = ["aggregate_predictor_outputs", "predictor_cross_entropy"]
