"""Shared training-loop scaffolding (WP9, incremental extraction).

The first extracted primitive is optimizer construction, duplicated across
the bp/ff/cafo/mf trainers as ``getattr(optim, <name>)(...)``. Later
increments migrate epoch iteration, tqdm handling, LR-schedule callbacks,
and peak-memory sampling behind :class:`EpochContext` / ``run_epochs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import optim


def build_optimizer(
    spec_name: str,
    params: Any,
    *,
    lr: float,
    weight_decay: float = 0.0,
    extra_kwargs: dict[str, Any] | None = None,
) -> optim.Optimizer:
    """Resolve an optimizer from its configuration string in one place.

    ``spec_name`` is a torch.optim class name (e.g. ``"Adam"``, ``"SGD"``).
    Raises ``ValueError`` for unknown names instead of an opaque AttributeError.
    """

    cls = getattr(optim, spec_name, None)
    if not (isinstance(cls, type) and issubclass(cls, optim.Optimizer)):
        raise ValueError(f"Unknown optimizer type: {spec_name!r}")
    kwargs: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
    kwargs.update(extra_kwargs or {})
    return cls(params, **kwargs)


@dataclass
class EpochContext:
    """Per-run loop state shared by trainer epoch loops."""

    step_ref: list[int]
    log_interval: int
    wandb_run: Any | None = None


__all__ = ["EpochContext", "build_optimizer"]
