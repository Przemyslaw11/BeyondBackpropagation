"""Shared training-loop scaffolding (WP9, incremental extraction).

The first extracted primitive was optimizer construction, duplicated across
the bp/ff/cafo/mf trainers as ``getattr(optim, <name>)(...)``. This increment
adds ``run_epochs``: a hook-based epoch-loop skeleton absorbing the duplicated
tqdm / batch / NaN-accounting / early-stopping scaffolding (MF migrated as the
pilot; CaFo and FF follow in later increments).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import optim
from tqdm import tqdm

from ..utils.training_support import log_metrics


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


BatchLossFn = Callable[[int, Any, Any], torch.Tensor | None]


@dataclass
class NanLossGuard:
    """RUN-004 accounting for NaN/Inf batch losses.

    Legacy semantics preserved verbatim: a NaN/Inf loss breaks out of the
    batch loop (legacy stop timing), and a *second* epoch aborting on
    NaN/Inf raises instead of looping forever. One guard per trainer scope —
    MF intentionally keeps separate counters for ``train_mf_matrix_only``
    and the layer-wise loop of ``train_mf_model``, matching the
    pre-extraction locals.
    """

    strikes: int = 0
    epoch_aborted: bool = False


@dataclass
class EpochLoopResult:
    """Outcome of one :func:`run_epochs` call."""

    epochs_trained: int
    final_avg_epoch_loss: float
    peak_mem: float


def run_epochs(
    ctx: EpochContext,
    train_loader: Iterable[tuple[Any, Any]],
    *,
    epochs: int,
    log_prefix: str,
    logger: logging.Logger,
    optimizer: optim.Optimizer,
    guard: NanLossGuard,
    batch_loss: BatchLossFn,
    on_epoch_start: Callable[[int], None] | None = None,
    on_epoch_summary: Callable[[int, float], None] | None = None,
    validate: Callable[[int], bool] | None = None,
    log_batch_loss: bool = False,
    abort_label: str = "MF",
) -> EpochLoopResult:
    """Drive the duplicated trainer epoch-loop scaffolding through one skeleton.

    Hooks:

    - ``batch_loss(batch_idx, inputs, targets)`` computes the loss for one
      batch; return ``None`` to skip the batch (e.g. malformed forward
      output). The skeleton owns ``zero_grad()/backward()/step()``.
    - ``on_epoch_start(epoch)`` per-epoch mode/grad wiring.
    - ``on_epoch_summary(epoch, avg_loss)`` end-of-epoch summary logging and
      metric emission; receives the completed epoch's average train loss.
    - ``validate(epoch)`` returns ``True`` to early-stop the loop;
      implementations own their :class:`~.early_stopping.EarlyStopping`
      update, so the D1 patience boundary stays caller-side.

    Behavior pinned by characterization tests:

    - tqdm progress bar with ``leave=False`` and desc
      ``"{log_prefix} Epoch {e+1}/{epochs}"``.
    - ``ctx.step_ref[0]`` increments exactly once per batch.
    - RUN-004 NaN/Inf accounting via ``guard`` (see :class:`NanLossGuard`).
      The abort message reads ``"{abort_label} training aborted on ..."``;
      keep it aligned with the pinned MF string when renaming.
    - every ``log_metrics`` dict carries ``global_step`` first and uses
      ``commit=True``.
    - ``epochs_trained`` counts *started* epochs (matches legacy counters).

    Pass the caller's module logger so log records keep their original
    emission source (caplog-based characterization pins this).
    """
    epochs_trained = 0
    final_avg_epoch_loss = float("nan")
    # ponytail: peak-mem sampling inside these loops never existed — the
    # legacy per-epoch value was computed but always 0.0. P2-cadence sampling
    # lands as an extra hook in a later WP9 increment; upgrade path is a
    # ``sample_peak_mem`` callable folded into ``EpochLoopResult.peak_mem``.
    peak_mem = 0.0

    for epoch in range(epochs):
        epochs_trained = epoch + 1
        if on_epoch_start is not None:
            on_epoch_start(epoch)

        epoch_loss, epoch_samples = 0.0, 0
        guard.epoch_aborted = False
        pbar = tqdm(
            train_loader,
            desc=f"{log_prefix} Epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        for batch_idx, (inputs, targets) in enumerate(pbar):
            ctx.step_ref[0] += 1
            current_global_step = ctx.step_ref[0]

            loss = batch_loss(batch_idx, inputs, targets)
            if loss is None:
                continue
            if torch.isnan(loss) or torch.isinf(loss):
                # RUN-004: legacy break semantics kept; second strike raises.
                guard.strikes += 1
                guard.epoch_aborted = True
                logger.error(
                    f"NaN/Inf loss at {log_prefix}, "
                    f"Epoch {epoch + 1}, Batch {batch_idx}."
                )
                if guard.strikes >= 2:
                    raise RuntimeError(
                        f"{abort_label} training aborted on NaN/Inf loss in "
                        f"{guard.strikes} epochs for {log_prefix}; aborting run."
                    )
                break  # Break from batch loop

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size

            if log_batch_loss and (
                (batch_idx + 1) % ctx.log_interval == 0
                # ponytail: DataLoader is Sized but not a Sequence, so the
                # Iterable annotation makes mypy reject len(); ignore locally.
                or batch_idx == len(train_loader) - 1  # type: ignore[arg-type]
            ):
                metrics: dict[str, int | float] = {
                    "global_step": current_global_step,
                    f"{log_prefix}/Train_Loss_Batch": loss.item(),
                }
                log_metrics(metrics, wandb_run=ctx.wandb_run, commit=True)
                pbar.set_postfix(loss=f"{loss.item():.6f}")

        if guard.epoch_aborted:
            # RUN-004: do not terminate the whole epoch loop on the first
            # NaN/Inf abort; a second aborting epoch raises instead.
            logger.error(
                f"Terminating {log_prefix} epoch {epoch + 1} due to invalid loss."
            )
            continue

        final_avg_epoch_loss = (
            epoch_loss / epoch_samples if epoch_samples > 0 else float("nan")
        )
        if on_epoch_summary is not None:
            on_epoch_summary(epoch, final_avg_epoch_loss)
        if validate is not None and validate(epoch):
            break

    return EpochLoopResult(
        epochs_trained=epochs_trained,
        final_avg_epoch_loss=final_avg_epoch_loss,
        peak_mem=peak_mem,
    )


__all__ = [
    "EpochContext",
    "EpochLoopResult",
    "NanLossGuard",
    "build_optimizer",
    "run_epochs",
]
