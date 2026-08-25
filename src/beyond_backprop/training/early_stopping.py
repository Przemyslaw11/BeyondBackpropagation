"""Serializable early-stopping state shared by algorithm trainers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

# Canonical early-stopping defaults (WP8): single source of truth for the
# fallbacks used by trainer config lookups; configs/base.yaml remains the
# user-facing source of truth.
DEFAULT_PATIENCE = 10
DEFAULT_MIN_DELTA = 0.0


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "min"
    min_delta: float = 0.0
    best_value: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        self.mode = self.mode.lower()
        if self.mode not in {"min", "max"}:
            raise ValueError("EarlyStopping.mode must be 'min' or 'max'")
        if self.patience < 0:
            raise ValueError("EarlyStopping.patience must be non-negative")

    def update(self, value: float, epoch: int) -> bool:
        """Record a metric and return whether training should stop."""

        if not isfinite(float(value)):
            self.bad_epochs += 1
            return self.bad_epochs > self.patience
        improved = self.best_value is None or (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )
        if improved:
            self.best_value = float(value)
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs > self.patience

    def state_dict(self) -> dict[str, float | int | str | None]:
        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> EarlyStopping:
        return cls(
            patience=int(state["patience"]),
            mode=str(state["mode"]),
            min_delta=float(state["min_delta"]),
            best_value=None
            if state.get("best_value") is None
            else float(state["best_value"]),
            best_epoch=None
            if state.get("best_epoch") is None
            else int(state["best_epoch"]),
            bad_epochs=int(state.get("bad_epochs", 0)),
        )
