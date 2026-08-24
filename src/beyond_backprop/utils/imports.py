"""Import-tree diagnostics used by verification and downstream tooling."""

from __future__ import annotations


def canonical_imports() -> tuple[str, ...]:
    """Return the canonical package roots that must remain importable."""

    return (
        "beyond_backprop.config",
        "beyond_backprop.runtime",
        "beyond_backprop.data",
        "beyond_backprop.algorithms",
        "beyond_backprop.experiment",
        "beyond_backprop.evaluation",
        "beyond_backprop.training",
        "beyond_backprop.monitoring",
        "beyond_backprop.tracking",
        "beyond_backprop.tuning",
    )


__all__ = ["canonical_imports"]
