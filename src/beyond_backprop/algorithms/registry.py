"""Explicit registry for canonical algorithm adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config.models import AlgorithmName
from .base import AlgorithmAdapter
from .bp import BPAdapter
from .cafo import CaFoAdapter
from .ff import FFAdapter
from .mf import MFAdapter

AlgorithmFactory = Callable[[], AlgorithmAdapter]


class AlgorithmRegistry:
    """Map scientific algorithm names to lifecycle adapters."""

    def __init__(self) -> None:
        self._factories: dict[str, AlgorithmFactory] = {}

    def register(self, name: str | AlgorithmName, factory: AlgorithmFactory) -> None:
        key = self._key(name)
        if key in self._factories:
            raise ValueError(f"Algorithm already registered: {name}")
        self._factories[key] = factory

    def lookup(self, name: str | AlgorithmName) -> AlgorithmFactory:
        key = self._key(name)
        try:
            return self._factories[key]
        except KeyError as exc:
            raise KeyError(f"Unknown algorithm: {name}") from exc

    def build(self, name: str | AlgorithmName, **_: Any) -> AlgorithmAdapter:
        return self.lookup(name)()

    @staticmethod
    def _key(name: str | AlgorithmName) -> str:
        if isinstance(name, AlgorithmName):
            return name.value
        return str(name).strip().lower().replace("-", "_")


ALGORITHM_REGISTRY = AlgorithmRegistry()
ALGORITHM_REGISTRY.register(AlgorithmName.BP, BPAdapter)
ALGORITHM_REGISTRY.register(AlgorithmName.FF, FFAdapter)
ALGORITHM_REGISTRY.register(AlgorithmName.CAFO, CaFoAdapter)
ALGORITHM_REGISTRY.register(AlgorithmName.MF, MFAdapter)


def build_algorithm(name: str | AlgorithmName, **kwargs: Any) -> AlgorithmAdapter:
    return ALGORITHM_REGISTRY.build(name, **kwargs)


def get_algorithm(name: str | AlgorithmName, **kwargs: Any) -> AlgorithmAdapter:
    """Compatibility spelling for registry-backed adapter construction."""
    return build_algorithm(name, **kwargs)


__all__ = [
    "ALGORITHM_REGISTRY",
    "AlgorithmFactory",
    "AlgorithmRegistry",
    "build_algorithm",
    "get_algorithm",
]
