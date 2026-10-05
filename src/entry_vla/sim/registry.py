"""Backend selection by name. Unknown names fail loudly; there is no fallback."""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import SimBackend


def _duobench():
    from .duobench import DuoBenchBackend
    return DuoBenchBackend


def _peract2():
    from .peract2 import PerAct2Backend
    return PerAct2Backend


def _mock():
    from .mock import MockBackend
    return MockBackend


BACKENDS = {"duobench": _duobench, "peract2": _peract2, "mock": _mock}


def backend_class(name: str) -> Type[SimBackend]:
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(BACKENDS)} (no silent substitution)")
    return BACKENDS[name]()


def make_backend(config: Dict[str, Any]) -> SimBackend:
    """``config`` is the ``backend`` section: ``{"name": ..., ...}``."""
    return backend_class(config["name"])(config)
