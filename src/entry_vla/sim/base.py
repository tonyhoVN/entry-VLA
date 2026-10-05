"""Abstract simulator backend and backend-agnostic helpers."""

from __future__ import annotations

import abc
import importlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..geometry.rotations import geodesic_angle
from .contracts import (
    ARMS,
    BackendCapabilities,
    BimanualTcpCommand,
    MissingDependencyError,
    Observation,
    StepResult,
    UnsupportedCapabilityError,
)


def require_module(name: str, hint: str):
    """Import a simulator module or raise an actionable MissingDependencyError.

    Callers must only use this lazily (inside methods) so that importing the
    adapter never triggers simulator/asset initialisation.
    """
    try:
        return importlib.import_module(name)
    except ImportError as e:  # pragma: no cover - depends on environment
        raise MissingDependencyError(f"cannot import {name!r} ({e}). {hint}") from e


class SimBackend(abc.ABC):
    """Common interface implemented by every backend adapter.

    Lifecycle: ``make(task)`` -> ``reset(seed)`` -> ``step(...)``* -> ``close()``.
    """

    name: str = "abstract"

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        self.task: Optional[str] = None

    # ------------------------------------------------------------- metadata
    @classmethod
    @abc.abstractmethod
    def capabilities(cls) -> BackendCapabilities: ...

    @abc.abstractmethod
    def list_tasks(self) -> List[str]:
        """Registered task identifiers in the pinned source (not display names)."""

    # ------------------------------------------------------------ lifecycle
    @abc.abstractmethod
    def make(self, task: str) -> None: ...

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> Observation: ...

    @abc.abstractmethod
    def step_native(self, action: Any) -> StepResult:
        """Step with an action in the backend's native convention."""

    @abc.abstractmethod
    def convert_command(self, cmd: BimanualTcpCommand, obs: Observation) -> Any:
        """Shared absolute TCP command -> native action."""

    def step(self, cmd: BimanualTcpCommand, obs: Observation) -> StepResult:
        return self.step_native(self.convert_command(cmd, obs))

    @abc.abstractmethod
    def close(self) -> None: ...

    # ------------------------------------------------------- optional parts
    def task_predicates(self, obs: Observation) -> Dict[str, Any]:
        """Privileged task predicates (oracle namespace). Diagnostics only."""
        raise UnsupportedCapabilityError(f"{self.name}: task predicates not implemented")

    def get_state_snapshot(self) -> Any:
        raise UnsupportedCapabilityError(
            f"{self.name}: full physical snapshots are not supported/verified; "
            "use reset(seed) + replay_prefix() and validate the replay error")

    def restore_state_snapshot(self, snapshot: Any) -> Observation:
        raise UnsupportedCapabilityError(f"{self.name}: snapshot restore not supported/verified")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def pose_errors(obs: Observation, ref_T: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out = {}
    for arm in ARMS:
        T = obs.arms[arm].T_WE
        out[arm] = {
            "pos_m": float(np.linalg.norm(T[:3, 3] - ref_T[arm][:3, 3])),
            "rot_rad": float(geodesic_angle(T[:3, :3], ref_T[arm][:3, :3])),
        }
    return out


def replay_prefix(
    backend: SimBackend,
    seed: int,
    native_actions: Sequence[Any],
    recorded_tcp: Optional[Sequence[Dict[str, np.ndarray]]] = None,
    pos_tol: float = 0.01,
    rot_tol: float = 0.05,
) -> Dict[str, Any]:
    """Reproduce a recorded prefix from initialisation: ``reset(seed)`` then
    replay native actions, measuring TCP error against the recording.

    Saved observations alone are not physical snapshots; this is the
    supported recovery procedure for backends without verified snapshot
    restore. The result reports whether the replay stayed within tolerance;
    callers must reject prefixes that do not.
    """
    obs = backend.reset(seed=seed)
    errors: List[Dict[str, Dict[str, float]]] = []
    for k, a in enumerate(native_actions):
        res = backend.step_native(a)
        obs = res.obs
        if recorded_tcp is not None:
            errors.append(pose_errors(obs, recorded_tcp[k]))
        if res.terminated or res.truncated:
            break
    max_pos = max((e[a]["pos_m"] for e in errors for a in ARMS), default=0.0)
    max_rot = max((e[a]["rot_rad"] for e in errors for a in ARMS), default=0.0)
    return {
        "obs": obs,
        "steps": len(errors) if recorded_tcp is not None else len(native_actions),
        "max_pos_err_m": max_pos,
        "max_rot_err_rad": max_rot,
        "within_tolerance": (recorded_tcp is None) or (max_pos <= pos_tol and max_rot <= rot_tol),
        "validated": recorded_tcp is not None,
    }
