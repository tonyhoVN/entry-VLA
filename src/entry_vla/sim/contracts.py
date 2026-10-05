"""Backend-independent simulator contract: normalised observations, commands,
capability metadata and explicit errors.

Normalised frames: every pose is ``T_W*`` in the backend's declared world
frame ``W`` (see ``BackendCapabilities.world_frame``); cameras are OpenCV
(+z forward, +y down). Quaternions never cross this boundary; adapters
convert them using the backend's declared order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

ARMS = ("left", "right")


class BackendError(RuntimeError):
    """Base class for actionable backend errors."""


class MissingDependencyError(BackendError):
    """A simulator package is not importable in this environment."""


class MissingAssetError(BackendError):
    """Assets/datasets are absent. Raised instead of triggering a download."""


class UnsupportedCapabilityError(BackendError):
    """The backend does not implement (or has not verified) a capability.
    Never silently substitute another backend or a degraded behaviour."""


@dataclass
class CameraFrame:
    name: str
    K: np.ndarray                       # (3,3) intrinsics matching the stored images
    T_WC: np.ndarray                    # (4,4) camera-to-world, OpenCV camera frame
    width: int
    height: int
    rgb: Optional[np.ndarray] = None    # uint8 (H,W,3)
    depth: Optional[np.ndarray] = None  # float32 metres (H,W); None = unavailable
    depth_valid: Optional[np.ndarray] = None
    near: Optional[float] = None
    far: Optional[float] = None
    depth_source_encoding: str = "unavailable"
    moving: bool = False                # wrist cameras move with the arm


@dataclass
class ArmState:
    T_WE: np.ndarray                          # measured TCP pose in W
    q: np.ndarray                             # joint positions (rad)
    dq: Optional[np.ndarray] = None           # joint velocities (rad/s); None = unavailable
    gripper_width: Optional[float] = None     # metres, measured
    gripper_native: Optional[float] = None    # backend-native gripper reading
    gripper_native_semantics: str = ""        # e.g. "normalized_width_0closed"


@dataclass
class Observation:
    step: int
    time: float
    arms: Dict[str, ArmState]
    cameras: Dict[str, CameraFrame]
    instruction: str = ""
    # Privileged simulator information (object poses, stage, success, contacts).
    # Annotation/diagnostics ONLY: predictor input builders must not read it.
    oracle: Dict[str, Any] = field(default_factory=dict)
    backend_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BimanualTcpCommand:
    """Absolute TCP targets in W plus normalised gripper commands
    (``1.0`` = open, ``0.0`` = closed). Adapters convert to native actions."""

    T_WE: Dict[str, np.ndarray]
    gripper: Dict[str, float]

    def __post_init__(self) -> None:
        for arm in ARMS:
            if arm not in self.T_WE or arm not in self.gripper:
                raise ValueError(f"command missing arm {arm!r}")
            g = float(self.gripper[arm])
            if not 0.0 <= g <= 1.0:
                raise ValueError(f"gripper command for {arm} must be in [0,1] (1=open), got {g}")


@dataclass
class StepResult:
    obs: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    world_frame: str
    quaternion_order: str                       # at the native API boundary
    native_action_modes: Tuple[str, ...]
    native_action_note: str                     # e.g. internal motion planning in the action mode
    camera_names: Tuple[str, ...]
    gripper: str
    live_depth: bool                            # simulator can render depth when asked
    recorded_depth: str                         # "absent" | "present" | "unverified"
    state_snapshot: str                         # "unsupported" | "unverified" | "supported"
    prefix_replay: bool                         # reset(seed) + replay of recorded actions
    task_predicates: bool
    stage_annotations: bool
    demo_formats: Tuple[str, ...]
    demo_provenance: str
    pinned_source: Dict[str, str] = field(default_factory=dict)

    def require(self, capability: str) -> None:
        val = getattr(self, capability)
        if val in (False, "unsupported", "absent"):
            raise UnsupportedCapabilityError(f"backend {self.name!r} does not support {capability!r} ({val})")

    def as_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}
