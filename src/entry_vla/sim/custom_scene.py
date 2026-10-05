"""Separately named custom scene: two Franka arms on ~45-degree tilted mounts,
default Franka hands, four cameras (ego, exo, two wrist).

Stock benchmark scenes are never modified. Episodes carry ``meta.scene_id``;
``check_scene_compat`` refuses to mix demonstrations across geometries unless a
validation/recollection record is supplied.

What this module provides (runnable): the scene specification, mount
transforms, per-base gravity vectors, gripper/TCP geometry, camera rigs, arm
models for planning/selection, and an RCS-config patch helper. Building the
custom MuJoCo assets inside RCS is user-run; the RCS fields it touches
(``robot_to_shared_base_frame``, ``camera_cfgs``, ``gripper_cfgs``) were read from
source, but the Franka-hand gripper config type in RCS is UNVERIFIED and must
be supplied by the user (``rcs_gripper_cfg_factory``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..geometry.camera import CameraIntrinsics
from ..geometry.franka import FRANKA_HAND_TCP, PandaArm
from ..geometry.rotations import rot_x, rot_z
from ..geometry.transforms import look_at, make_T

STOCK_SCENE_IDS = {"duobench": "duobench_stock_fr3duo_robotiq", "peract2": "peract2_stock_dual_panda", "mock": "mock_v1"}
GRAVITY_W = np.array([0.0, 0.0, -9.81])


@dataclass
class CameraRig:
    name: str
    width: int
    height: int
    fovy_deg: float
    mount: str                       # "world" or "left_tcp" / "right_tcp"
    T_parent_C: np.ndarray           # OpenCV camera pose in the parent frame
    depth: bool = True               # simulation renders depth; REAL-system depth availability is unconfirmed


@dataclass
class TiltedDualFrankaSpec:
    scene_id: str = "tilted_dual_franka_v1"
    tilt_deg: float = 45.0           # each base rolled toward the shared workspace
    base_separation_m: float = 0.80
    base_height_m: float = 0.60
    gripper: str = "franka_hand"
    T_flange_tcp: np.ndarray = field(default_factory=lambda: FRANKA_HAND_TCP.copy())
    finger_max_width_m: float = 0.08
    table_z: float = 0.0
    cameras: List[CameraRig] = field(default_factory=list)

    def mounts(self) -> Dict[str, np.ndarray]:
        """``T_WB`` per arm: bases at +-y, rolled about world x by -+tilt so each faces the centre."""
        t = np.deg2rad(self.tilt_deg)
        y = self.base_separation_m / 2
        return {"left": make_T(rot_x(-t), [0.0, y, self.base_height_m]),
                "right": make_T(rot_x(t), [0.0, -y, self.base_height_m])}

    def gravity_in_base(self) -> Dict[str, np.ndarray]:
        """Gravity expressed in each arm base frame (needed by gravity compensation)."""
        return {a: T[:3, :3].T @ GRAVITY_W for a, T in self.mounts().items()}

    def arms(self) -> Dict[str, PandaArm]:
        return {a: PandaArm(T_WB=T, T_flange_tcp=self.T_flange_tcp.copy()) for a, T in self.mounts().items()}

    def camera_intrinsics(self) -> Dict[str, CameraIntrinsics]:
        return {c.name: CameraIntrinsics.from_fovy(np.deg2rad(c.fovy_deg), c.width, c.height) for c in self.cameras}

    def as_dict(self) -> Dict[str, Any]:
        return {"scene_id": self.scene_id, "tilt_deg": self.tilt_deg, "base_separation_m": self.base_separation_m,
                "base_height_m": self.base_height_m, "gripper": self.gripper, "T_flange_tcp": self.T_flange_tcp.tolist(),
                "finger_max_width_m": self.finger_max_width_m,
                "mounts": {a: T.tolist() for a, T in self.mounts().items()},
                "gravity_in_base": {a: g.tolist() for a, g in self.gravity_in_base().items()},
                "cameras": [{"name": c.name, "wh": [c.width, c.height], "fovy_deg": c.fovy_deg, "mount": c.mount,
                             "T_parent_C": np.asarray(c.T_parent_C).tolist(), "depth": c.depth} for c in self.cameras]}


def default_four_camera_rig() -> List[CameraRig]:
    wrist = make_T(rot_z(0.0), [0.05, 0.0, -0.05])
    return [
        CameraRig("ego", 640, 480, 70.0, "world", look_at([-0.20, 0.0, 1.25], [0.45, 0.0, 0.05])),
        CameraRig("exo", 640, 480, 60.0, "world", look_at([1.40, 0.30, 0.90], [0.45, 0.0, 0.05])),
        CameraRig("left_wrist", 424, 240, 87.0, "left_tcp", wrist),
        CameraRig("right_wrist", 424, 240, 87.0, "right_tcp", wrist),
    ]


def load_spec(cfg: Dict[str, Any]) -> TiltedDualFrankaSpec:
    cams = []
    for c in cfg.get("cameras") or []:
        T = c.get("T_parent_C")
        if T is None and c.get("look_at"):
            T = look_at(c["look_at"]["eye"], c["look_at"]["target"])
        cams.append(CameraRig(c["name"], int(c["width"]), int(c["height"]), float(c["fovy_deg"]), c["mount"],
                              np.asarray(T, dtype=np.float64), bool(c.get("depth", True))))
    spec = TiltedDualFrankaSpec(scene_id=cfg.get("scene_id", "tilted_dual_franka_v1"),
                                tilt_deg=float(cfg.get("tilt_deg", 45.0)),
                                base_separation_m=float(cfg.get("base_separation_m", 0.8)),
                                base_height_m=float(cfg.get("base_height_m", 0.6)),
                                finger_max_width_m=float(cfg.get("finger_max_width_m", 0.08)),
                                table_z=float(cfg.get("table_z", 0.0)),
                                cameras=cams or default_four_camera_rig())
    if cfg.get("T_flange_tcp") is not None:
        spec.T_flange_tcp = np.asarray(cfg["T_flange_tcp"], dtype=np.float64)
    if spec.scene_id in STOCK_SCENE_IDS.values():
        raise ValueError("custom scenes must not reuse a stock benchmark scene_id")
    return spec


def check_scene_compat(episode_meta: Dict[str, Any], target_scene_id: str,
                       validation: Optional[Dict[str, Any]] = None) -> None:
    """Refuse demonstrations recorded in a different scene geometry unless an
    explicit validation/recollection record for that pair is given."""
    src = episode_meta.get("scene_id") or STOCK_SCENE_IDS.get(episode_meta.get("backend", ""), "unknown")
    if src == target_scene_id:
        return
    key = f"{src}->{target_scene_id}"
    if validation and validation.get(key, {}).get("validated"):
        return
    raise ValueError(f"episode {episode_meta.get('episode_id')} was recorded in scene {src!r}; target is "
                     f"{target_scene_id!r}. Validate or recollect demonstrations after geometry changes.")


def patch_rcs_config(rcs_cfg: Any, spec: TiltedDualFrankaSpec, pose_factory: Callable[[np.ndarray], Any],
                     rcs_gripper_cfg_factory: Optional[Callable[[TiltedDualFrankaSpec, str], Any]] = None,
                     robot_keys: Dict[str, str] = None) -> Any:  # pragma: no cover - user-run with RCS
    """Apply mounts (and, if a factory is supplied, grippers) to a COPY of an RCS
    ``SimEnvCreatorConfig``. ``pose_factory`` converts 4x4 -> ``rcs.common.Pose``.
    ``robot_keys`` maps left/right to the config's robot names."""
    import copy
    cfg = copy.deepcopy(rcs_cfg)
    keys = robot_keys or {"left": "left", "right": "right"}
    cfg.robot_to_shared_base_frame = {keys[a]: pose_factory(T) for a, T in spec.mounts().items()}
    if rcs_gripper_cfg_factory is None:
        raise NotImplementedError("RCS Franka-hand gripper config type is UNVERIFIED; pass rcs_gripper_cfg_factory")
    cfg.gripper_cfgs = {keys[a]: rcs_gripper_cfg_factory(spec, a) for a in ("left", "right")}
    return cfg
