"""DuoBench / Robot Control Stack (MuJoCo, Gymnasium) backend adapter.

Verified against source (read, not executed) at
  duobench@85fdb6cc0615c078af8247f49576072f1f36fa8b
  robot-control-stack@c7e81ed359d95ff40713fcac37d71759ba10e8f8
Facts used (file:line in those commits):
  * tasks register ``gym.register(id="duobench/<task>", entry_point=<Task>EnvConfig())``
    only on import of ``duobench.tasks.<task>``; ``gym.make(id, cfg=cfg)`` with
    ``cfg = <Task>EnvConfig().config()`` (scripts/gym_make.py).
  * ``SimEnvCreatorConfig`` fields: control_mode, relative_to, sim_cfg, headless,
    wrapper_cfg(binary_gripper, home_on_reset, include_depth), root_frame_to_world,
    shared_base_frame_to_root_frame (rcs/envs/scenes.py:49-136).
  * obs = {"left"/"right": {joints(7), tquat(7, xyzw, shared base frame), xyzrpy,
    gripper([0,1], 0=closed)}, "frames": {cam: {"rgb"/"depth": {data, intrinsics(3x4),
    extrinsics(4x4 world->camera, z forward)}}}} (rcs/envs/base.py, camera/sim.py:55-118).
  * depth only with include_depth=True; SimCameraSet(physical_units=True) -> metric
    depth * DEPTH_SCALE(1000) as uint16 => millimetres (scenes.py:375, camera/sim.py:67-88).
  * action = {arm: {"tquat"|"joints"|"xyzrpy": ..., "gripper": [g]}}, gripper 1=open.

UNVERIFIED until the user runs ``scripts/smoke_backend.py --backend duobench``:
runtime behaviour, the exact frame of ``tquat`` vs the MuJoCo world (we compose
``root_frame_to_world @ shared_base_frame_to_root_frame``), and the Robotiq
gripper width scale.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..geometry.camera import decode_depth, valid_depth_mask
from ..geometry.transforms import T_to_pose, invert_T, pose_to_T
from .base import SimBackend, require_module
from .contracts import (
    ARMS,
    ArmState,
    BackendCapabilities,
    BimanualTcpCommand,
    CameraFrame,
    MissingAssetError,
    Observation,
    StepResult,
    UnsupportedCapabilityError,
)

DUOBENCH_COMMIT = "85fdb6cc0615c078af8247f49576072f1f36fa8b"
RCS_COMMIT = "c7e81ed359d95ff40713fcac37d71759ba10e8f8"

# registered id suffix -> EnvConfig class name in duobench.tasks.<suffix>
DUOBENCH_TASKS = {
    "ball_maze": "BallMazeEnvConfig",
    "bin_sort": "BinSortEnvConfig",
    "block_balance": "BlockBalanceEnvConfig",
    "carry_pot": "CarryPotEnvConfig",
    "hinge_chest": "HingeChestEnvConfig",
    "join_blocks": "JoinBlocksEnvConfig",
    "pour_marbles": "PourMarblesEnvConfig",
    "spring_door": "SpringDoorEnvConfig",
    "transfer_cube": "TransferCubeEnvConfig",
    "transfer_gate": "TransferGateEnvConfig",
    "transfer_reorient": "TransferReorientEnvConfig",
}
# Paper/display names are NOT registered ids.
DISPLAY_NAMES = {"Carry-Pot": "carry_pot", "Transfer-Cube": "transfer_cube", "Bin-Sort": "bin_sort"}
CAMERAS = ("head", "left_wrist", "right_wrist")
WRIST = {"left_wrist", "right_wrist"}
RCS_QUAT_ORDER = "xyzw"
ROBOTIQ_2F85_MAX_WIDTH_M = 0.085   # nominal stroke of the published gripper (spec sheet)


def resolve_task_id(task: str) -> str:
    """Accept ``carry_pot``, ``duobench/carry_pot`` or a known display name."""
    t = DISPLAY_NAMES.get(task, task)
    t = t.split("/", 1)[1] if t.startswith("duobench/") else t
    if t not in DUOBENCH_TASKS:
        raise ValueError(f"unknown DuoBench task {task!r}; registered: {sorted(DUOBENCH_TASKS)}")
    return t


def _pose_matrix(pose: Any) -> np.ndarray:
    """rcs.common.Pose -> 4x4 (``pose_matrix()`` as used in rcs/camera/sim.py)."""
    return np.asarray(pose.pose_matrix(), dtype=np.float64)


class DuoBenchBackend(SimBackend):
    name = "duobench"

    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            name=cls.name,
            world_frame="mujoco_world",
            quaternion_order=RCS_QUAT_ORDER,
            native_action_modes=("CARTESIAN_TQuat", "CARTESIAN_TRPY", "JOINTS"),
            native_action_note=("RCS Cartesian modes solve IK internally per step (no path planning); "
                                "relative_to selects LAST_STEP/CONFIGURED_ORIGIN/NONE"),
            camera_names=CAMERAS,
            gripper="Robotiq 2F-85 (stock DuoBench scene)",
            live_depth=True,
            recorded_depth="absent",  # duobench_raw carry_pot/sim parquet has no depth columns
            state_snapshot="unverified",
            prefix_replay=True,
            task_predicates=True,
            stage_annotations=True,
            demo_formats=("duobench_raw_parquet", "lerobot_v3_joint"),
            demo_provenance="human teleoperation recordings (RCS StorageWrapper)",
            pinned_source={"duobench": DUOBENCH_COMMIT, "robot-control-stack": RCS_COMMIT},
        )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.env = None
        self.cfg = None
        self.T_W_shared = np.eye(4)
        self._step = 0
        self._last_info: Dict[str, Any] = {}
        self.control_hz = float(config.get("control_hz", 30.0))
        self.gripper_max_width = float(config.get("gripper_max_width_m", ROBOTIQ_2F85_MAX_WIDTH_M))

    def list_tasks(self) -> List[str]:
        return sorted(DUOBENCH_TASKS)

    # ---------------------------------------------------------------- build
    def build_cfg(self, task: str):
        """Create the RCS env config object without creating the env."""
        t = resolve_task_id(task)
        mod = require_module(f"duobench.tasks.{t}", "Install the DuoBench env: see envs/duobench/README.md")
        cfg = getattr(mod, DUOBENCH_TASKS[t])().config()
        rcs_base = require_module("rcs.envs.base", "Install robot-control-stack (rcs-core)")
        c = self.config
        cfg.control_mode = getattr(rcs_base.ControlMode, c.get("control_mode", "CARTESIAN_TQuat"))
        cfg.relative_to = getattr(rcs_base.RelativeTo, c.get("relative_to", "NONE"))
        cfg.headless = bool(c.get("headless", True))
        cfg.wrapper_cfg.include_depth = bool(c.get("include_depth", True))
        cfg.wrapper_cfg.binary_gripper = bool(c.get("binary_gripper", False))
        sim = c.get("sim", {})
        for k in ("async_control", "realtime", "frequency", "max_convergence_steps"):
            if k in sim:
                setattr(cfg.sim_cfg, k, sim[k])
        res = c.get("camera_resolution")  # [H, W] override, None keeps stock 1280x720
        if res and cfg.camera_cfgs:
            for cam_cfg in cfg.camera_cfgs.values():
                cam_cfg.resolution_height, cam_cfg.resolution_width = int(res[0]), int(res[1])
        return t, cfg

    def make(self, task: str) -> None:
        gym = require_module("gymnasium", "Install gymnasium~=0.29.1 (pulled in by rcs-core)")
        t, cfg = self.build_cfg(task)
        try:
            self.env = gym.make(f"duobench/{t}", cfg=cfg)
        except FileNotFoundError as e:
            raise MissingAssetError(
                f"DuoBench/RCS assets missing ({e}). Assets are fetched into ~/.duobench and ~/.rcs on "
                "first import (DUOBENCH_PREFIX/RCS_PREFIX); fetch them deliberately per envs/duobench/README.md"
            ) from e
        self.cfg = cfg
        self.task = t
        self.T_W_shared = _pose_matrix(cfg.root_frame_to_world) @ _pose_matrix(cfg.shared_base_frame_to_root_frame)

    # ----------------------------------------------------------- conversion
    def _arm_state(self, arm_obs: Dict[str, Any], arm_info: Dict[str, Any]) -> ArmState:
        tq = np.asarray(arm_obs["tquat"], dtype=np.float64)
        T_shared_E = pose_to_T(tq[:3], tq[3:7], RCS_QUAT_ORDER)
        g = float(np.asarray(arm_obs["gripper"]).reshape(-1)[0])
        return ArmState(
            T_WE=self.T_W_shared @ T_shared_E,
            q=np.asarray(arm_obs["joints"], dtype=np.float64),
            dq=None,  # not in obs; available only through info sim_state (qvel) if parsed
            gripper_width=g * self.gripper_max_width,
            gripper_native=g,
            gripper_native_semantics="normalized_width_0closed_1open",
        )

    def _camera(self, name: str, frame: Dict[str, Any]) -> CameraFrame:
        rgb = frame["rgb"]
        K = np.asarray(rgb["intrinsics"], dtype=np.float64)[:3, :3]
        T_CW = np.asarray(rgb["extrinsics"], dtype=np.float64)
        img = np.asarray(rgb["data"])
        cam = CameraFrame(name=name, K=K, T_WC=invert_T(T_CW), width=img.shape[1], height=img.shape[0],
                          rgb=img, moving=name in WRIST)
        if "depth" in frame:
            raw = np.asarray(frame["depth"]["data"])
            raw = raw[..., 0] if raw.ndim == 3 else raw
            d = decode_depth(raw, "uint16_mm")
            cam.depth = d
            cam.depth_valid = valid_depth_mask(d) & (raw > 0) & (raw < np.iinfo(np.uint16).max)
            cam.depth_source_encoding = "uint16_mm"
        return cam

    def normalize_observation(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Observation:
        arms = {a: self._arm_state(obs[a], info.get(a, {})) for a in ARMS}
        cams = {n: self._camera(n, f) for n, f in obs.get("frames", {}).items()}
        oracle = {k: info[k] for k in ("success", "stage", "max_stage", "current_subinstruction") if k in info}
        for a in ARMS:
            ai = info.get(a, {})
            for k in ("is_grasped", "collision", "ik_success", "is_sim_converged", "gripper_width"):
                if k in ai:
                    oracle[f"{a}.{k}"] = ai[k]
        return Observation(step=self._step, time=self._step / self.control_hz, arms=arms, cameras=cams,
                           instruction=str(info.get("instruction", "")), oracle=oracle,
                           backend_info={"frame_timestamp": info.get("frame_timestamp"),
                                         "camera_available": info.get("camera_available")})

    # ------------------------------------------------------------ lifecycle
    def _require_env(self):
        if self.env is None:
            raise RuntimeError("call make(task) before reset/step")

    def reset(self, seed: Optional[int] = None) -> Observation:
        self._require_env()
        obs, info = self.env.reset(seed=seed)
        self._step = 0
        self._last_info = info
        return self.normalize_observation(obs, info)

    def step_native(self, action: Any) -> StepResult:
        self._require_env()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        self._last_info = info
        return StepResult(self.normalize_observation(obs, info), float(reward), bool(terminated),
                          bool(truncated), info)

    def convert_command(self, cmd: BimanualTcpCommand, obs: Observation) -> Dict[str, Any]:
        mode = self.config.get("control_mode", "CARTESIAN_TQuat")
        rel = self.config.get("relative_to", "NONE")
        if mode != "CARTESIAN_TQuat" or rel != "NONE":
            raise UnsupportedCapabilityError(
                f"absolute TCP commands need control_mode=CARTESIAN_TQuat, relative_to=NONE (got {mode}, {rel})")
        T_shared_W = invert_T(self.T_W_shared)
        out = {}
        for a in ARMS:
            p, q = T_to_pose(T_shared_W @ cmd.T_WE[a], RCS_QUAT_ORDER)
            out[a] = {"tquat": np.concatenate([p, q]).tolist(), "gripper": [float(cmd.gripper[a])]}
        return out

    def task_predicates(self, obs: Observation) -> Dict[str, Any]:
        return dict(obs.oracle)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
