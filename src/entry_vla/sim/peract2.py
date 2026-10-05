"""PerAct2 bimanual RLBench (CoppeliaSim 4.1 / PyRep fork) backend adapter.

Verified against source (read, not executed) at
  markusgrotz/RLBench@8af748c51287989294e00c9c670e3330a0e35ed5 (branch main)
  markusgrotz/PyRep@b8bd1d7a3182adcd570d001649c0849047ebf197
  markusgrotz/peract_bimanual@bb0232a6ba3fe116566e9568f0c7af980ed6703d
Facts used:
  * ``Environment(action_mode, obs_config=..., headless=..., robot_setup='dual_panda')``;
    ``env.launch()``; ``env.get_task(cls)``; ``task.reset() -> (descriptions, obs)``;
    ``task.step(action) -> (obs, reward, terminate)`` (rlbench/environment.py,
    task_environment.py).
  * bimanual tasks resolve via ``rlbench.utils.name_to_task_class(name, bimanual=True)``
    (``Environment._string_to_task`` only searches unimanual tasks).
  * ``BimanualMoveArmThenGripper(BimanualEndEffectorPoseViaPlanning(), BimanualDiscrete())``
    takes an 18-vector, RIGHT arm first: per arm ``[x,y,z,qx,qy,qz,qw,gripper,ignore_collisions]``,
    quaternion xyzw, world frame, gripper > 0.5 = open. The arm mode runs CoppeliaSim
    motion planning internally (report this; keep it matched across baselines).
  * ``BimanualObservation.left/right``: gripper_pose (xyz + xyzw), joint_positions,
    joint_velocities, gripper_open, gripper_joint_positions. Images in
    ``obs.perception_data['{cam}_{rgb|depth|mask}']``; ``obs.misc`` has
    ``{cam}_camera_extrinsics`` (camera->world), ``_intrinsics`` (PyRep: negative focal
    lengths), ``_near``, ``_far``.
  * ``CameraConfig.depth_in_meters`` defaults to False (linear [0,1] buffer).

Demonstrations are simulator-generated from task waypoints + motion planning,
NOT human teleoperation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..geometry.camera import decode_depth, normalize_negative_focal, valid_depth_mask
from ..geometry.transforms import T_to_pose, pose_to_T
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
)

RLBENCH_COMMIT = "8af748c51287989294e00c9c670e3330a0e35ed5"
PYREP_COMMIT = "b8bd1d7a3182adcd570d001649c0849047ebf197"
PERACT_BIMANUAL_COMMIT = "bb0232a6ba3fe116566e9568f0c7af980ed6703d"
COPPELIASIM_VERSION = "4.1.0"

BENCHMARK_TASKS = (
    "bimanual_push_box", "bimanual_lift_ball", "bimanual_dual_push_buttons", "bimanual_pick_plate",
    "bimanual_put_item_in_drawer", "bimanual_put_bottle_in_fridge", "bimanual_handover_item",
    "bimanual_pick_laptop", "bimanual_straighten_rope", "bimanual_sweep_to_dustpan", "bimanual_lift_tray",
    "bimanual_handover_item_easy", "bimanual_take_tray_out_of_oven",
)
EXTRA_TASKS = (
    "bimanual_set_the_table", "coordinated_close_jar", "coordinated_lift_stick",
    "coordinated_put_item_in_drawer_right", "coordinated_take_shoes_out_of_box", "handover_item_medium",
    "left_open_drawer", "right_open_drawer", "unimanual_push_single_button_left",
    "unimanual_push_single_button_right",
)
CAMERAS = ("over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front")
NATIVE_ARM_ORDER = ("right", "left")  # BimanualMoveArmThenGripper splits right first
RLBENCH_QUAT_ORDER = "xyzw"


def class_name_for(task: str) -> str:
    return "".join(w[0].upper() + w[1:] for w in task.split("_"))


def encode_native_action(cmd: BimanualTcpCommand, ignore_collisions: Dict[str, float]) -> np.ndarray:
    """Shared command -> 18D BimanualMoveArmThenGripper action (right first)."""
    parts = []
    for arm in NATIVE_ARM_ORDER:
        p, q = T_to_pose(cmd.T_WE[arm], RLBENCH_QUAT_ORDER)
        parts.append(np.concatenate([p, q, [1.0 if cmd.gripper[arm] > 0.5 else 0.0],
                                     [float(ignore_collisions[arm])]]))
    return np.concatenate(parts)


def decode_native_action(action: np.ndarray) -> Dict[str, Dict[str, Any]]:
    a = np.asarray(action, dtype=np.float64).reshape(18)
    out = {}
    for i, arm in enumerate(NATIVE_ARM_ORDER):
        b = a[9 * i: 9 * (i + 1)]
        out[arm] = {"T_WE": pose_to_T(b[:3], b[3:7], RLBENCH_QUAT_ORDER), "gripper_open": b[7] > 0.5,
                    "ignore_collisions": b[8] > 0.5}
    return out


def camera_from_misc(name: str, misc: Dict[str, Any], rgb: Optional[np.ndarray], depth_raw: Optional[np.ndarray],
                     depth_encoding: str) -> CameraFrame:
    """Build a normalised camera from RLBench ``misc`` + images.

    PyRep intrinsics carry negative focal lengths; they are rewritten as an
    equivalent positive-focal camera (frame rotated pi about the optical axis).
    """
    K_raw = np.asarray(misc[f"{name}_camera_intrinsics"], dtype=np.float64)
    T_WC_raw = np.asarray(misc[f"{name}_camera_extrinsics"], dtype=np.float64)
    near = float(misc[f"{name}_camera_near"])
    far = float(misc[f"{name}_camera_far"])
    K, T_WC = normalize_negative_focal(K_raw, T_WC_raw)
    ref = rgb if rgb is not None else depth_raw
    H, W = (ref.shape[0], ref.shape[1]) if ref is not None else (0, 0)
    cam = CameraFrame(name=name, K=K, T_WC=T_WC, width=W, height=H, rgb=rgb, near=near, far=far,
                      moving=name.startswith("wrist"))
    if depth_raw is not None:
        d = decode_depth(depth_raw, depth_encoding, near=near, far=far)
        cam.depth = d
        cam.depth_valid = valid_depth_mask(d, near=near, far=far)
        cam.depth_source_encoding = depth_encoding
    return cam


def arm_state_from_obs(arm_obs: Any) -> ArmState:
    gp = np.asarray(arm_obs.gripper_pose, dtype=np.float64)
    gjp = getattr(arm_obs, "gripper_joint_positions", None)
    width = float(np.sum(gjp)) if gjp is not None else None  # two Panda finger joints (m)
    dq = getattr(arm_obs, "joint_velocities", None)
    return ArmState(
        T_WE=pose_to_T(gp[:3], gp[3:7], RLBENCH_QUAT_ORDER),
        q=np.asarray(arm_obs.joint_positions, dtype=np.float64),
        dq=None if dq is None else np.asarray(dq, dtype=np.float64),
        gripper_width=width,
        gripper_native=float(arm_obs.gripper_open) if arm_obs.gripper_open is not None else None,
        gripper_native_semantics="gripper_open_binary_1open",
    )


class PerAct2Backend(SimBackend):
    name = "peract2"

    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            name=cls.name,
            world_frame="coppeliasim_world",
            quaternion_order=RLBENCH_QUAT_ORDER,
            native_action_modes=("BimanualMoveArmThenGripper(BimanualEndEffectorPoseViaPlanning, BimanualDiscrete)",
                                 "BimanualJointPositionActionMode"),
            native_action_note="EndEffectorPoseViaPlanning runs CoppeliaSim motion planning inside each step",
            camera_names=CAMERAS,
            gripper="Franka/Panda hand (dual_panda)",
            live_depth=True,
            recorded_depth="present",  # rgb24-packed linear buffer PNGs in the PerAct2 dataset
            state_snapshot="unsupported",
            prefix_replay=True,
            task_predicates=True,
            stage_annotations=False,
            demo_formats=("rlbench_bimanual_episode_dir",),
            demo_provenance="simulator-generated waypoint demonstrations with motion planning (not teleoperation)",
            pinned_source={"RLBench(markusgrotz,main)": RLBENCH_COMMIT, "PyRep(markusgrotz)": PYREP_COMMIT,
                           "peract_bimanual": PERACT_BIMANUAL_COMMIT, "CoppeliaSim": COPPELIASIM_VERSION},
        )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.env = None
        self.task_env = None
        self._step = 0
        self.descriptions: List[str] = []
        self.cameras = tuple(config.get("cameras", ("front", "over_shoulder_left", "over_shoulder_right",
                                                    "wrist_left", "wrist_right")))
        self.ignore_collisions = {a: float(config.get("ignore_collisions", 0.0)) for a in ARMS}

    def list_tasks(self) -> List[str]:
        return list(BENCHMARK_TASKS) + list(EXTRA_TASKS)

    def _obs_config(self):
        oc_mod = require_module("rlbench.observation_config", "Install the PerAct2 RLBench fork: envs/peract2/README.md")
        size = tuple(self.config.get("image_size", (128, 128)))
        cams = {}
        for name in CAMERAS:
            on = name in self.cameras
            cams[name] = oc_mod.CameraConfig(rgb=on, depth=on, point_cloud=False, mask=False, image_size=size,
                                             depth_in_meters=True)
        oc = oc_mod.ObservationConfig(camera_configs=cams)
        oc.set_all_low_dim(True)
        return oc

    def make(self, task: str) -> None:
        import os
        if task not in self.list_tasks():
            raise ValueError(f"unknown PerAct2 task {task!r}; known: {self.list_tasks()}")
        if not os.environ.get("COPPELIASIM_ROOT"):
            raise MissingAssetError("COPPELIASIM_ROOT is not set; install CoppeliaSim 4.1.0 (envs/peract2/README.md)")
        env_mod = require_module("rlbench.environment", "Install the PerAct2 RLBench fork")
        am = require_module("rlbench.action_modes.action_mode", "PerAct2 RLBench fork")
        arm = require_module("rlbench.action_modes.arm_action_modes", "PerAct2 RLBench fork")
        grip = require_module("rlbench.action_modes.gripper_action_modes", "PerAct2 RLBench fork")
        utils = require_module("rlbench.utils", "PerAct2 RLBench fork")
        action_mode = am.BimanualMoveArmThenGripper(arm.BimanualEndEffectorPoseViaPlanning(), grip.BimanualDiscrete())
        self.env = env_mod.Environment(action_mode, obs_config=self._obs_config(),
                                       headless=bool(self.config.get("headless", True)), robot_setup="dual_panda")
        self.env.launch()
        self.task_env = self.env.get_task(utils.name_to_task_class(task, bimanual=True))
        self.task = task

    def normalize_observation(self, obs: Any) -> Observation:
        arms = {"left": arm_state_from_obs(obs.left), "right": arm_state_from_obs(obs.right)}
        cams = {}
        for name in self.cameras:
            rgb = obs.perception_data.get(f"{name}_rgb")
            depth = obs.perception_data.get(f"{name}_depth")  # metric: depth_in_meters=True above
            cams[name] = camera_from_misc(name, obs.misc, None if rgb is None else np.asarray(rgb),
                                          None if depth is None else np.asarray(depth, dtype=np.float32), "metric_m")
        low = obs.task_low_dim_state
        if isinstance(low, tuple) and len(low) == 1:  # trailing-comma quirk in scene.get_observation
            low = low[0]
        return Observation(step=self._step, time=float(self._step), arms=arms, cameras=cams,
                           instruction=self.descriptions[0] if self.descriptions else "",
                           oracle={"task_low_dim_state": None if low is None else np.asarray(low)},
                           backend_info={"time_unit": "planner_steps", "variation_index": obs.misc.get("variation_index")})

    def reset(self, seed: Optional[int] = None) -> Observation:
        if self.task_env is None:
            raise RuntimeError("call make(task) before reset")
        if seed is not None:
            np.random.seed(seed)  # RLBench samples variations/poses with numpy's global RNG
        self.descriptions, obs = self.task_env.reset()
        self._step = 0
        return self.normalize_observation(obs)

    def step_native(self, action: Any) -> StepResult:
        obs, reward, terminate = self.task_env.step(np.asarray(action, dtype=np.float64))
        self._step += 1
        return StepResult(self.normalize_observation(obs), float(reward), bool(terminate), False,
                          {"success": float(reward) > 0 and bool(terminate)})

    def convert_command(self, cmd: BimanualTcpCommand, obs: Observation) -> np.ndarray:
        return encode_native_action(cmd, self.ignore_collisions)

    def task_predicates(self, obs: Observation) -> Dict[str, Any]:
        success, _ = self.task_env._task.success()  # rlbench Task.success() -> (bool, bool)
        return {"success": bool(success), **obs.oracle}

    def close(self) -> None:
        if self.env is not None:
            self.env.shutdown()
            self.env = None
            self.task_env = None
