"""MOCK kinematic bimanual backend for offline tests and CLI dry checks.

This is NOT a simulator benchmark: it has no physics, contacts or meshes.
Arms track commanded TCP poses through damped-least-squares IK with a joint
speed limit; cameras z-buffer a point-sampled table + object. Every result
produced with it is labelled ``backend=mock``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from ..geometry.camera import CameraIntrinsics, render_point_depth, valid_depth_mask
from ..geometry.franka import PANDA_Q_READY, PandaArm
from ..geometry.rotations import rot_x, rot_z
from ..geometry.transforms import look_at, make_T
from .base import SimBackend
from .contracts import (
    ARMS,
    ArmState,
    BackendCapabilities,
    BimanualTcpCommand,
    CameraFrame,
    Observation,
    StepResult,
)

MOCK_TASKS = ("mock_bimanual_reach",)
T_EC_WRIST = make_T(rot_z(0.0), [0.05, 0.0, -0.05])   # wrist camera in TCP frame (mock value)


class MockBackend(SimBackend):
    name = "mock"

    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            name=cls.name, world_frame="mock_world", quaternion_order="xyzw",
            native_action_modes=("absolute_tcp",), native_action_note="kinematic IK tracking, no planning",
            camera_names=("front", "left_wrist", "right_wrist"), gripper="mock parallel (0.08 m)",
            live_depth=True, recorded_depth="present", state_snapshot="supported", prefix_replay=True,
            task_predicates=True, stage_annotations=True, demo_formats=("entry_vla_episode",),
            demo_provenance="scripted mock trajectories", pinned_source={},
        )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        c = config
        self.dt = float(c.get("dt", 0.1))
        self.max_joint_speed = float(c.get("max_joint_speed", 1.5))   # rad/s
        self.gripper_speed = float(c.get("gripper_speed", 0.2))      # m/s
        self.image_hw = tuple(c.get("image_hw", (48, 64)))
        self.success_radius = float(c.get("success_radius", 0.03))
        self.max_steps = int(c.get("max_steps", 200))
        self.lift_height = float(c.get("lift_height", 0.05))
        self.arms = {"left": PandaArm(T_WB=make_T(rot_z(-np.pi / 2), [0.0, 0.45, 0.0])),
                     "right": PandaArm(T_WB=make_T(rot_z(np.pi / 2), [0.0, -0.45, 0.0]))}
        self.state: Dict[str, Any] = {}
        self.instruction = "lift the bar with both hands"

    def list_tasks(self) -> List[str]:
        return list(MOCK_TASKS)

    def make(self, task: str) -> None:
        if task not in MOCK_TASKS:
            raise ValueError(f"unknown mock task {task!r}")
        self.task = task

    # -------------------------------------------------------------- scene
    def _handles(self) -> Dict[str, np.ndarray]:
        c, yaw = self.state["bar_center"], self.state["bar_yaw"]
        d = rot_z(yaw) @ np.array([0.0, 0.12, 0.0])
        return {"left": c + d, "right": c - d}

    def _scene_points(self) -> np.ndarray:
        xs, ys = np.meshgrid(np.linspace(0.2, 0.9, 40), np.linspace(-0.5, 0.5, 50))
        table = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], -1)
        h = self._handles()
        t = np.linspace(0, 1, 40)[:, None]
        bar = h["right"] + t * (h["left"] - h["right"]) + np.array([0, 0, 0.03])
        return np.vstack([table, bar])

    def _cameras(self) -> Dict[str, CameraFrame]:
        H, W = self.image_hw
        intr = CameraIntrinsics.from_fovy(np.deg2rad(60), W, H)
        poses = {"front": look_at([1.4, 0.0, 0.9], [0.35, 0.0, 0.2])}
        for a in ARMS:
            poses[f"{a}_wrist"] = self._T_WE(a) @ T_EC_WRIST
        pts = self._scene_points()
        cams = {}
        for name, T_WC in poses.items():
            d = render_point_depth(pts, intr.K, T_WC, W, H).astype(np.float32)
            valid = valid_depth_mask(d, near=0.01, far=5.0)
            d[~valid] = 0.0
            rgb = np.zeros((H, W, 3), np.uint8)
            rgb[valid] = 180
            cams[name] = CameraFrame(name=name, K=intr.K, T_WC=T_WC, width=W, height=H, rgb=rgb, depth=d,
                                     depth_valid=valid, near=0.01, far=5.0, depth_source_encoding="metric_m",
                                     moving=name != "front")
        return cams

    def _T_WE(self, arm: str) -> np.ndarray:
        return self.arms[arm].fk(self.state["q"][arm])

    def _observe(self) -> Observation:
        arms = {a: ArmState(T_WE=self._T_WE(a), q=self.state["q"][a].copy(), dq=self.state["dq"][a].copy(),
                            gripper_width=self.state["width"][a], gripper_native=self.state["width"][a] / 0.08,
                            gripper_native_semantics="normalized_width_0closed_1open") for a in ARMS}
        h = self._handles()
        oracle = {"handles": h, "bar_center": self.state["bar_center"].copy(), **self._predicates(arms, h)}
        return Observation(step=self.state["step"], time=self.state["step"] * self.dt, arms=arms,
                           cameras=self._cameras(), instruction=self.instruction, oracle=oracle,
                           backend_info={"mock": True})

    def _predicates(self, arms: Dict[str, ArmState], h: Dict[str, np.ndarray]) -> Dict[str, Any]:
        near = {a: float(np.linalg.norm(arms[a].T_WE[:3, 3] - h[a])) for a in ARMS}
        closed = {a: arms[a].gripper_width < 0.02 for a in ARMS}
        grasp = {a: near[a] < self.success_radius and closed[a] for a in ARMS}
        attached = bool(self.state.get("attached", False))
        lifted = attached and float(self.state["bar_center"][2]) > self.lift_height
        stage = int(any(near[a] < 0.10 for a in ARMS)) + int(all(grasp.values()) or attached) + int(lifted)
        return {"dist_to_handle": near, "grasped": grasp, "attached": attached, "stage": stage, "max_stage": 3,
                "success": lifted}

    # ---------------------------------------------------------- lifecycle
    def reset(self, seed: Optional[int] = None) -> Observation:
        rng = np.random.default_rng(seed)
        self.state = {
            "q": {a: PANDA_Q_READY.copy() for a in ARMS},
            "dq": {a: np.zeros(7) for a in ARMS},
            "width": {a: 0.08 for a in ARMS},
            "bar_center": np.array([0.55 + rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.0]),
            "bar_yaw": float(rng.uniform(-0.3, 0.3)),
            "attached": False,
            "attach_offset": None,
            "step": 0,
            "seed": seed,
        }
        return self._observe()

    def step_native(self, action: Any) -> StepResult:
        cmd: BimanualTcpCommand = action
        for a in ARMS:
            q0 = self.state["q"][a]
            q_goal, ok, _ = self.arms[a].ik(cmd.T_WE[a], q_init=q0, max_iters=60)
            dq = q_goal - q0
            lim = self.max_joint_speed * self.dt
            n = np.max(np.abs(dq))
            if n > lim:
                dq *= lim / n
            self.state["q"][a] = q0 + dq
            self.state["dq"][a] = dq / self.dt
            target_w = 0.08 * float(cmd.gripper[a])
            w = self.state["width"][a]
            self.state["width"][a] = w + np.clip(target_w - w, -self.gripper_speed * self.dt, self.gripper_speed * self.dt)
        self._update_attachment()
        self.state["step"] += 1
        obs = self._observe()
        success = bool(obs.oracle["success"])
        return StepResult(obs, float(success), success, self.state["step"] >= self.max_steps,
                          {"success": success, "stage": obs.oracle["stage"]})

    def _update_attachment(self) -> None:
        """Kinematic attachment: once both handles are grasped the bar follows
        the TCP midpoint; opening either gripper drops it back to the table."""
        tcp = {a: self._T_WE(a)[:3, 3] for a in ARMS}
        mid = 0.5 * (tcp["left"] + tcp["right"])
        if self.state["attached"]:
            if any(self.state["width"][a] > 0.03 for a in ARMS):
                self.state["attached"] = False
                self.state["bar_center"][2] = 0.0
            else:
                self.state["bar_center"] = mid - self.state["attach_offset"]
            return
        h = self._handles()
        if all(np.linalg.norm(tcp[a] - h[a]) < self.success_radius and self.state["width"][a] < 0.02 for a in ARMS):
            self.state["attached"] = True
            self.state["attach_offset"] = mid - self.state["bar_center"]

    def convert_command(self, cmd: BimanualTcpCommand, obs: Observation) -> BimanualTcpCommand:
        return cmd

    def task_predicates(self, obs: Observation) -> Dict[str, Any]:
        return {k: obs.oracle[k] for k in ("dist_to_handle", "grasped", "attached", "stage", "max_stage", "success")}

    def get_state_snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    def restore_state_snapshot(self, snapshot: Dict[str, Any]) -> Observation:
        self.state = copy.deepcopy(snapshot)
        return self._observe()

    def close(self) -> None:
        self.state = {}

    # handy for tests/demos: a hand pose looking down at a world point
    @staticmethod
    def downward_pose(p: np.ndarray, yaw: float = 0.0) -> np.ndarray:
        return make_T(rot_z(yaw) @ rot_x(np.pi), p)
