"""Synthetic schema fixtures: scripted demonstrations on the MOCK backend.

Used by tests and for exercising pipelines before real data exists. Every
fixture is labelled ``provenance=synthetic_mock``; nothing derived from it
is evidence about DuoBench or PerAct2.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..sim.contracts import ARMS, BimanualTcpCommand, Observation
from ..sim.mock import MockBackend
from .episode import ActionSpec, Episode, new_meta


def _interp(T0: np.ndarray, T1: np.ndarray, n: int) -> List[np.ndarray]:
    from ..geometry.rotations import so3_exp, so3_log
    out = []
    dR = so3_log(T0[:3, :3].T @ T1[:3, :3])
    for s in np.linspace(0, 1, n + 1)[1:]:
        T = np.eye(4)
        T[:3, :3] = T0[:3, :3] @ so3_exp(s * dR)
        T[:3, 3] = (1 - s) * T0[:3, 3] + s * T1[:3, 3]
        out.append(T)
    return out


def scripted_plan(obs: Observation, n_approach: int = 15, n_descend: int = 10, n_close: int = 4,
                  n_lift: int = 8) -> List[BimanualTcpCommand]:
    h = obs.oracle["handles"]
    yaw = float(np.arctan2(h["left"][1] - h["right"][1], h["left"][0] - h["right"][0])) - np.pi / 2
    cmds: List[BimanualTcpCommand] = []
    pre = {a: MockBackend.downward_pose(h[a] + np.array([0, 0, 0.15]), yaw) for a in ARMS}
    grasp = {a: MockBackend.downward_pose(h[a] + np.array([0, 0, 0.0]), yaw) for a in ARMS}
    lift = {a: MockBackend.downward_pose(h[a] + np.array([0, 0, 0.12]), yaw) for a in ARMS}
    start = {a: obs.arms[a].T_WE for a in ARMS}
    seg = [(start, pre, n_approach, 1.0), (pre, grasp, n_descend, 1.0), (grasp, grasp, n_close, 0.0),
           (grasp, lift, n_lift, 0.0)]
    for T0, T1, n, g in seg:
        paths = {a: _interp(T0[a], T1[a], n) for a in ARMS}
        for k in range(n):
            cmds.append(BimanualTcpCommand({a: paths[a][k] for a in ARMS}, {a: g for a in ARMS}))
    return cmds


def record_mock_episode(seed: int, episode_id: Optional[str] = None, split: str = "train",
                        backend_cfg: Optional[Dict] = None, settle_steps: int = 3) -> Episode:
    be = MockBackend(dict(backend_cfg or {"name": "mock"}))
    be.make("mock_bimanual_reach")
    obs = be.reset(seed=seed)
    cmds = scripted_plan(obs)
    # hold the final command a few steps so tracking can finish
    cmds += [cmds[-1]] * settle_steps
    frames: List[Observation] = []
    for c in cmds:
        frames.append(obs)                 # obs[t] is recorded BEFORE command[t] is applied
        obs = be.step(c, obs).obs
    T = len(frames)
    lowdim: Dict[str, np.ndarray] = {
        "time": np.array([f.time for f in frames]),
        "step": np.array([f.step for f in frames]),
        "valid": np.ones(T, dtype=bool),
        "source_index": np.arange(T),
    }
    for a in ARMS:
        lowdim[f"{a}.T_WE"] = np.stack([f.arms[a].T_WE for f in frames])
        lowdim[f"{a}.q"] = np.stack([f.arms[a].q for f in frames])
        lowdim[f"{a}.dq"] = np.stack([f.arms[a].dq for f in frames])
        lowdim[f"{a}.gripper_width"] = np.array([f.arms[a].gripper_width for f in frames])
        lowdim[f"{a}.cmd_T_WE"] = np.stack([c.T_WE[a] for c in cmds])
        lowdim[f"{a}.cmd_gripper"] = np.array([c.gripper[a] for c in cmds])
    cams = {}
    cam_meta = {}
    for name in frames[0].cameras:
        cf = [f.cameras[name] for f in frames]
        c0 = cf[0]
        cams[name] = {
            "rgb": np.stack([c.rgb for c in cf]),
            "depth": np.stack([c.depth for c in cf]).astype(np.float32),
            "depth_valid": np.stack([c.depth_valid for c in cf]),
            "K": c0.K,
            "T_WC": np.stack([c.T_WC for c in cf]) if c0.moving else c0.T_WC,
        }
        cam_meta[name] = {"width": c0.width, "height": c0.height, "near": c0.near, "far": c0.far,
                          "depth_source_encoding": c0.depth_source_encoding, "moving": c0.moving,
                          "modality": "rgbd_sim"}
    spec = ActionSpec(key="cmd", kind="commanded_target", controller="mock absolute TCP (IK tracking)",
                      frame="W", alignment="cmd[t] applied after obs[t] was recorded")
    meta = new_meta(
        episode_id=episode_id or f"mock_seed{seed:05d}",
        backend="mock", task_id="mock_bimanual_reach", variation_id=0,
        instruction=frames[0].instruction, split=split, seed=seed,
        sim_timestep_s=be.dt, recording_rate_hz=1.0 / be.dt,
        provenance="synthetic_mock scripted demonstration (not a benchmark demonstration)",
        world_frame="mock_world",
        alignment={"obs_vs_action": "obs_before_action", "verified_by": "construction (mock)"},
        source={"scene_seed": seed, "source_demo_id": episode_id or f"mock_seed{seed:05d}"},
        cameras=cam_meta,
        gripper={"semantics": "width_m", "max_width_m": 0.08, "model": "mock parallel"},
        reset={"procedure": "reset(seed)", "snapshot": "supported (mock)"},
    )
    meta["action_specs"] = [spec.__dict__]
    oracle = {f"handles.{a}": np.stack([f.oracle["handles"][a] for f in frames]) for a in ARMS}
    oracle["stage"] = np.array([f.oracle["stage"] for f in frames])
    oracle["success"] = np.array([f.oracle["success"] for f in frames])
    oracle["region_center"] = np.stack([f.oracle["bar_center"] for f in frames])
    for a in ARMS:
        oracle[f"grasped.{a}"] = np.array([f.oracle["grasped"][a] for f in frames])
    be.close()
    return Episode(meta=meta, lowdim=lowdim, cameras=cams, annotations={}, oracle=oracle)
