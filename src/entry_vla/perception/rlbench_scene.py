"""Scene geometry helpers for single-arm RLBench episodes saved by scripts/rlbench_single_generator.py:
per-camera RGB-D + simulator mask ids, robot-link ids, end-effector (hand) points and environment points.

Mask images encode CoppeliaSim object handles as RGB; they are privileged simulator data, so anything
built on ``robot_ids`` is an oracle/annotation tool, not a deployable perception input.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from ..geometry.camera import backproject_to_world
from ..geometry.transforms import apply_T, invert_T, pose_to_T

EXO_CAMERAS = ("front", "overhead", "over_shoulder_left", "over_shoulder_right")
# Hand-sized box in the TCP frame (m): x thin, y = palm / finger opening, z = approach (palm behind TCP).
HAND_BOX = ((-0.05, 0.05), (-0.12, 0.12), (-0.12, 0.02))


def load_cam(ep: str, misc: Dict, cam: str, i: int):
    """(CameraFrame with metric depth, mask-id image) for camera ``cam`` at frame ``i``."""
    from PIL import Image
    from ..sim.peract2 import camera_from_misc
    rgb = np.asarray(Image.open(os.path.join(ep, f"{cam}_rgb", f"rgb_{i:04d}.png")).convert("RGB"))
    d = np.asarray(Image.open(os.path.join(ep, f"{cam}_depth", f"depth_{i:04d}.png")).convert("RGB"))
    m = np.asarray(Image.open(os.path.join(ep, f"{cam}_mask", f"mask_{i:04d}.png")).convert("RGB"))
    ids = m.astype(np.int64) @ np.array([1, 256, 65536])
    return camera_from_misc(cam, misc, rgb, d, "rgb24_linear_buffer"), ids


def tcp_T(obs) -> np.ndarray:
    gp = np.asarray(obs.gripper_pose, dtype=np.float64)
    return pose_to_T(gp[:3], gp[3:7], "xyzw")


def id_points(ep: str, obs_t, t: int, cameras: Sequence[str] = EXO_CAMERAS) -> Dict[int, np.ndarray]:
    """World points per mask id at frame ``t``, merged over cameras."""
    out: Dict[int, List[np.ndarray]] = {}
    for cam in cameras:
        c, ids = load_cam(ep, obs_t.misc, cam, t)
        for v in np.unique(ids[c.depth_valid]):
            out.setdefault(int(v), []).append(backproject_to_world(c.depth, c.K, c.T_WC, (ids == v) & c.depth_valid))
    return {v: np.concatenate(p) for v, p in out.items()}


def robot_ids(ep: str, obs: Sequence, frames: Iterable[int], ref: int = 0, move_thresh: float = 0.02,
              min_pts: int = 20, max_pts: int = 4000) -> Set[int]:
    """Mask ids that moved between frame ``ref`` and any of ``frames``: the median distance from an id's
    points at frame t to its own points at ``ref`` exceeds ``move_thresh`` (or it is visible only at t).
    Static objects - occluded by the arm or settling by millimetres - keep matching their own points.
    Choose ``frames`` before any object moves so that manipulated objects are not mistaken for links."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    P0 = id_points(ep, obs[ref], ref)
    trees = {v: cKDTree(p) for v, p in P0.items()}
    out: Set[int] = set()
    for t in frames:
        for v, PE in id_points(ep, obs[t], t).items():
            if v in out or len(PE) < min_pts:
                continue
            if v not in trees:
                out.add(v)
                continue
            if len(PE) > max_pts:
                PE = PE[rng.choice(len(PE), max_pts, replace=False)]
            if np.median(trees[v].query(PE)[0]) > move_thresh:
                out.add(v)
    return out


def in_box(points_W: np.ndarray, T_WE: np.ndarray, box=HAND_BOX) -> np.ndarray:
    Q = apply_T(invert_T(T_WE), points_W)
    return np.all([(Q[:, k] >= lo) & (Q[:, k] <= hi) for k, (lo, hi) in enumerate(box)], axis=0)


def hand_and_env_points(ep: str, obs_t, t: int, rid: Set[int], cameras: Sequence[str] = EXO_CAMERAS,
                        box=HAND_BOX) -> Tuple[np.ndarray, np.ndarray]:
    """(end-effector points, environment points) at frame t. End-effector = robot-id points inside the
    hand box of this frame's TCP; environment = all non-robot points (robot links outside the box dropped)."""
    T = tcp_T(obs_t)
    hand, env = [], []
    for cam in cameras:
        c, ids = load_cam(ep, obs_t.misc, cam, t)
        P = backproject_to_world(c.depth, c.K, c.T_WC, c.depth_valid)
        is_robot = np.isin(ids[c.depth_valid], list(rid))
        hand.append(P[is_robot][in_box(P[is_robot], T, box)])
        env.append(P[~is_robot])
    return np.concatenate(hand), np.concatenate(env)
