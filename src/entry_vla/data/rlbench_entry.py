"""Single-arm entry-pose samples from RLBench episodes + contact phase split (scripts/contact_phase_split.py).

Label: the entry (manipulation-start) pose is the measured TCP pose at ``split_frame`` of
``contact/contact.json`` (``--pre-frames`` before the first end-effector contact). Episodes whose split
is undefined (contact within the first pre-frames) are skipped and reported.

One sample per query frame t < split_frame (free-motion phase, pre-travel observation):
  points (N, 6)  world xyz (m) + rgb in [0, 1] fused from the static exo cameras at frame t,
                 cropped to WORKSPACE, randomly subsampled (robot included; no privileged masks)
  state  (17,)   current TCP position (3) + rotation 6D (6) + gripper open (1) + 7 joint positions
  text   instruction (variation_descriptions[0])
  target T_WE at split_frame (4x4) + gripper open at split_frame
Positions stay in the RLBench world frame (metres); quaternions from RLBench are xyzw.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from ..geometry.camera import backproject_to_world
from ..geometry.rotations import matrix_to_rot6d
from ..perception.rlbench_scene import EXO_CAMERAS, load_cam, tcp_T
from .peract2_raw import demo_observations, load_pickle

WORKSPACE_MIN = np.array([-0.6, -0.8, 0.70])
WORKSPACE_MAX = np.array([1.2, 0.8, 1.90])
STATE_DIM = 17


def scene_points(ep: str, obs_t, t: int, n_points: int, rng: np.random.Generator,
                 cameras=EXO_CAMERAS) -> np.ndarray:
    pts, cols = [], []
    for cam in cameras:
        c, _ = load_cam(ep, obs_t.misc, cam, t)
        P = backproject_to_world(c.depth, c.K, c.T_WC, c.depth_valid)
        pts.append(P)
        cols.append(c.rgb[c.depth_valid].astype(np.float32) / 255.0)
    P, C = np.concatenate(pts), np.concatenate(cols)
    keep = np.all((P >= WORKSPACE_MIN) & (P <= WORKSPACE_MAX), axis=1)
    P, C = P[keep], C[keep]
    idx = rng.choice(len(P), n_points, replace=len(P) < n_points)
    return np.concatenate([P[idx], C[idx]], 1).astype(np.float32)


def robot_state(o) -> np.ndarray:
    T = tcp_T(o)
    return np.concatenate([T[:3, 3], matrix_to_rot6d(T[:3, :3]), [float(o.gripper_open)],
                           np.asarray(o.joint_positions, dtype=np.float64)[:7]]).astype(np.float32)


def episode_samples(raw_ep: str, contact_json: str, n_points: int = 4096, stride: int = 5,
                    max_queries: int = 12, min_gap: int = 1, seed: int = 0) -> Optional[Dict[str, np.ndarray]]:
    """Stacked samples of one episode, or None when the episode has no defined split."""
    j = json.load(open(contact_json))
    split = j.get("split_frame")
    if split is None or split < min_gap:
        return None
    obs = demo_observations(load_pickle(os.path.join(raw_ep, "low_dim_obs.pkl")))
    rng = np.random.default_rng(seed)
    queries = list(range(0, split - min_gap + 1, stride))
    if len(queries) > max_queries:
        queries = sorted(rng.choice(queries, max_queries, replace=False).tolist())
    T_entry = tcp_T(obs[split])
    instr = load_pickle(os.path.join(raw_ep, "variation_descriptions.pkl"))[0]
    return {"points": np.stack([scene_points(raw_ep, obs[t], t, n_points, rng) for t in queries]),
            "state": np.stack([robot_state(obs[t]) for t in queries]),
            "query_frame": np.array(queries, np.int32),
            "tgt_T": np.repeat(T_entry[None].astype(np.float32), len(queries), 0),
            "tgt_grip_open": np.full(len(queries), float(obs[split].gripper_open), np.float32),
            "split_frame": np.int32(split), "contact_onset": np.int32(j["first_contact_onset"]),
            "instruction": np.array(instr)}


def list_contact_episodes(out_root: str, raw_root: str) -> List[Dict[str, str]]:
    """Processed episodes <out_root>/<task>/variation<V>/seed<S>/episode<N>/contact/contact.json with the
    matching raw episode <raw_root>/<task>/variation<V>/seed<S>/episodes/episode<N>."""
    rows = []
    for task in sorted(os.listdir(out_root)):
        tdir = os.path.join(out_root, task)
        if not os.path.isdir(tdir):
            continue
        for var in sorted(os.listdir(tdir)):
            for seed in sorted(os.listdir(os.path.join(tdir, var)), key=lambda s: int(s[4:]) if s[4:].isdigit() else 0):
                sdir = os.path.join(tdir, var, seed)
                for ep in sorted(os.listdir(sdir)):
                    cj = os.path.join(sdir, ep, "contact", "contact.json")
                    raw = os.path.join(raw_root, task, var, seed, "episodes", ep)
                    if os.path.exists(cj) and os.path.exists(os.path.join(raw, "low_dim_obs.pkl")):
                        rows.append({"task": task, "variation": var, "seed": seed, "episode": ep,
                                     "contact_json": cj, "raw": raw})
    return rows


def seed_split(seed_name: str, val: List[int], test: List[int]) -> str:
    s = int(seed_name.replace("seed", ""))
    return "test" if s in test else "val" if s in val else "train"
