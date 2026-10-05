"""Supervised entry-proposal samples from exported episodes + skill-start annotations.

Leakage rules enforced here:
  * query frames are strictly before the annotated start (pre-travel), and only
    data at the query frame is used: no future images, poses or actions;
  * point clouds come from GLOBAL (non-moving) cameras unless explicitly allowed;
  * geometry inputs must come from a source available at test time and are
    labelled (``oracle:<key>`` = oracle condition). The annotation's region
    point (TCP at contact onset) is future information and is never an input;
  * ``pose_frame=region_point`` requires such a geometry source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..geometry.camera import backproject
from ..geometry.rotations import rot_z
from ..geometry.transforms import apply_T, invert_T, make_T
from ..policies.cartesian_chunk import encode_state
from .episode import Episode

ARMS = ("left", "right")
ROLES = ("approach", "holding")


@dataclass
class EntrySampleConfig:
    label: str = "bimanual_local_skill"
    cameras: List[str] = field(default_factory=list)       # empty = all non-moving cameras
    allow_moving_cameras: bool = False                       # current (query-time) wrist views only
    n_points: int = 1024
    workspace_min: Sequence[float] = (-1.0, -1.0, -0.05)     # crop in W (m)
    workspace_max: Sequence[float] = (1.5, 1.0, 1.0)
    query_stride: int = 5
    min_gap_frames: int = 5                                  # query frames end this many frames before start
    max_queries_per_episode: int = 8
    geometry_source: Optional[str] = None                    # e.g. "oracle:handles" (needs .left/.right keys)
    pose_frame: str = "W"                                     # "W" | "region_point"
    include_joints: bool = True


def region_frame(points: Dict[str, np.ndarray]) -> np.ndarray:
    c = 0.5 * (points["left"] + points["right"])
    d = points["left"] - points["right"]
    return make_T(rot_z(float(np.arctan2(d[1], d[0])) - np.pi / 2), c)


def geometry_at(ep: Episode, t: int, source: str) -> Dict[str, np.ndarray]:
    if not source.startswith("oracle:"):
        raise ValueError(f"unsupported geometry source {source!r}")
    key = source.split(":", 1)[1]
    return {a: np.asarray(ep.oracle[f"{key}.{a}"])[t] for a in ARMS}


def point_cloud(ep: Episode, t: int, cfg: EntrySampleConfig, rng: np.random.Generator):
    cams = cfg.cameras or [c for c, m in ep.meta["cameras"].items() if not m.get("moving")]
    pts, cols = [], []
    for c in cams:
        if ep.meta["cameras"][c].get("moving") and not cfg.allow_moving_cameras:
            raise ValueError(f"camera {c} moves with the arm; set allow_moving_cameras for query-time views")
        cam = ep.cameras[c]
        if "depth" not in cam:
            raise KeyError(f"{ep.episode_id}: camera {c} has no depth (RGB-D input required)")
        T_WC = cam["T_WC"][t] if cam["T_WC"].ndim == 3 else cam["T_WC"]
        m = cam["depth_valid"][t] if "depth_valid" in cam else cam["depth"][t] > 0
        P = apply_T(T_WC, backproject(cam["depth"][t], cam["K"], m))
        pts.append(P)
        cols.append(cam["rgb"][t][m].astype(np.float32) / 255.0 if "rgb" in cam else np.zeros((len(P), 3), np.float32))
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    C = np.concatenate(cols) if cols else np.zeros((0, 3), np.float32)
    lo, hi = np.asarray(cfg.workspace_min), np.asarray(cfg.workspace_max)
    keep = np.all((P >= lo) & (P <= hi), axis=1)
    P, C = P[keep], C[keep]
    out = np.zeros((cfg.n_points, 6), np.float32)
    mask = np.zeros(cfg.n_points, bool)
    if len(P):
        idx = rng.choice(len(P), size=min(cfg.n_points, len(P)), replace=False)
        out[:len(idx), :3], out[:len(idx), 3:] = P[idx], C[idx]
        mask[:len(idx)] = True
    return out, mask


def start_annotation(ep: Episode, label: str) -> Optional[Dict[str, Any]]:
    for s in ep.annotations.get("skill_starts", []):
        if s.get("label") == label and s.get("status") in ("proposed", "manual"):
            return s
    return None


def make_samples(ep: Episode, cfg: EntrySampleConfig, text_emb: np.ndarray, seed: int = 0) -> List[Dict[str, Any]]:
    ann = start_annotation(ep, cfg.label)
    if ann is None:
        return []
    if cfg.pose_frame == "region_point" and not cfg.geometry_source:
        raise ValueError("pose_frame=region_point needs a test-time geometry_source (labelled)")
    start = int(ann["proposed_start"])
    last_q = start - cfg.min_gap_frames
    if last_q < 0:
        return []
    qs = list(range(0, last_q + 1, cfg.query_stride))[-cfg.max_queries_per_episode:]
    rng = np.random.default_rng(seed)
    tgt_T = {a: ep.lowdim[f"{a}.T_WE"][start] for a in ARMS}
    roles = ann.get("roles", {a: "approach" for a in ARMS})
    grip_open = {a: float(ep.lowdim[f"{a}.gripper_width"][start] >= 0.03) for a in ARMS}
    samples = []
    for t in qs:
        pts, mask = point_cloud(ep, t, cfg, rng)
        geom = np.zeros(6, np.float32)
        T_FW = np.eye(4)              # pose frame F in W
        if cfg.geometry_source:
            g = geometry_at(ep, t, cfg.geometry_source)
            if cfg.pose_frame == "region_point":
                T_FW = region_frame(g)
            T_WF_inv = invert_T(T_FW)
            geom = np.concatenate([apply_T(T_WF_inv, g[a]) for a in ARMS]).astype(np.float32)
        T_inv = invert_T(T_FW)
        pts_f = pts.copy()
        pts_f[mask, :3] = apply_T(T_inv, pts[mask, :3])
        Tl, Tr = T_inv @ ep.lowdim["left.T_WE"][t], T_inv @ ep.lowdim["right.T_WE"][t]
        state = encode_state(Tl, Tr, ep.lowdim["left.gripper_width"][t], ep.lowdim["right.gripper_width"][t])
        if cfg.include_joints:
            state = np.concatenate([state, ep.lowdim["left.q"][t], ep.lowdim["right.q"][t]]).astype(np.float32)
        tgt_F = {a: T_inv @ tgt_T[a] for a in ARMS}
        samples.append({
            "episode_id": ep.episode_id, "query_frame": t, "start_frame": start,
            "points": pts_f, "points_mask": mask, "text": text_emb.astype(np.float32), "state": state,
            "geometry": geom,
            "tgt_pos": np.stack([tgt_F[a][:3, 3] for a in ARMS]).astype(np.float32),
            "tgt_rot": np.stack([tgt_F[a][:3, :3] for a in ARMS]).astype(np.float32),
            "tgt_grip_open": np.array([grip_open[a] for a in ARMS], np.float32),
            "tgt_role": np.array([ROLES.index(roles.get(a, "approach")) for a in ARMS], np.int64),
            "T_FW": T_FW.astype(np.float32),
            "input_provenance": {"geometry_source": cfg.geometry_source or "none", "pose_frame": cfg.pose_frame,
                                 "cameras": cfg.cameras or "non_moving", "oracle_condition": bool(cfg.geometry_source)},
        })
    return samples


def collate(samples: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    keys = ("points", "points_mask", "text", "state", "geometry", "tgt_pos", "tgt_rot", "tgt_grip_open", "tgt_role", "T_FW")
    return {k: np.stack([s[k] for s in samples]) for k in keys}
