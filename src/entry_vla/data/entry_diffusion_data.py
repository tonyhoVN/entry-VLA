"""Inputs/targets for the conditional entry-pose diffusion model (RLBench single arm).

Per episode (one sample, first frame t=0):
  object        the scene object touched at the first contact onset of the contact split
                (contact/contact.json): the non-robot, non-structure mask id nearest to the TCP at the onset,
                measured on frame-0 points. ORACLE detection: simulator mask ids (privileged);
                a learned detector (e.g. Qwen3-VL + SAM 2.1) can replace it later.
  obj_points    (N, 6) object points at frame 0 fused from the exo cameras (+ wrist when visible),
                world xyz - centroid c (canonical frame: translated only, axes parallel to world) + rgb
  centroid      c (3,) world, metres
  crops         (V, S, S, 3) uint8 bounding-box crops of the object at frame 0, one per camera where
                at least ``min_crop_px`` pixels are visible; ``crop_cams`` names them (wrist = ego)
  instruction   variation_descriptions[0]
  target        entry TCP pose at split_frame: p_C = p_W - c, R_C = R_W (quaternion xyzw from RLBench)
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

from ..geometry.camera import backproject_to_world
from ..perception.rlbench_scene import EXO_CAMERAS, load_cam, tcp_T
from .peract2_raw import demo_observations, load_pickle

EGO_CAMERAS = ("wrist",)
ALL_CAMERAS = EGO_CAMERAS + EXO_CAMERAS


def contact_object_ids(ep: str, obs, onset: int, robot_ids, cameras=ALL_CAMERAS, max_static_extent: float = 0.6,
                       min_pts: int = 20) -> Dict[int, float]:
    """{mask id: distance (m)} of the object touched at the contact onset: the non-robot id whose frame-0
    points are nearest to the TCP position at ``onset``. Frame 0 is used because objects have not moved
    yet and are not occluded by the hand. Large static structures (extent > max_static_extent in at least
    two axes: table, walls, floor) are excluded."""
    from scipy.spatial import cKDTree
    from ..perception.rlbench_scene import id_points
    p = tcp_T(obs[onset])[:3, 3]
    best = {}
    for i, P in id_points(ep, obs[0], 0, cameras).items():
        if i in robot_ids or len(P) < min_pts or (np.ptp(P, 0) > max_static_extent).sum() >= 2:
            continue
        best[int(i)] = float(cKDTree(P).query(p)[0])
    if not best:
        return {}
    i = min(best, key=best.get)
    return {i: best[i]}


def crop_box(mask: np.ndarray, pad: float = 0.15, min_side: int = 24):
    v, u = np.nonzero(mask)
    u0, u1, v0, v1 = u.min(), u.max() + 1, v.min(), v.max() + 1
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    side = max(u1 - u0, v1 - v0, min_side) * (1 + 2 * pad)
    H, W = mask.shape
    a0, a1 = int(max(0, cu - side / 2)), int(min(W, cu + side / 2))
    b0, b1 = int(max(0, cv - side / 2)), int(min(H, cv + side / 2))
    return a0, b0, a1, b1


def episode_inputs(raw_ep: str, contact_json: str, n_points: int = 2048, crop_size: int = 224,
                   min_crop_px: int = 30, seed: int = 0, max_object_dist: float = 0.15) -> Optional[Dict[str, np.ndarray]]:
    import cv2
    j = json.load(open(contact_json))
    split, onset = j.get("split_frame"), j.get("first_contact_onset")
    if split is None or onset is None:
        return None
    obs = demo_observations(load_pickle(os.path.join(raw_ep, "low_dim_obs.pkl")))
    obj = contact_object_ids(raw_ep, obs, onset, set(j["robot_ids"]))
    if not obj or min(obj.values()) > max_object_dist:   # touched part not visible at frame 0
        return None
    oid = np.asarray(sorted(obj))
    pts, cols, crops, crop_cams = [], [], [], []
    for cam in ALL_CAMERAS:
        c, ids = load_cam(raw_ep, obs[0].misc, cam, 0)
        m = np.isin(ids, oid)
        mv = m & c.depth_valid
        if mv.sum():
            pts.append(backproject_to_world(c.depth, c.K, c.T_WC, mv))
            cols.append(c.rgb[mv].astype(np.float32) / 255.0)
        if m.sum() >= min_crop_px:
            a0, b0, a1, b1 = crop_box(m)
            crops.append(cv2.resize(c.rgb[b0:b1, a0:a1], (crop_size, crop_size), interpolation=cv2.INTER_LINEAR))
            crop_cams.append(cam)
    if not pts or not crops:
        return None
    P, C = np.concatenate(pts), np.concatenate(cols)
    centroid = P.mean(0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), n_points, replace=len(P) < n_points)
    T_e = tcp_T(obs[split])
    return {"obj_points": np.concatenate([P[idx] - centroid, C[idx]], 1).astype(np.float32),
            "n_obj_points_raw": np.int32(len(P)), "centroid": centroid.astype(np.float32),
            "crops": np.stack(crops), "crop_cams": np.array(crop_cams),
            "object_ids": oid.astype(np.int32), "object_tcp_dist_m": np.array([obj[i] for i in oid], np.float32),
            "instruction": np.array(load_pickle(os.path.join(raw_ep, "variation_descriptions.pkl"))[0]),
            "tgt_pos_C": (T_e[:3, 3] - centroid).astype(np.float32), "tgt_rot": T_e[:3, :3].astype(np.float32),
            "tgt_grip_open": np.float32(obs[split].gripper_open), "split_frame": np.int32(split),
            "contact_onset": np.int32(onset)}
