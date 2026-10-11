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
  (detect_object_frame0 / episode_inputs_detected: deployable variant with Qwen3-VL + SAM 2.1 masks)
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


# ---------------------------------------------------------------------------------------------------
# Learned detection (deployable): Qwen3-VL box + SAM 2.1 mask on frame 0 instead of simulator mask ids.
# ---------------------------------------------------------------------------------------------------
CLOUD_CAMERAS = ("front", "overhead")
CROP_CAMERAS = ("wrist", "front", "overhead")      # ego first, then exo


def object_cluster(P: np.ndarray, link_m: float = 0.02, voxel_m: float = 0.01, min_frac: float = 0.15) -> np.ndarray:
    """Mask of the object cluster: connected components of voxel centres linked within link_m; among the
    components holding >= min_frac of the points, the LOWEST one (5th-percentile z) is the table-top object.
    Mask leaks onto the robot (e.g. the gripper hovering above the object at frame 0) form separate,
    higher components; choosing by size alone picks them when the object is partly filtered."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    vox = np.floor(P / voxel_m).astype(np.int64)
    uniq, inv = np.unique(vox, axis=0, return_inverse=True)
    inv = inv.ravel()
    centres = (uniq + 0.5) * voxel_m
    pairs = cKDTree(centres).query_pairs(link_m, output_type="ndarray")
    n = len(centres)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    _, lab = connected_components(g, directed=False)
    plab = lab[inv]
    counts = np.bincount(plab)
    big = [k for k in np.flatnonzero(counts >= min_frac * len(P))] or [int(counts.argmax())]
    low = min(big, key=lambda k: np.percentile(P[plab == k, 2], 5))
    return plab == low


def detect_object_frame0(raw_ep: str, instruction: str, grounder, segmenter, cloud_cams=CLOUD_CAMERAS,
                         crop_cams=CROP_CAMERAS, n_points: int = 2048, crop_size: int = 224, erode_px: int = 2,
                         occlusion_m: float = 0.02, link_m: float = 0.02, seed: int = 0) -> Optional[Dict[str, np.ndarray]]:
    """Object inputs from frame 0 only (no future frames, no simulator masks): the instruction's target
    object is boxed by the VLM grounder and segmented by SAM in every camera of cloud_cams | crop_cams.
    Point cloud: eroded masks of cloud_cams back-projected with metric depth and fused in world; a point is
    kept only if every other cloud camera sees it inside its (dilated) object mask or cannot see it
    (occluded), which removes mask leaks such as the robot arm in one view; finally the lowest large connected
    cluster (points linked within link_m) is kept: the object rests on the table, leaks onto the arm are higher. Crops: padded boxes of crop_cams."""
    import cv2
    from scipy.ndimage import binary_erosion
    from scipy.ndimage import binary_dilation
    from ..geometry.camera import project_world
    obs0 = demo_observations(load_pickle(os.path.join(raw_ep, "low_dim_obs.pkl")))[0]
    pts, cols, crops, crop_names, dets, views = [], [], [], [], {}, []
    for cam in dict.fromkeys(tuple(crop_cams) + tuple(cloud_cams)):
        c, _ = load_cam(raw_ep, obs0.misc, cam, 0)
        det = grounder.locate(c.rgb, instruction)
        if det is None:
            dets[cam] = None
            continue
        box = det.bbox_xyxy
        mask = segmenter.segment(c.rgb, box)
        dets[cam] = {"label": det.label, "bbox_xyxy": [float(v) for v in box], "mask_px": int(mask.sum())}
        if cam in cloud_cams and mask.sum():
            m = binary_erosion(mask, iterations=erode_px) & c.depth_valid
            if m.sum():
                pts.append(backproject_to_world(c.depth, c.K, c.T_WC, m))
                cols.append(c.rgb[m].astype(np.float32) / 255.0)
                views.append((c, binary_dilation(mask, iterations=3)))
        if cam in crop_cams:
            x0, y0, x1, y1 = box
            pad = 0.15 * max(x1 - x0, y1 - y0, 24)
            H, W = c.rgb.shape[:2]
            a0, b0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
            a1, b1 = int(min(W, x1 + pad + 1)), int(min(H, y1 + pad + 1))
            crops.append(cv2.resize(c.rgb[b0:b1, a0:a1], (crop_size, crop_size), interpolation=cv2.INTER_LINEAR))
            crop_names.append(cam)
    if not pts or not crops:
        return None
    if len(views) >= 2:   # multi-view consistency: drop points another view sees outside its object mask
        for i in range(len(pts)):
            keep = np.ones(len(pts[i]), bool)
            for j, (c, mask_d) in enumerate(views):
                if j == i:
                    continue
                uv, z = project_world(pts[i], c.K, c.T_WC)
                u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
                inside = (z > 0) & (u >= 0) & (v >= 0) & (u < mask_d.shape[1]) & (v < mask_d.shape[0])
                ok = np.ones(len(z), bool)
                ui, vi = u[inside], v[inside]
                hidden = c.depth[vi, ui] < z[inside] - occlusion_m     # occluded in view j: cannot judge
                ok[inside] = mask_d[vi, ui] | hidden
                keep &= ok
            pts[i], cols[i] = pts[i][keep], cols[i][keep]
        dets["multiview_kept"] = [int(len(p)) for p in pts]
    P, C = np.concatenate(pts), np.concatenate(cols)
    if len(P) < 20:
        return None
    keep = object_cluster(P, link_m)
    dets["object_cluster_frac"] = float(keep.mean())
    P, C = P[keep], C[keep]
    centroid = P.mean(0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), n_points, replace=len(P) < n_points)
    return {"obj_points": np.concatenate([P[idx] - centroid, C[idx]], 1).astype(np.float32),
            "n_obj_points_raw": np.int32(len(P)), "centroid": centroid.astype(np.float32),
            "crops": np.stack(crops), "crop_cams": np.array(crop_names),
            "detections": np.array(json.dumps(dets)), "instruction": np.array(instruction)}


def episode_inputs_detected(raw_ep: str, contact_json: str, grounder, segmenter, **kw) -> Optional[Dict[str, np.ndarray]]:
    """Training sample with learned detection; the target comes from the contact split (label only)."""
    j = json.load(open(contact_json))
    split = j.get("split_frame")
    if split is None:
        return None
    instr = load_pickle(os.path.join(raw_ep, "variation_descriptions.pkl"))[0]
    s = detect_object_frame0(raw_ep, instr, grounder, segmenter, **kw)
    if s is None:
        return None
    obs = demo_observations(load_pickle(os.path.join(raw_ep, "low_dim_obs.pkl")))
    T_e = tcp_T(obs[split])
    s.update(tgt_pos_C=(T_e[:3, 3] - s["centroid"]).astype(np.float32), tgt_rot=T_e[:3, :3].astype(np.float32),
             tgt_grip_open=np.float32(obs[split].gripper_open), split_frame=np.int32(split),
             contact_onset=np.int32(j["first_contact_onset"]))
    return s
