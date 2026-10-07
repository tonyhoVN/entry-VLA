#!/usr/bin/env python3
"""Perception-based manipulation-start detection for one single-arm RLBench episode,
compared against a privileged simulator reference.

Method (README "Proposed method for decomposed skills"):
  1. Qwen3-VL locates the task's interaction object in the RGB of each exo camera (frame 0).
  2. A promptable segmenter (SAM 2.1; SAM 3 is gated, see perception/segmentation.py)
     turns each box into a mask.
  3. Masked depth is back-projected and fused into a world-frame object cloud; the distance
     field is the nearest-surface-point distance.
  4. Manipulation starts at the first frame whose end-effector (TCP from proprioception)
     distance is <= R (R swept, default baseline 0.20 m).

Reference (privileged, never given to the method): the simulator's own mask image gives the
true object pixels (object ids at the projected true object position), back-projected to a
true surface cloud. Reference start(R) = first frame with true TCP-surface distance <= R. This
isolates perception error; it is NOT a human phase label (the dataset has none).

  python scripts/perception_phase_split.py \
      --episode data/raw/rlbench_single/push_button/variation4/episodes/episode0 \
      --out data/entry_process/push_button/variation4/episode0/perception
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from entry_vla.data.peract2_raw import demo_observations, load_pickle  # noqa: E402
from entry_vla.geometry.camera import project_world  # noqa: E402
from entry_vla.perception.distance import (distance_to_points, object_points, radius_sweep,  # noqa: E402
                                           remove_outliers)
from entry_vla.sim.peract2 import camera_from_misc  # noqa: E402

RADII = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]


def task_state(o):
    s = o.task_low_dim_state
    return np.asarray(s[0] if isinstance(s, tuple) else s, dtype=np.float64).reshape(-1)


def load_cam(ep, o, cam, i):
    rgb = np.asarray(Image.open(os.path.join(ep, f"{cam}_rgb", f"rgb_{i:04d}.png")).convert("RGB"))
    d = np.asarray(Image.open(os.path.join(ep, f"{cam}_depth", f"depth_{i:04d}.png")).convert("RGB"))
    mask_rgb = np.asarray(Image.open(os.path.join(ep, f"{cam}_mask", f"mask_{i:04d}.png")).convert("RGB"))
    return camera_from_misc(cam, o.misc, rgb, d, "rgb24_linear_buffer"), mask_rgb


def oracle_mask(mask_rgb, uv, cam, true_pos, erode, window=7, max_dist=0.08):
    """True object pixels: simulator object ids (mask colours) found around the projected true
    object position, kept only if that id's own back-projected points lie within ``max_dist``
    of the true position (drops table, robot and neighbouring objects caught by the window).
    Returns (bool mask, kept ids, rejected ids)."""
    ids = mask_rgb.astype(np.int64) @ np.array([1, 256, 65536])
    H, W = ids.shape
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= u < W and 0 <= v < H):
        return np.zeros((H, W), bool), [], []
    win = ids[max(0, v - window):v + window + 1, max(0, u - window):u + window + 1]
    keep, reject = [], []
    for x in (int(x) for x in np.unique(win)):
        pts = object_points(cam.depth, cam.depth_valid, ids == x, cam.K, cam.T_WC, erode)
        if len(pts) and np.linalg.norm(np.median(pts, axis=0) - true_pos) <= max_dist:
            keep.append(x)
        else:
            reject.append(x)
    return np.isin(ids, keep), keep, reject


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def overlay(rgb, pred, ref, box, path):
    import cv2
    img = rgb.copy().astype(np.float32)
    img[pred] = 0.5 * img[pred] + 0.5 * np.array([0, 255, 0])
    img = img.astype(np.uint8)
    cnt, _ = cv2.findContours(ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnt, -1, (255, 0, 255), 1)
    if box is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
    Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", nargs="+", default=["front", "over_shoulder_left"], help="exo cameras")
    ap.add_argument("--frame", type=int, default=0, help="frame used for detection/segmentation (object static)")
    ap.add_argument("--target-slice", default="0:3", help="true object position in task_low_dim_state (reference only)")
    ap.add_argument("--radius", type=float, default=0.20)
    ap.add_argument("--segmenter", default="sam2.1")
    ap.add_argument("--vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--erode", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    obs = demo_observations(load_pickle(os.path.join(a.episode, "low_dim_obs.pkl")))
    instruction = load_pickle(os.path.join(a.episode, "variation_descriptions.pkl"))[0]
    o0 = obs[a.frame]
    lo, hi = (int(x) for x in a.target_slice.split(":"))
    true_pos = task_state(o0)[lo:hi]
    tcp = np.array([np.asarray(o.gripper_pose[:3], dtype=np.float64) for o in obs])

    from entry_vla.perception.grounding import Qwen3VLGrounder
    from entry_vla.perception.segmentation import make_segmenter
    t0 = time.time()
    grounder = Qwen3VLGrounder(a.vlm, device=a.device)
    segmenter = make_segmenter(a.segmenter, device=a.device)
    t_load = time.time() - t0

    pred_pts, ref_pts, per_cam = [], [], {}
    for cam in a.cameras:
        c, mask_rgb = load_cam(a.episode, o0, cam, a.frame)
        t1 = time.time()
        det = grounder.locate(c.rgb, instruction)
        pred = segmenter.segment(c.rgb, det.bbox_xyxy) if det is not None else np.zeros(c.depth.shape, bool)
        t_inf = time.time() - t1
        uv, _ = project_world(true_pos[None], c.K, c.T_WC)
        ref, ref_ids, ref_rejected = oracle_mask(mask_rgb, uv[0], c, true_pos, a.erode)
        p = object_points(c.depth, c.depth_valid, pred, c.K, c.T_WC, a.erode)
        r = object_points(c.depth, c.depth_valid, ref, c.K, c.T_WC, a.erode)
        pred_pts.append(p)
        ref_pts.append(r)
        overlay(c.rgb, pred, ref, None if det is None else det.bbox_xyxy, os.path.join(a.out, f"seg_{cam}.png"))
        Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(a.out, f"mask_pred_{cam}.png"))
        per_cam[cam] = {
            "detection": None if det is None else {"label": det.label, "bbox_xyxy_px": det.bbox_xyxy.round(1).tolist(),
                                                   "raw": det.raw_text},
            "pred_mask_px": int(pred.sum()), "ref_mask_px": int(ref.sum()), "ref_object_ids": ref_ids, "ref_rejected_ids": ref_rejected,
            "mask_iou_vs_reference": iou(pred, ref), "pred_points": len(p), "ref_points": len(r),
            "inference_s": round(t_inf, 2)}

    pred_cloud = remove_outliers(np.concatenate(pred_pts)) if pred_pts else np.zeros((0, 3))
    ref_cloud = np.concatenate(ref_pts)
    np.save(os.path.join(a.out, "object_points_pred.npy"), pred_cloud)
    np.save(os.path.join(a.out, "object_points_ref.npy"), ref_cloud)

    d_pred = distance_to_points(tcp, pred_cloud)
    d_ref = distance_to_points(tcp, ref_cloud)
    d_center = np.linalg.norm(tcp - true_pos, axis=1)
    with open(os.path.join(a.out, "distances.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "tcp_dist_pred_m", "tcp_dist_ref_m", "tcp_dist_true_center_m"])
        for i in range(len(obs)):
            w.writerow([i, round(d_pred[i], 5), round(d_ref[i], 5), round(d_center[i], 5)])

    sweep_pred, sweep_ref = radius_sweep(d_pred, RADII), radius_sweep(d_ref, RADII)
    rows = []
    for k in sweep_pred:
        p, r = sweep_pred[k], sweep_ref[k]
        rows.append({"radius_m": float(k), "pred_start": p, "ref_start": r,
                     "error_frames": None if p is None or r is None else p - r})
    cloud_err = {
        "pred_centroid_W": pred_cloud.mean(0).round(4).tolist() if len(pred_cloud) else None,
        "ref_centroid_W": ref_cloud.mean(0).round(4).tolist(),
        "centroid_error_m": float(np.linalg.norm(pred_cloud.mean(0) - ref_cloud.mean(0))) if len(pred_cloud) else None,
        "pred_points_within_1cm_of_ref": float((distance_to_points(pred_cloud, ref_cloud) <= 0.01).mean())
        if len(pred_cloud) else None,
        "distance_abs_error_m": {"mean": float(np.abs(d_pred - d_ref).mean()), "max": float(np.abs(d_pred - d_ref).max())}}
    rule_path = os.path.join(os.path.dirname(a.out.rstrip("/")), "transition.json")
    refs = {}
    if os.path.exists(rule_path):
        with open(rule_path) as f:
            t = json.load(f)
        refs = {"stopped_handoff_frame": t["entry_frame"], "contact_frame": t["contact_frame"]}
    summary = {
        "format": "entry_vla.perception_phase_split/1", "episode": os.path.abspath(a.episode),
        "instruction": instruction, "cameras": a.cameras, "perception_frame": a.frame,
        "vlm": a.vlm, "segmenter": a.segmenter, "model_load_s": round(t_load, 1),
        "baseline_radius_m": a.radius,
        "pred_start_at_baseline": sweep_pred[f"{a.radius:.2f}"], "ref_start_at_baseline": sweep_ref[f"{a.radius:.2f}"],
        "radius_sweep": rows, "per_camera": per_cam, "cloud": cloud_err, "other_references": refs,
        "reference_note": "reference = same radius rule on the simulator's true object surface (mask ids + depth); "
                          "privileged and not a human phase label"}
    with open(os.path.join(a.out, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("pred_start_at_baseline", "ref_start_at_baseline", "radius_sweep",
                                                "cloud", "other_references")}, indent=1))
    for cam, v in per_cam.items():
        print(cam, v["detection"] and v["detection"]["label"], v["detection"] and v["detection"]["bbox_xyxy_px"],
              "IoU %.3f" % v["mask_iou_vs_reference"], "ids", v["ref_object_ids"])


if __name__ == "__main__":
    main()
