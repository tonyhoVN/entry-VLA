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
    ap.add_argument("--episode", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", nargs="+", default=["front", "over_shoulder_left"], help="exo cameras")
    ap.add_argument("--frame", type=int, default=0, help="frame used for detection/segmentation (object static)")
    ap.add_argument("--target-slice", default="0:3", help="true object position in task_low_dim_state (reference only); 'none' skips the reference")
    ap.add_argument("--radius", type=float, default=0.20)
    ap.add_argument("--segmenter", default="sam2.1")
    ap.add_argument("--vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--erode", type=int, default=1)
    ap.add_argument("--no-phase-videos", action="store_true", help="skip transition/manipulation phase videos")
    ap.add_argument("--video-fps", type=int, default=20)
    ap.add_argument("--episodes-root", default=None,
                    help="process every episode*/ under this dir (models loaded once); --out is then a root")
    ap.add_argument("--max-episodes", type=int, default=None)
    a = ap.parse_args()
    if (a.episode is None) == (a.episodes_root is None):
        raise SystemExit("give exactly one of --episode or --episodes-root")

    from entry_vla.perception.grounding import Qwen3VLGrounder
    from entry_vla.perception.segmentation import make_segmenter
    t0 = time.time()
    grounder = Qwen3VLGrounder(a.vlm, device=a.device)
    segmenter = make_segmenter(a.segmenter, device=a.device)
    t_load = time.time() - t0

    if a.episode:
        s = run_episode(a, a.episode, a.out, grounder, segmenter, t_load)
        print(json.dumps({k: s[k] for k in ("pred_start_at_baseline", "ref_start_at_baseline", "radius_sweep",
                                             "cloud", "other_references")}, indent=1))
        return
    eps = sorted((d for d in os.listdir(a.episodes_root) if d.startswith("episode")), key=lambda d: int(d[7:]))
    results = []
    for d in eps[:a.max_episodes]:
        out = os.path.join(a.out, d, "perception")
        s = run_episode(a, os.path.join(a.episodes_root, d), out, grounder, segmenter, t_load)
        results.append((d, s))
        print(d, "pred", s["pred_start_at_baseline"], "ref", s["ref_start_at_baseline"],
              "stop", s["other_references"].get("stopped_handoff_frame"), flush=True)
    write_batch_summary(a, results)


def write_batch_summary(a, results):
    rows = []
    for d, s in results:
        r = {"episode": d, "num_frames": s["num_frames"],
             "stopped_handoff_frame": s["other_references"].get("stopped_handoff_frame"),
             "contact_frame": s["other_references"].get("contact_frame"),
             "centroid_error_mm": None if s["cloud"]["centroid_error_m"] is None else round(1000 * s["cloud"]["centroid_error_m"], 2),
             "dist_err_mean_mm": None if s["cloud"]["distance_abs_error_m"]["mean"] is None
             else round(1000 * s["cloud"]["distance_abs_error_m"]["mean"], 2)}
        for cam, v in s["per_camera"].items():
            r[f"iou_{cam}"] = None if v["mask_iou_vs_reference"] is None else round(v["mask_iou_vs_reference"], 3)
            r[f"label_{cam}"] = v["detection"]["label"] if v["detection"] else None
        for row in s["radius_sweep"]:
            k = f"{row['radius_m']:.2f}"
            r[f"pred_{k}"], r[f"ref_{k}"], r[f"err_{k}"] = row["pred_start"], row["ref_start"], row["error_frames"]
        rows.append(r)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "batch_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    agg = {"episodes": len(rows), "per_radius": {}}
    for rad in RADII:
        k = f"{rad:.2f}"
        errs = [r[f"err_{k}"] for r in rows if r[f"err_{k}"] is not None]
        missing = sum(1 for r in rows if r[f"err_{k}"] is None)
        agg["per_radius"][k] = {
            "n_compared": len(errs), "n_missing": missing,
            "exact_match_rate": float(np.mean([e == 0 for e in errs])) if errs else None,
            "mean_abs_error_frames": float(np.mean(np.abs(errs))) if errs else None,
            "max_abs_error_frames": int(np.max(np.abs(errs))) if errs else None,
            "ref_start_minus_stop_mean": float(np.mean([r[f"ref_{k}"] - r["stopped_handoff_frame"] for r in rows
                                                        if r[f"ref_{k}"] is not None and r["stopped_handoff_frame"] is not None]))
            if any(r["stopped_handoff_frame"] is not None for r in rows) else None}
    for key in [k for k in rows[0] if k.startswith("iou_")] + ["centroid_error_mm", "dist_err_mean_mm"]:
        vals = [r[key] for r in rows if r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        agg[key] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))} if vals else None
    with open(os.path.join(a.out, "batch_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=1))


def run_episode(a, episode, out, grounder, segmenter, t_load):
    os.makedirs(out, exist_ok=True)
    obs = demo_observations(load_pickle(os.path.join(episode, "low_dim_obs.pkl")))
    instruction = load_pickle(os.path.join(episode, "variation_descriptions.pkl"))[0]
    o0 = obs[a.frame]
    has_ref = a.target_slice.lower() != "none"
    if has_ref:
        lo, hi = (int(x) for x in a.target_slice.split(":"))
        true_pos = task_state(o0)[lo:hi]
    tcp = np.array([np.asarray(o.gripper_pose[:3], dtype=np.float64) for o in obs])

    pred_pts, ref_pts, per_cam = [], [], {}
    for cam in a.cameras:
        c, mask_rgb = load_cam(episode, o0, cam, a.frame)
        t1 = time.time()
        det = grounder.locate(c.rgb, instruction)
        pred = segmenter.segment(c.rgb, det.bbox_xyxy) if det is not None else np.zeros(c.depth.shape, bool)
        t_inf = time.time() - t1
        if has_ref:
            uv, _ = project_world(true_pos[None], c.K, c.T_WC)
            ref, ref_ids, ref_rejected = oracle_mask(mask_rgb, uv[0], c, true_pos, a.erode)
        else:
            ref, ref_ids, ref_rejected = np.zeros_like(pred), [], []
        p = object_points(c.depth, c.depth_valid, pred, c.K, c.T_WC, a.erode)
        r = object_points(c.depth, c.depth_valid, ref, c.K, c.T_WC, a.erode)
        pred_pts.append(p)
        ref_pts.append(r)
        overlay(c.rgb, pred, ref, None if det is None else det.bbox_xyxy, os.path.join(out, f"seg_{cam}.png"))
        Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(out, f"mask_pred_{cam}.png"))
        per_cam[cam] = {
            "detection": None if det is None else {"label": det.label, "bbox_xyxy_px": det.bbox_xyxy.round(1).tolist(),
                                                   "raw": det.raw_text},
            "pred_mask_px": int(pred.sum()), "ref_mask_px": int(ref.sum()), "ref_object_ids": ref_ids, "ref_rejected_ids": ref_rejected,
            "mask_iou_vs_reference": iou(pred, ref) if has_ref else None, "pred_points": len(p), "ref_points": len(r),
            "inference_s": round(t_inf, 2)}

    pred_cloud = remove_outliers(np.concatenate(pred_pts)) if pred_pts else np.zeros((0, 3))
    ref_cloud = np.concatenate(ref_pts)
    if has_ref and len(ref_cloud) == 0:
        raise RuntimeError(f"{episode}: no reference object pixels in {a.cameras}; check --target-slice/--cameras")
    np.save(os.path.join(out, "object_points_pred.npy"), pred_cloud)
    np.save(os.path.join(out, "object_points_ref.npy"), ref_cloud)

    d_pred = distance_to_points(tcp, pred_cloud)
    nan = np.full(len(obs), np.nan)
    d_ref = distance_to_points(tcp, ref_cloud) if has_ref else nan
    d_center = np.linalg.norm(tcp - true_pos, axis=1) if has_ref else nan
    with open(os.path.join(out, "distances.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "tcp_dist_pred_m", "tcp_dist_ref_m", "tcp_dist_true_center_m"])
        for i in range(len(obs)):
            w.writerow([i, round(d_pred[i], 5), round(d_ref[i], 5), round(d_center[i], 5)])

    sweep_pred = radius_sweep(d_pred, RADII)
    sweep_ref = radius_sweep(d_ref, RADII) if has_ref else {k: None for k in sweep_pred}
    rows = []
    for k in sweep_pred:
        p, r = sweep_pred[k], sweep_ref[k]
        rows.append({"radius_m": float(k), "pred_start": p, "ref_start": r,
                     "error_frames": None if p is None or r is None else p - r})
    cloud_err = {"distance_abs_error_m": {"mean": None, "max": None}, "centroid_error_m": None} if not has_ref else {
        "pred_centroid_W": pred_cloud.mean(0).round(4).tolist() if len(pred_cloud) else None,
        "ref_centroid_W": ref_cloud.mean(0).round(4).tolist(),
        "centroid_error_m": float(np.linalg.norm(pred_cloud.mean(0) - ref_cloud.mean(0))) if len(pred_cloud) else None,
        "pred_points_within_1cm_of_ref": float((distance_to_points(pred_cloud, ref_cloud) <= 0.01).mean())
        if len(pred_cloud) else None,
        "distance_abs_error_m": {"mean": float(np.abs(d_pred - d_ref).mean()), "max": float(np.abs(d_pred - d_ref).max())}}
    rule_path = os.path.join(os.path.dirname(out.rstrip("/")), "transition.json")
    refs = {}
    if os.path.exists(rule_path):
        with open(rule_path) as f:
            t = json.load(f)
        refs = {"stopped_handoff_frame": t["entry_frame"], "contact_frame": t["contact_frame"]}
    # task-agnostic state events (always available): gripper toggles and arm stops
    speed = np.array([np.linalg.norm(o.joint_velocities) for o in obs])
    grip = np.array([float(o.gripper_open) for o in obs])
    toggles = [int(i) for i in range(1, len(obs)) if grip[i] != grip[i - 1]]
    stops = [int(i) for i in range(1, len(obs) - 1)
             if speed[i] < 0.05 and speed[i] <= speed[i - 1] and speed[i] <= speed[i + 1]]
    refs["gripper_toggle_frames"] = toggles
    refs["arm_stop_frames"] = stops
    summary = {
        "format": "entry_vla.perception_phase_split/1", "episode": os.path.abspath(episode),
        "instruction": instruction, "num_frames": len(obs), "cameras": a.cameras, "perception_frame": a.frame,
        "vlm": a.vlm, "segmenter": a.segmenter, "model_load_s": round(t_load, 1),
        "baseline_radius_m": a.radius,
        "pred_start_at_baseline": sweep_pred[f"{a.radius:.2f}"], "ref_start_at_baseline": sweep_ref[f"{a.radius:.2f}"],
        "radius_sweep": rows, "per_camera": per_cam, "cloud": cloud_err, "other_references": refs,
        "has_reference": has_ref,
        "reference_note": "reference = same radius rule on the simulator's true object surface (mask ids + depth); "
                          "privileged and not a human phase label"}
    with open(os.path.join(out, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if not a.no_phase_videos:
        write_phase_videos(episode, out, summary["pred_start_at_baseline"], len(obs), a.radius, a.video_fps)
    return summary


def write_phase_videos(episode, out, start, n, radius, fps):
    """transition_phase.mp4 = frames [0, start), manipulation_phase.mp4 = [start, n): all cameras,
    RGB row over depth row (detect_transition_rlbench.tile). No split -> nothing written."""
    if start is None:
        return
    sys.path.insert(0, os.path.dirname(__file__))
    from detect_transition_rlbench import tile, write_video
    tag = f"R={radius:.2f} m"
    if start > 0:
        write_video(os.path.join(out, "transition_phase.mp4"),
                    [tile(episode, i, f"transition ({tag})", 1) for i in range(0, start)], fps)
    write_video(os.path.join(out, "manipulation_phase.mp4"),
                [tile(episode, i, f"manipulation ({tag})", 1) for i in range(start, n)], fps)



if __name__ == "__main__":
    main()
