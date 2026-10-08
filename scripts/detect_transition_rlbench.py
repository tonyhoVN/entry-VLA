#!/usr/bin/env python3
"""Detect the transition (planner travel) -> manipulation handoff frame in one
single-arm RLBench episode and write per-frame signals, a summary and two
camera-stacked videos.

Reads the episode with the stub unpickler (no CoppeliaSim needed), so it runs in
any env with numpy, PIL, cv2 and an ``ffmpeg`` binary. The logic is documented
in the generated README.md next to the outputs.

  python scripts/detect_transition_rlbench.py \
      --episode data/raw/rlbench_single/push_button/variation4/episodes/episode0 \
      --out data/entry_process/push_button/variation4/episode0
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from entry_vla.data.peract2_raw import demo_observations, load_pickle  # noqa: E402

CAMERAS = ["front", "over_shoulder_left", "over_shoulder_right", "overhead", "wrist"]
DEPTH_SCALE = 2 ** 24 - 1  # rlbench.backend.const.DEPTH_SCALE
RLBENCH_DT = 0.05  # assumed CoppeliaSim step per recorded frame (see docs/PLAN.md)


def task_state(o) -> np.ndarray:
    s = o.task_low_dim_state
    if isinstance(s, tuple):  # this fork stores (array,)
        s = s[0]
    return np.asarray(s, dtype=np.float64).reshape(-1)


def frame_signals(obs, target_slice):
    s0 = task_state(obs[0])
    rows = []
    for i, o in enumerate(obs):
        s = task_state(o)
        p = np.asarray(o.gripper_pose[:3], dtype=np.float64)
        b = s[target_slice]
        rows.append({
            "frame": i,
            "time_s": round(i * RLBENCH_DT, 3),
            "tcp_x": p[0], "tcp_y": p[1], "tcp_z": p[2],
            "dist_to_target_m": float(np.linalg.norm(p - b)),
            "xy_offset_m": float(np.linalg.norm(p[:2] - b[:2])),
            "height_above_target_m": float(p[2] - b[2]),
            "joint_speed": float(np.linalg.norm(o.joint_velocities)),
            "gripper_open": float(o.gripper_open),
            "touch_force": float(np.linalg.norm(o.gripper_touch_forces)) if o.gripper_touch_forces is not None else 0.0,
            "object_state_change": float(np.abs(s - s0).max()),
        })
    return rows


def keyframes(rows, stop_speed):
    """PerAct-style keyframes: robot stopped (joint speed local minimum below
    ``stop_speed``) or gripper state changed; first frame excluded, last included."""
    T = len(rows)
    ks = []
    for i in range(1, T):
        sp = rows[i]["joint_speed"]
        stopped = sp < stop_speed and sp <= rows[i - 1]["joint_speed"] and (i == T - 1 or sp <= rows[i + 1]["joint_speed"])
        grip = rows[i]["gripper_open"] != rows[i - 1]["gripper_open"]
        if stopped or grip or i == T - 1:
            if not ks or i - ks[-1] > 1:
                ks.append(i)
    return ks


def detect(rows, radius, stop_speed, state_eps, force_eps):
    T = len(rows)
    contact = next((r["frame"] for r in rows
                    if r["object_state_change"] > state_eps or r["touch_force"] > force_eps), None)
    ks = keyframes(rows, stop_speed)
    end = contact if contact is not None else T - 1
    cands = [k for k in ks if k < end and rows[k]["dist_to_target_m"] <= radius]
    if cands:
        entry, rule = cands[-1], "last stopped keyframe within radius before interaction onset"
    else:  # fallback: first frame inside the radius
        inside = [r["frame"] for r in rows if r["dist_to_target_m"] <= radius and r["frame"] < end]
        entry, rule = (inside[0], "fallback: first frame within radius") if inside else (None, "not found")
    return {"entry_frame": entry, "contact_frame": contact, "keyframes": ks, "rule": rule,
            "first_frame_within_radius": next((r["frame"] for r in rows if r["dist_to_target_m"] <= radius), None)}


def decode_depth(png_rgb):
    a = png_rgb.astype(np.uint32)
    return ((a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]).astype(np.float32) / DEPTH_SCALE


def tile(ep, i, phase, scale):
    import cv2
    from PIL import Image
    rgbs, deps = [], []
    for cam in CAMERAS:
        rgb = np.asarray(Image.open(os.path.join(ep, f"{cam}_rgb", f"rgb_{i:04d}.png")).convert("RGB"))
        d = decode_depth(np.asarray(Image.open(os.path.join(ep, f"{cam}_depth", f"depth_{i:04d}.png")).convert("RGB")))
        lo, hi = np.percentile(d, 1), np.percentile(d, 99)
        d8 = (np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        dc = cv2.applyColorMap(255 - d8, cv2.COLORMAP_VIRIDIS)[..., ::-1]
        h, w = rgb.shape[:2]
        rgb = cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        dc = cv2.resize(dc, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        cv2.putText(rgb, cam, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        rgbs.append(rgb)
        deps.append(np.ascontiguousarray(dc))
    img = np.concatenate([np.concatenate(rgbs, 1), np.concatenate(deps, 1)], 0)
    banner = np.zeros((24, img.shape[1], 3), np.uint8)
    tag = f"  phase: {phase}" if phase else ""
    cv2.putText(banner, f"frame {i:03d}{tag}  (top: RGB, bottom: depth buffer, per-camera normalised)",
                (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([banner, img], 0)


def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
    cmd = [shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-slice", default="0:3",
                    help="slice of task_low_dim_state holding the interaction-region position (push_button: 0:3)")
    ap.add_argument("--radius", type=float, default=0.20, help="baseline interaction radius in metres")
    ap.add_argument("--stop-speed", type=float, default=0.05, help="joint-speed threshold for a stopped keyframe (rad/s)")
    ap.add_argument("--state-eps", type=float, default=5e-4, help="object state change marking interaction onset (m)")
    ap.add_argument("--force-eps", type=float, default=0.01, help="gripper touch force marking interaction onset")
    ap.add_argument("--fps", type=int, default=10, help="video playback rate (recording is 20 Hz)")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.target_slice.split(":"))
    obs = demo_observations(load_pickle(os.path.join(a.episode, "low_dim_obs.pkl")))
    rows = frame_signals(obs, slice(lo, hi))
    det = detect(rows, a.radius, a.stop_speed, a.state_eps, a.force_eps)
    e = det["entry_frame"]
    if e is None:
        raise SystemExit("no transition frame found; inspect frames.csv and adjust --radius/--target-slice")

    for r in rows:
        r["phase"] = "transition" if r["frame"] < e else "manipulation"
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})

    oe = obs[e]
    descr = load_pickle(os.path.join(a.episode, "variation_descriptions.pkl"))
    summary = {
        "format": "entry_vla.rlbench_transition/1",
        "source_episode": os.path.abspath(a.episode),
        "instruction": descr[0] if descr else None,
        "variation": int(load_pickle(os.path.join(a.episode, "variation_number.pkl"))),
        "num_frames": len(rows),
        "frame_dt_s_assumed": RLBENCH_DT,
        "entry_frame": e,
        "contact_frame": det["contact_frame"],
        "phases": {"transition": [0, e - 1], "manipulation": [e, len(rows) - 1]},
        "keyframes": det["keyframes"],
        "first_frame_within_radius": det["first_frame_within_radius"],
        "rule": det["rule"],
        "params": {"target_slice": a.target_slice, "radius_m": a.radius, "stop_speed": a.stop_speed,
                   "state_eps_m": a.state_eps, "force_eps": a.force_eps},
        "entry_state": {
            "tcp_pose_xyz_qxyzw_world": [float(x) for x in oe.gripper_pose],
            "joint_positions_rad": [float(x) for x in oe.joint_positions],
            "joint_speed": rows[e]["joint_speed"],
            "gripper_open": float(oe.gripper_open),
            "dist_to_target_m": rows[e]["dist_to_target_m"],
            "xy_offset_m": rows[e]["xy_offset_m"],
            "height_above_target_m": rows[e]["height_above_target_m"],
        },
        "provenance": "RLBench scripted waypoint + motion-planning demo (not teleoperation); measured states, "
                      "no commanded actions stored",
    }
    with open(os.path.join(a.out, "transition.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if not a.no_video:
        trans = [tile(a.episode, i, "transition", a.scale) for i in range(0, e)]
        manip = [tile(a.episode, i, "manipulation", a.scale) for i in range(e, len(rows))]
        write_video(os.path.join(a.out, "transition_phase.mp4"), trans, a.fps)
        write_video(os.path.join(a.out, "manipulation_phase.mp4"), manip, a.fps)

    print(json.dumps({k: summary[k] for k in ("entry_frame", "contact_frame", "phases", "keyframes", "rule")}))


if __name__ == "__main__":
    main()
