#!/usr/bin/env python3
"""Contact phase split for a VLABench episode recorded by scripts/vlabench_generator.py (entryvla env).

Same rule as scripts/contact_phase_split.py (RLBench), adapted to VLABench data:
  hand(t) = depth points whose MuJoCo geom id belongs to the Franka hand/finger subtree (privileged
            segmentation) within 0.25 m of the TCP, from the static exo cameras (moving cameras dropped)
  Silhouette-edge pixels (mixed geom ids in a 3x3 window) are dropped for both clouds.
  env(t)  = depth points of non-robot geoms (pixels without a geom are dropped)
  contact(t) = #env points within --eps of hand(t) >= --k, OR MuJoCo finger contact with a non-robot
               geom (gripper contact; --no-gripper-touch disables it)
Contact frames -> intervals (--merge-gap, --min-len) -> phases (--mode); the split is placed
--pre-frames frames before the onset. References saved, not used for the split: MuJoCo robot-object
contact onset, gripper closing onset, expert skill boundaries.

Outputs in <out>/<subdir>/ (default contact): contact.json, contact_curve.csv/png, phase<i>_*.mp4.

  python scripts/contact_phase_split_vlabench.py \
      --episode data/raw/vlabench/select_fruit/seed42/episodes/episode0 \
      --out data/entry_process/vlabench_short5_seed42/select_fruit/episode0
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from entry_vla.geometry.camera import backproject_to_world  # noqa: E402
from entry_vla.perception.contact import (close_counts, contact_intervals, first_onset,  # noqa: E402
                                          phases_from_contacts)


def first_true(x):
    i = np.flatnonzero(x)
    return int(i[0]) if len(i) else None


def static_cameras(T_WC, tol=1e-4):
    """Indices of cameras whose pose never changes (wrist cameras move with the arm)."""
    return [c for c in range(T_WC.shape[1]) if np.abs(T_WC[:, c] - T_WC[:1, c]).max() < tol]


class FrameBlocks:
    """Reads depth/segmentation in blocks of frames (HDF5 chunks may span several frames)."""

    def __init__(self, f, cams, block=24):
        self.f, self.cams, self.block, self.t0, self.d, self.s = f, cams, block, None, None, None
        self.K, self.T = f["cam_K"][:], f["cam_T_WC"][:]

    def get(self, t):
        if self.t0 is None or not (self.t0 <= t < self.t0 + self.block):
            self.t0 = t - t % self.block
            sl = slice(self.t0, self.t0 + self.block)
            self.d = self.f["depth"][sl][:, self.cams]
            self.s = self.f["seg_geom"][sl][:, self.cams]
        i = t - self.t0
        return [(self.d[i, j], self.s[i, j], self.K[t, c], self.T[t, c]) for j, c in enumerate(self.cams)]


def interior(seg):
    """Pixels whose 3x3 neighbourhood has a single geom id: drops silhouette edges, where the depth and
    segmentation renders disagree (a hand-labelled edge pixel can carry the depth of the wall behind)."""
    p = np.pad(seg, 1, mode="edge")
    H, W = seg.shape
    ok = np.ones_like(seg, bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            ok &= p[dy:dy + H, dx:dx + W] == seg
    return ok


def hand_env_points(frames, t, hand_ids, robot_ids, tcp=None, hand_radius=0.25, crop=0.3):
    hand, env = [], []
    for d, seg, K, T in frames.get(t):
        valid = (seg >= 0) & np.isfinite(d) & (d > 0) & interior(seg)
        hand.append(backproject_to_world(d, K, T, valid & np.isin(seg, hand_ids)))
        env.append(backproject_to_world(d, K, T, valid & ~np.isin(seg, robot_ids)))
    hand, env = np.concatenate(hand), np.concatenate(env)
    if tcp is not None:   # the Franka hand spans ~0.2 m around the TCP
        hand = hand[np.linalg.norm(hand - tcp, axis=1) <= hand_radius]
    if len(hand):   # only env points near the hand matter for eps-contact
        lo, hi = hand.min(0) - crop, hand.max(0) + crop
        env = env[np.all((env >= lo) & (env <= hi), axis=1)]
    return hand, env


def tile(f, t, cams, label):
    import cv2
    rows = []
    for c in cams:
        rgb = f["rgb"][t, c]
        d = f["depth"][t, c]
        lo, hi = np.percentile(d, 1), np.percentile(d, 99)
        d8 = (np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        rows.append(np.vstack([rgb, cv2.applyColorMap(255 - d8, cv2.COLORMAP_VIRIDIS)[..., ::-1]]))
    img = np.ascontiguousarray(np.hstack(rows))
    cv2.putText(img, f"{t:04d} {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True, help="dir with data.hdf5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, default=0.015)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--merge-gap", type=int, default=2)
    ap.add_argument("--mode", default="first", choices=["first", "all"])
    ap.add_argument("--pre-frames", type=int, default=5)
    ap.add_argument("--no-gripper-touch", action="store_true")
    ap.add_argument("--subdir", default="contact")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--video-fps", type=int, default=10)
    ap.add_argument("--video-scale", type=float, default=0.5, help="downscale of the 480 px views")
    a = ap.parse_args()
    import h5py
    out = os.path.join(a.out, a.subdir)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    with h5py.File(os.path.join(a.episode, "data.hdf5"), "r") as f:
        T = f["rgb"].shape[0]
        names = json.loads(f.attrs["camera_names"])
        cams = static_cameras(f["cam_T_WC"][:])
        hand_ids, robot_ids = np.asarray(f.attrs["hand_geom_ids"]), np.asarray(f.attrs["robot_geom_ids"])
        sim_f, sim_r = f["sim_finger_contact"][:], f["sim_robot_contact"][:]
        width = f["finger_qpos"][:].sum(1)
        skill_end = f["skill_end"][:].tolist()
        tcp = f["tcp_T_W"][:, :3, 3]
        print(f"{T} frames, static cameras {[names[c] for c in cams]}", flush=True)
        rows, frames = [], FrameBlocks(f, cams)
        for t in range(T):
            hand, env = hand_env_points(frames, t, hand_ids, robot_ids, tcp[t])
            n, dmin = close_counts(hand, env, a.eps)
            geo = n >= a.k
            grip = (not a.no_gripper_touch) and sim_f[t] > 0
            rows.append({"frame": t, "n_hand_pts": len(hand), "n_close": n, "d_min_m": round(dmin, 5),
                         "geometric_contact": int(geo), "gripper_contact": int(grip), "contact": int(geo or grip),
                         "sim_finger_contact": int(sim_f[t]), "sim_robot_contact": int(sim_r[t]),
                         "finger_width_m": round(float(width[t]), 5)})
        intervals = contact_intervals([r["contact"] for r in rows], a.min_len, a.merge_gap)
        phases = phases_from_contacts(intervals, T, a.mode, a.pre_frames)
        onset = first_onset(intervals)
        split = phases[1]["start"] if len(phases) > 1 else None
        closing = first_true(np.diff(width) < -1e-3)
        refs = {"sim_robot_contact_onset": first_true(sim_r > 0), "sim_finger_contact_onset": first_true(sim_f > 0),
                "gripper_closing_onset": None if closing is None else closing + 1, "skill_end_frames": skill_end}
        with open(os.path.join(out, "contact_curve.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summary = {"format": "entry_vla.contact_phase_split/1", "backend": "vlabench",
                   "episode": os.path.abspath(a.episode), "task": f.attrs["task"],
                   "instruction": f.attrs["instruction"], "num_frames": T,
                   "cameras": [names[c] for c in cams],
                   "params": {k: getattr(a, k) for k in ("eps", "k", "min_len", "merge_gap", "mode", "pre_frames",
                                                         "no_gripper_touch")},
                   "contact_intervals": [list(map(int, i)) for i in intervals], "first_contact_onset": onset,
                   "split_frame": split, "phases": phases, "references": refs,
                   "onset_minus_sim_robot_contact": None if onset is None or refs["sim_robot_contact_onset"] is None
                   else onset - refs["sim_robot_contact_onset"],
                   "note": "hand/env points from privileged MuJoCo geom segmentation; gripper contact from MuJoCo "
                           "finger contacts; references are not used for the split"}
        json.dump(summary, open(os.path.join(out, "contact.json"), "w"), indent=1)
        from contact_phase_split import plot_curves
        plot_curves(os.path.join(out, "contact_curve.png"), rows, intervals,
                    [("sim robot contact", refs["sim_robot_contact_onset"], "--"),
                     ("gripper closes", refs["gripper_closing_onset"], ":")], a.eps, a.k, split)
        if not a.no_video:
            import cv2
            from detect_transition_rlbench import write_video
            for old in (x for x in os.listdir(out) if x.startswith("phase") and x.endswith(".mp4")):
                os.remove(os.path.join(out, old))
            for i, p in enumerate(phases, 1):
                frames = []
                for t in range(p["start"], p["end"] + 1):
                    img = tile(f, t, cams, f"{i}/{len(phases)} {p['phase']} (contact split)")
                    img = cv2.resize(img, None, fx=a.video_scale, fy=a.video_scale, interpolation=cv2.INTER_AREA)
                    frames.append(img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2])
                write_video(os.path.join(out, f"phase{i}_{p['phase']}_{p['start']:03d}-{p['end']:03d}.mp4"),
                            frames, a.video_fps)
    print(json.dumps({k: summary[k] for k in ("contact_intervals", "first_contact_onset", "split_frame", "phases",
                                              "references", "onset_minus_sim_robot_contact")}, indent=1))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
