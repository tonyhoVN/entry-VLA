#!/usr/bin/env python3
"""Split a single-arm RLBench episode into phases at contact events between the robot end-effector and
environment objects, using RGB-D point clouds of the four exo cameras.

Per frame t:
  hand(t) = robot-link points (simulator mask ids of links that move before any object moves) inside a
            hand-sized box in the TCP frame of frame t      -> the end-effector, not the arm
  env(t)  = all other valid depth points (objects, table, scene)
  n_close(t) = #env points within --eps of hand(t); d_min(t) = min hand-env distance
  contact(t) = n_close(t) >= --k, OR finger touch force > --force-eps (gripper contact; --no-gripper-touch
               disables it)
Contact frames -> intervals (--merge-gap, --min-len) -> phases (--mode first: free | contact onward;
--mode all: alternating free/contact). Each phase boundary is placed --pre-frames frames before the
contact onset (default 5). Saved for comparison, NOT used for the split: object motion
(task_low_dim_state change), joint-force norm.

Outputs in <out>/contact/: contact.json, contact_curve.csv, contact_curve.png, phase<i>_<a>-<b>.mp4.
Robot ids come from privileged mask images, so this is an annotation/oracle tool.

  python scripts/contact_phase_split.py \
      --episode data/raw/rlbench_single/close_fridge/variation0/seed42/episodes/episode0 \
      --out data/entry_process/rlbench_single_seed42/close_fridge/variation0/episode0
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
from entry_vla.data.peract2_raw import demo_observations, load_pickle  # noqa: E402
from entry_vla.perception.contact import (close_counts, contact_intervals, first_onset,  # noqa: E402
                                          phases_from_contacts)
from entry_vla.perception.rlbench_scene import hand_and_env_points, robot_ids  # noqa: E402


def task_state(o):
    s = o.task_low_dim_state
    return np.asarray(s[0] if isinstance(s, tuple) else s, dtype=np.float64).reshape(-1)


def first_true(x):
    i = np.flatnonzero(x)
    return int(i[0]) if len(i) else None


def plot_curves(path, rows, intervals, ref_lines, eps, k, split=None):
    """ref_lines: [(label, frame or None, linestyle)] drawn as grey reference lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["frame"] for r in rows])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, facecolor="#fcfcfb")
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        for s, e in intervals:
            if split is not None:
                ax.axvline(split, color="#eb6834", ls="-.", lw=1)
            ax.axvspan(s, e - 1, color="#eb6834", alpha=0.12, lw=0)
        for name, f, ls in ref_lines:
            if f is not None:
                ax.axvline(f, color="#52514e", ls=ls, lw=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    dmin = np.array([r["d_min_m"] for r in rows])
    a1.plot(t, np.minimum(dmin, 0.3), color="#2a78d6", lw=2)
    a1.axhline(eps, color="#eb6834", lw=1)
    a1.text(t[-1], eps, f" eps={eps * 100:.1f} cm", color="#0b0b0b", va="bottom", ha="right", fontsize=8)
    a1.set_ylabel("min hand-env\ndistance (m, cap 0.3)")
    a2.plot(t, [r["n_close"] for r in rows], color="#2a78d6", lw=2)
    a2.axhline(k, color="#eb6834", lw=1)
    a2.text(t[-1], k, f" k={k}", color="#0b0b0b", va="bottom", ha="right", fontsize=8)
    a2.set_ylabel("#env points\nwithin eps")
    a2.set_xlabel("frame")
    on = first_onset(intervals)
    styles = {"--": "dashed", ":": "dotted", "-": "solid"}
    a1.set_title(f"End-effector contact (shaded; geometric or finger contact), first onset {on}, split {split} "
                 f"(dash-dot)\n" + ", ".join(f"{styles.get(ls, ls)} = {n} ({f})" for n, f, ls in ref_lines),
                 fontsize=9, loc="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True, help="processed episode dir; results go to <out>/contact")
    ap.add_argument("--eps", type=float, default=0.015, help="contact distance (m)")
    ap.add_argument("--k", type=int, default=20, help="min env points within eps for contact")
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--merge-gap", type=int, default=2)
    ap.add_argument("--mode", default="first", choices=["first", "all"])
    ap.add_argument("--state-eps", type=float, default=1e-3, help="object motion threshold for the reference")
    ap.add_argument("--force-eps", type=float, default=0.02, help="finger touch force threshold (N) for gripper contact")
    ap.add_argument("--no-gripper-touch", action="store_true", help="geometric contact only, ignore touch force")
    ap.add_argument("--pre-frames", type=int, default=5, help="split this many frames before contact onset")
    ap.add_argument("--subdir", default="contact", help="output subdir of --out")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--video-fps", type=int, default=20)
    a = ap.parse_args()
    out = os.path.join(a.out, a.subdir)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    obs = demo_observations(load_pickle(os.path.join(a.episode, "low_dim_obs.pkl")))
    T = len(obs)
    s0 = task_state(obs[0])
    dstate = np.array([np.abs(task_state(o) - s0).max() for o in obs])
    touch = np.array([np.linalg.norm(o.gripper_touch_forces) for o in obs])
    jforce = np.array([np.linalg.norm(o.joint_forces) for o in obs])
    t_obj = first_true(dstate > a.state_eps)
    refs = {"object_motion_onset": t_obj, "touch_force_onset": first_true(touch > a.force_eps),
            "gripper_toggles": [i for i in range(1, T) if obs[i].gripper_open != obs[i - 1].gripper_open]}

    # robot links: ids that moved before any object moved
    horizon = (t_obj if t_obj is not None else T) - 1
    frames = sorted({max(1, horizon // 3), max(1, 2 * horizon // 3), max(1, horizon)})
    rid = robot_ids(a.episode, obs, frames)
    print(f"robot ids {sorted(rid)} from frames {frames}", flush=True)

    rows = []
    for t in range(T):
        hand, env = hand_and_env_points(a.episode, obs[t], t, rid)
        n, dmin = close_counts(hand, env, a.eps)
        geo = n >= a.k
        grip = (not a.no_gripper_touch) and touch[t] > a.force_eps
        rows.append({"frame": t, "n_hand_pts": len(hand), "n_close": n, "d_min_m": round(dmin, 5),
                     "geometric_contact": int(geo), "gripper_contact": int(grip), "contact": int(geo or grip), "object_state_change": round(float(dstate[t]), 5),
                     "touch_force": round(float(touch[t]), 4), "joint_force_norm": round(float(jforce[t]), 2)})
    intervals = contact_intervals([r["contact"] for r in rows], a.min_len, a.merge_gap)
    phases = phases_from_contacts(intervals, T, a.mode, a.pre_frames)
    onset = first_onset(intervals)
    with open(os.path.join(out, "contact_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"format": "entry_vla.contact_phase_split/1", "episode": os.path.abspath(a.episode),
               "instruction": load_pickle(os.path.join(a.episode, "variation_descriptions.pkl"))[0],
               "num_frames": T, "params": {k: getattr(a, k) for k in ("eps", "k", "min_len", "merge_gap", "mode", "pre_frames",
                                                     "force_eps", "no_gripper_touch")},
               "robot_ids": sorted(rid), "robot_id_frames": frames,
               "contact_intervals": [list(map(int, i)) for i in intervals], "first_contact_onset": onset,
               "split_frame": phases[1]["start"] if len(phases) > 1 else None,
               "phases": phases, "references": refs,
               "onset_minus_object_motion": None if onset is None or t_obj is None else onset - t_obj,
               "note": "geometric contact of end-effector RGB-D points with environment points; robot ids from "
                       "simulator masks (privileged); references are not used for the split"}
    json.dump(summary, open(os.path.join(out, "contact.json"), "w"), indent=1)
    plot_curves(os.path.join(out, "contact_curve.png"), rows, intervals,
                [("object moves", t_obj, "--"), ("touch force", refs["touch_force_onset"], ":")], a.eps, a.k,
                phases[1]["start"] if len(phases) > 1 else None)
    if not a.no_video:
        from detect_transition_rlbench import tile, write_video
        for old in (f for f in os.listdir(out) if f.startswith("phase") and f.endswith(".mp4")):
            os.remove(os.path.join(out, old))
        for i, p in enumerate(phases, 1):
            write_video(os.path.join(out, f"phase{i}_{p['phase']}_{p['start']:03d}-{p['end']:03d}.mp4"),
                        [tile(a.episode, t, f"{i}/{len(phases)} {p['phase']} (contact split)", 1)
                         for t in range(p["start"], p["end"] + 1)], a.video_fps)
    print(json.dumps({k: summary[k] for k in ("contact_intervals", "first_contact_onset", "split_frame", "phases", "references",
                                              "onset_minus_object_motion")}, indent=1))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
