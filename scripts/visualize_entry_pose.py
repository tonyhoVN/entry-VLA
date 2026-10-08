#!/usr/bin/env python3
"""Report figures of the manipulation-start (entry) pose for single-arm RLBench episodes.

For each episode processed by perception_phase_split.py (perception/comparison.json):
  entry_front_overlay.png - front RGB of frame 0 with the robot at the entry frame blended on top at low
                            opacity; robot outlines per frame (frame 0 vs entry) and the entry TCP XYZ axes
  entry_pose_3d.png       - segmented object point cloud (perception), robot hand at the entry frame
                            (robot depth points near the TCP + schematic Franka hand), TCP XYZ frame, and the
                            TCP path from frame 0 to the entry frame
The entry frame is pred_start_at_baseline (the perception split at its radius). Robot pixels come from the
simulator mask images: ids whose 3-D points (all exo cameras) moved by more than 2 cm between frame 0 and the
entry frame (static objects - occluded or settling - keep matching their own points).

  python scripts/visualize_entry_pose.py --raw-root data/raw/rlbench_single_seed42 \
      --out-root data/entry_process/rlbench_short5_seed42
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from entry_vla.data.peract2_raw import demo_observations, load_pickle  # noqa: E402
from entry_vla.geometry.camera import backproject_to_world, project_world  # noqa: E402
from entry_vla.geometry.transforms import apply_T, pose_to_T  # noqa: E402
from entry_vla.sim.peract2 import camera_from_misc  # noqa: E402

START_C, ENTRY_C = (42, 120, 214), (235, 104, 52)       # categorical slots 1 / 2 (blue = frame 0, orange = entry)
AXIS_C = {"X": (214, 39, 40), "Y": (44, 160, 44), "Z": (31, 119, 180)}  # conventional RGB frame axes
SURFACE, INK = "#fcfcfb", "#0b0b0b"
EXO = ("front", "overhead", "over_shoulder_left", "over_shoulder_right")


def load_cam(ep, misc, cam, i):
    rgb = np.asarray(Image.open(os.path.join(ep, f"{cam}_rgb", f"rgb_{i:04d}.png")).convert("RGB"))
    d = np.asarray(Image.open(os.path.join(ep, f"{cam}_depth", f"depth_{i:04d}.png")).convert("RGB"))
    m = np.asarray(Image.open(os.path.join(ep, f"{cam}_mask", f"mask_{i:04d}.png")).convert("RGB"))
    ids = m.astype(np.int64) @ np.array([1, 256, 65536])
    return camera_from_misc(cam, misc, rgb, d, "rgb24_linear_buffer"), ids


def robot_ids(ep, obs, e, move_thresh=0.02, min_pts=20, max_pts=4000):
    """Mask ids of robot links that moved between frame 0 and the entry frame. Each id's pixels in all exo
    cameras are back-projected to world points in both frames; an id moved if the median distance from its
    entry-frame points to its own frame-0 points exceeds ``move_thresh`` (or it is visible only at entry).
    Static objects - occluded by the arm or settling by millimetres - keep matching their own points."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    pts = {0: {}, 1: {}}
    for k, t in enumerate((0, e)):
        for cam in EXO:
            c, ids = load_cam(ep, obs[t].misc, cam, t)
            for v in np.unique(ids[c.depth_valid]):
                P = backproject_to_world(c.depth, c.K, c.T_WC, (ids == v) & c.depth_valid)
                pts[k].setdefault(int(v), []).append(P)
    out = set()
    for v, lst in pts[1].items():
        PE = np.concatenate(lst)
        if len(PE) < min_pts:
            continue
        if v not in pts[0]:
            out.add(v)
            continue
        P0 = np.concatenate(pts[0][v])
        if len(PE) > max_pts:
            PE = PE[rng.choice(len(PE), max_pts, replace=False)]
        if np.median(cKDTree(P0).query(PE)[0]) > move_thresh:
            out.add(v)
    return out


def outline(mask):
    import cv2
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cnt


def front_overlay(ep, obs, e, path, alpha, scale):
    import cv2
    rid = robot_ids(ep, obs, e)
    c0, ids0 = load_cam(ep, obs[0].misc, "front", 0)
    cE, idsE = load_cam(ep, obs[e].misc, "front", e)
    r0, rE = np.isin(ids0, list(rid)), np.isin(idsE, list(rid))
    img = c0.rgb.astype(np.float32).copy()
    img[rE] = (1 - alpha) * img[rE] + alpha * cE.rgb.astype(np.float32)[rE]  # entry robot, low opacity
    img = cv2.resize(img.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    up = lambda m: cv2.resize(m.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.drawContours(img, outline(up(r0)), -1, START_C, 2, cv2.LINE_AA)
    cv2.drawContours(img, outline(up(rE)), -1, ENTRY_C, 2, cv2.LINE_AA)
    T = pose_to_T(np.asarray(obs[e].gripper_pose[:3]), np.asarray(obs[e].gripper_pose[3:7]), "xyzw")
    L = 0.08
    pts = np.array([T[:3, 3]] + [T[:3, 3] + L * T[:3, k] for k in range(3)])
    uv, z = project_world(pts, cE.K, cE.T_WC)
    uv = uv * scale
    o = tuple(int(v) for v in uv[0])
    for k, name in enumerate("XYZ"):
        p = tuple(int(v) for v in uv[k + 1])
        cv2.arrowedLine(img, o, p, AXIS_C[name], 3, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(img, name, (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, AXIS_C[name], 2, cv2.LINE_AA)
    cv2.circle(img, o, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, o, 5, (0, 0, 0), 1, cv2.LINE_AA)
    bar = np.full((34, img.shape[1], 3), 252, np.uint8)
    cv2.rectangle(bar, (8, 10), (26, 24), START_C, 2)
    cv2.putText(bar, "frame 0 (start)", (32, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
    cv2.rectangle(bar, (190, 10), (208, 24), ENTRY_C, 2)
    cv2.putText(bar, f"frame {e} (manipulation start, {int(alpha * 100)}% opacity)", (214, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
    Image.fromarray(np.concatenate([img, bar], 0)).save(path)
    return rid


def hand_points(ep, obs, e, rid, T_WE, box=((-0.05, 0.05), (-0.12, 0.12), (-0.12, 0.02))):
    """Robot depth points at the entry frame cropped to a hand-sized box in the TCP frame
    (x thin, y = finger opening / palm width, z = approach; palm behind the TCP at z < 0)."""
    from entry_vla.geometry.transforms import invert_T
    pts = []
    for cam in EXO:
        c, ids = load_cam(ep, obs[e].misc, cam, e)
        m = np.isin(ids, list(rid)) & c.depth_valid
        if m.any():
            pts.append(backproject_to_world(c.depth, c.K, c.T_WC, m))
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    Q = apply_T(invert_T(T_WE), P)
    keep = np.all([(Q[:, k] >= lo) & (Q[:, k] <= hi) for k, (lo, hi) in enumerate(box)], axis=0)
    return P[keep]


def hand_schematic(T_WE, width):
    """Franka hand outline in the TCP frame (z = approach, y = finger opening), metres -> world segments."""
    h = width / 2
    segs_E = [
        [(-0.02, -0.10, -0.065), (0.02, -0.10, -0.065), (0.02, 0.10, -0.065), (-0.02, 0.10, -0.065), (-0.02, -0.10, -0.065)],
        [(0, -h - 0.01, -0.045), (0, -h - 0.01, 0.0), (0, -h, 0.0), (0, -h, -0.045)],   # finger 1
        [(0, h + 0.01, -0.045), (0, h + 0.01, 0.0), (0, h, 0.0), (0, h, -0.045)],       # finger 2
        [(0, -h - 0.01, -0.045), (0, -0.10, -0.065)], [(0, h + 0.01, -0.045), (0, 0.10, -0.065)],
        [(0, 0, -0.065), (0, 0, -0.11)],                                                # flange
    ]
    return [apply_T(T_WE, np.array(s)) for s in segs_E]


def pose_3d(obs, e, obj, hand, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = pose_to_T(np.asarray(obs[e].gripper_pose[:3]), np.asarray(obs[e].gripper_pose[3:7]), "xyzw")
    width = float(np.sum(obs[e].gripper_joint_positions)) if obs[e].gripper_joint_positions is not None else 0.08
    path_tcp = np.array([o.gripper_pose[:3] for o in obs[:e + 1]])
    fig = plt.figure(figsize=(7.2, 6.4), facecolor=SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(SURFACE)
    if len(obj):
        ax.scatter(*obj.T, s=1.2, c="#2a78d6", alpha=0.25, depthshade=False, label=f"object point cloud ({len(obj)} pts)")
    if len(hand):
        ax.scatter(*hand.T, s=1.0, c="#eb6834", alpha=0.18, depthshade=False, label="robot hand (depth points)")
    for k, s in enumerate(hand_schematic(T, width)):
        ax.plot(*s.T, color="#7a2e0c", lw=2.4, zorder=6, label="hand outline" if k == 0 else None)
    L = 0.08
    for k, name in enumerate("XYZ"):
        c = np.array(AXIS_C[name]) / 255
        ax.quiver(*T[:3, 3], *(L * T[:3, k]), color=c, lw=2.5, arrow_length_ratio=0.2)
        ax.text(*(T[:3, 3] + 1.18 * L * T[:3, k]), name, color=c, fontsize=11, weight="bold")
    ax.scatter(*T[:3, 3], s=40, c="white", edgecolors=INK, zorder=5, label=f"TCP at frame {e}")
    pts = np.concatenate([p for p in (obj, hand, T[None, :3, 3]) if len(p)])
    c, r = pts.mean(0), max(np.ptp(pts, 0).max() / 2, 0.12) * 1.1
    inside = np.all(np.abs(path_tcp - c) <= r, axis=1)  # 3-D lines are not clipped by matplotlib
    first = int(np.argmax(inside)) if inside.any() else len(path_tcp) - 1
    ax.plot(*path_tcp[first:].T, color="#8c8b86", lw=1.2, ls="--",
            label=f"TCP path, frame {first} \u2192 {e}")
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_zlabel("world z (m)")
    ax.view_init(elev=22, azim=-60)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--alpha", type=float, default=0.45, help="opacity of the entry-frame robot on frame 0")
    ap.add_argument("--scale", type=int, default=3, help="upscale of the 256 px front image")
    a = ap.parse_args()
    tiles = []
    for cmp_path in sorted(glob.glob(os.path.join(a.out_root, "*", "variation*", "episode*", "perception",
                                                  "comparison.json"))):
        task, var, epn = cmp_path.split(os.sep)[-5:-2]
        d = json.load(open(cmp_path))
        e = d["pred_start_at_baseline"]
        if e is None:
            print(task, "no manipulation start; skipped", flush=True)
            continue
        ep = os.path.join(a.raw_root, task, var, "episodes", epn)
        out = os.path.join(os.path.dirname(os.path.dirname(cmp_path)), "entry_viz")
        os.makedirs(out, exist_ok=True)
        obs = demo_observations(load_pickle(os.path.join(ep, "low_dim_obs.pkl")))
        rid = front_overlay(ep, obs, e, os.path.join(out, "entry_front_overlay.png"), a.alpha, a.scale)
        T = pose_to_T(np.asarray(obs[e].gripper_pose[:3]), np.asarray(obs[e].gripper_pose[3:7]), "xyzw")
        obj = np.load(os.path.join(os.path.dirname(cmp_path), "object_points_pred.npy"))
        hand = hand_points(ep, obs, e, rid, T)
        labels = sorted({(v["detection"] or {}).get("label") for v in d["per_camera"].values()} - {None})
        pose_3d(obs, e, obj, hand, os.path.join(out, "entry_pose_3d.png"),
                f"{task} ({var}): entry pose at frame {e}, R={d['baseline_radius_m']:.2f} m\n"
                f"object: {', '.join(labels)}")
        json.dump({"entry_frame": e, "radius_m": d["baseline_radius_m"], "robot_mask_ids": sorted(rid),
                   "tcp_T_WE": T.round(5).tolist(), "hand_points": int(len(hand)), "object_points": int(len(obj))},
                  open(os.path.join(out, "entry_viz.json"), "w"), indent=1)
        tiles.append((task, out))
        print(task, var, "entry", e, "robot ids", len(rid), "hand pts", len(hand), flush=True)
    if tiles:  # report sheet: overlay above 3-D per task
        ims = [(Image.open(os.path.join(o, "entry_front_overlay.png")), Image.open(os.path.join(o, "entry_pose_3d.png")))
               for _, o in tiles]
        W = max(max(x.width, y.width) for x, y in ims)
        cols = [Image.new("RGB", (W, x.height + y.height), SURFACE) for x, y in ims]
        for col, (x, y) in zip(cols, ims):
            col.paste(x, ((W - x.width) // 2, 0))
            col.paste(y, ((W - y.width) // 2, x.height))
        sheet = Image.new("RGB", (W * len(cols), max(c.height for c in cols)), SURFACE)
        for i, col in enumerate(cols):
            sheet.paste(col, (i * W, 0))
        sheet.save(os.path.join(a.out_root, "entry_pose_report.png"))
        print("report:", os.path.join(a.out_root, "entry_pose_report.png"))


if __name__ == "__main__":
    main()
