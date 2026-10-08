#!/usr/bin/env python3
"""Split single-arm RLBench episodes into phases by comparing latent frame representations.

For every episode: embed each frame per camera with a frozen encoder (DINOv2-small CLS,
L2-normalised; cached as embeddings_<cam>.npy), then for several camera sets find
  * k2:      the single best boundary (exact 2-segment least-squares split of the
             per-episode standardised embedding trajectory) = transition -> manipulation
  * k3:      the best 3-segment split (two boundaries)
  * novelty: the strongest peak of a sliding-window cosine novelty curve
and compare them with existing references in <out>/<episode>/ when present:
  transition.json (state-based stop keyframe + contact) and perception/comparison.json
  (Qwen3-VL + SAM distance split at --perception-radius). Neither reference is a human label.

  python scripts/latent_phase_split.py \
      --episodes-root data/raw/rlbench_single_seed42/push_button/variation4/episodes \
      --out data/entry_process/push_button_seed42/variation4
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
from entry_vla.perception.latent_split import (first_boundary, novelty_curve, novelty_peaks,  # noqa: E402
                                               optimal_segmentation, standardise)

ALL_CAMERAS = ["front", "over_shoulder_left", "over_shoulder_right", "overhead", "wrist"]
VIEW_SETS = {"exo": ["front", "over_shoulder_left"], "wrist": ["wrist"], "all": ALL_CAMERAS}


def n_frames(ep):
    return len([f for f in os.listdir(os.path.join(ep, "front_rgb")) if f.endswith(".png")])


def embed_episode(encoder, ep, out, cameras, force=False):
    embs = {}
    for cam in cameras:
        path = os.path.join(out, f"embeddings_{cam}.npy")
        if os.path.exists(path) and not force:
            embs[cam] = np.load(path)
            continue
        imgs = [np.asarray(Image.open(os.path.join(ep, f"{cam}_rgb", f"rgb_{i:04d}.png")).convert("RGB"))
                for i in range(n_frames(ep))]
        embs[cam] = encoder.encode(imgs).astype(np.float32)
        np.save(path, embs[cam])
    return embs


def references(ep_out, perception_radius):
    refs = {}
    t = os.path.join(ep_out, "transition.json")
    if os.path.exists(t):
        d = json.load(open(t))
        refs["stop_frame"], refs["contact_frame"] = d.get("entry_frame"), d.get("contact_frame")
    c = os.path.join(ep_out, "perception", "comparison.json")
    if os.path.exists(c):
        d = json.load(open(c))
        hit = [r["pred_start"] for r in d["radius_sweep"] if abs(r["radius_m"] - perception_radius) < 1e-9]
        refs[f"perception_r{perception_radius:.2f}"] = hit[0] if hit else None
        refs["gripper_toggles"] = d.get("other_references", {}).get("gripper_toggle_frames")
    return refs


def split_episode(embs, k3_min_len, window):
    res = {}
    for name, cams in VIEW_SETS.items():
        if not all(c in embs for c in cams):
            continue
        X = standardise(np.concatenate([embs[c] for c in cams], axis=1))
        nov = novelty_curve(X, window)
        res[name] = {"k2": first_boundary(optimal_segmentation(X, 2)),
                     "k3": optimal_segmentation(X, 3, min_len=k3_min_len),
                     "novelty_peak": first_boundary(novelty_peaks(nov, 1)),
                     "novelty": nov.round(5).tolist()}
    return res


def write_phase_videos(ep, lat_out, bounds, n, label, fps):
    """phase<i>_<start>-<end>.mp4 for every segment between ``bounds`` (all cameras, RGB over depth)."""
    sys.path.insert(0, os.path.dirname(__file__))
    from detect_transition_rlbench import tile, write_video
    for old in (f for f in os.listdir(lat_out) if f.startswith("phase") and f.endswith(".mp4")):
        os.remove(os.path.join(lat_out, old))
    edges = [0] + list(bounds) + [n]
    for i, (s, e) in enumerate(zip(edges[:-1], edges[1:]), 1):
        if e > s:
            write_video(os.path.join(lat_out, f"phase{i}_{s:03d}-{e - 1:03d}.mp4"),
                        [tile(ep, t, f"{i}/{len(edges) - 1} ({label})", 1) for t in range(s, e)], fps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes-root", required=True, help="raw .../variation<V>/episodes")
    ap.add_argument("--out", required=True, help="processed root holding <episode>/ dirs")
    ap.add_argument("--encoder", default="facebook/dinov2-small")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--window", type=int, default=5, help="novelty window (frames)")
    ap.add_argument("--k3-min-len", type=int, default=5)
    ap.add_argument("--perception-radius", type=float, default=0.10)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="recompute cached embeddings")
    ap.add_argument("--video-split", default="wrist:k3", help="view:method for phase videos (k2|k3|novelty); 'none' skips")
    ap.add_argument("--video-fps", type=int, default=20)
    a = ap.parse_args()

    from entry_vla.perception.latent_split import FrameEncoder
    t0 = time.time()
    enc = FrameEncoder(a.encoder, device=a.device, batch_size=a.batch_size)
    print(f"encoder {a.encoder} loaded in {time.time() - t0:.1f}s", flush=True)

    eps = sorted((d for d in os.listdir(a.episodes_root) if d.startswith("episode")), key=lambda d: int(d[7:]))
    rows = []
    for d in eps[:a.max_episodes]:
        ep, ep_out = os.path.join(a.episodes_root, d), os.path.join(a.out, d)
        lat_out = os.path.join(ep_out, "latent")
        os.makedirs(lat_out, exist_ok=True)
        t1 = time.time()
        embs = embed_episode(enc, ep, lat_out, ALL_CAMERAS, a.force)
        splits = split_episode(embs, a.k3_min_len, a.window)
        refs = references(ep_out, a.perception_radius)
        summary = {"format": "entry_vla.latent_phase_split/1", "episode": os.path.abspath(ep), "encoder": a.encoder,
                   "num_frames": int(len(next(iter(embs.values())))), "view_sets": VIEW_SETS, "window": a.window,
                   "splits": splits, "references": refs,
                   "note": "unsupervised change points in DINOv2 CLS space; references are not human labels"}
        with open(os.path.join(lat_out, "latent_split.json"), "w") as f:
            json.dump(summary, f, indent=1)
        if a.video_split != "none":
            view, method = a.video_split.split(":")
            sp = splits[view]
            b = sp["k3"] if method == "k3" else [sp["k2" if method == "k2" else "novelty_peak"]]
            write_phase_videos(ep, lat_out, [x for x in b if x is not None], summary["num_frames"],
                               f"latent {view} {method}", a.video_fps)
        row = {"episode": d, "frames": summary["num_frames"], **{k: v for k, v in refs.items() if k != "gripper_toggles"}}
        for v, s in splits.items():
            row[f"{v}_k2"], row[f"{v}_novelty"] = s["k2"], s["novelty_peak"]
            row[f"{v}_k3"] = " ".join(map(str, s["k3"]))
        rows.append(row)
        print(d, {k: row[k] for k in row if k.endswith(("_k2", "_k3")) or k in ("stop_frame", "contact_frame")},
              f"{time.time() - t1:.1f}s", flush=True)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(a.out, "latent_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    agg = {"episodes": len(rows), "encoder": a.encoder, "error_vs_reference_frames": {}}
    ref_keys = [k for k in ("stop_frame", "contact_frame", f"perception_r{a.perception_radius:.2f}") if k in keys]
    for col in [k for k in keys if k.endswith(("_k2", "_novelty"))]:
        for ref in ref_keys:
            e = [r[col] - r[ref] for r in rows if r.get(col) is not None and r.get(ref) is not None]
            if e:
                agg["error_vs_reference_frames"][f"{col} - {ref}"] = {
                    "n": len(e), "mean": round(float(np.mean(e)), 2), "mean_abs": round(float(np.mean(np.abs(e))), 2),
                    "median_abs": float(np.median(np.abs(e))), "within_5_frames": round(float(np.mean(np.abs(e) <= 5)), 2)}
    with open(os.path.join(a.out, "latent_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
