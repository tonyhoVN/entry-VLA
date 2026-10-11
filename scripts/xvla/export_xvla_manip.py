#!/usr/bin/env python3
"""Export the manipulation phase of RLBench single-arm episodes (contact split) for X-VLA fine-tuning.

For every episode with contact/contact.json, frames [split_frame, T-1] (manipulation phase: from
--pre-frames before the first end-effector contact to the end) are written to
<out>/<task>/<variation>_<seed>_<episode>.hdf5:
  images/<cam>    (T,) JPEG bytes of the RGB image (encoded so cv2.imdecode returns RGB order, which
                  X-VLA's decode_image_from_bytes passes to PIL unchanged)
  ee_pos (T,3)    measured TCP position, world frame, metres
  ee_quat_xyzw (T,4)  measured TCP orientation (RLBench gripper_pose, xyzw)
  gripper_open (T,)   RLBench gripper_open (1 open, 0 closed)
  frame_index (T,)    original frame indices;  instruction: variation_descriptions[0]
Targets are measured future poses (RLBench demos have no recorded commands); X-VLA's handler turns
them into absolute EE6D chunks. Meta files (X-VLA "General Style"): <out>/meta_<split>.json, split by seed.

  python scripts/xvla/export_xvla_manip.py --tasks close_microwave --out data/xvla/rlbench_manip_close_microwave
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from entry_vla.data.peract2_raw import demo_observations, load_pickle  # noqa: E402
from entry_vla.data.rlbench_entry import list_contact_episodes, seed_split  # noqa: E402

DATASET_NAME = "rlbench_manip"


def export_episode(raw, contact_json, out_path, cams, quality):
    import cv2
    import h5py
    from PIL import Image
    j = json.load(open(contact_json))
    s = j.get("split_frame")
    if s is None:
        return None
    obs = demo_observations(load_pickle(os.path.join(raw, "low_dim_obs.pkl")))
    frames = list(range(s, len(obs)))
    gp = np.array([np.asarray(obs[t].gripper_pose, dtype=np.float64) for t in frames])
    instr = load_pickle(os.path.join(raw, "variation_descriptions.pkl"))[0]
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        g = f.create_group("images")
        vlen = h5py.vlen_dtype(np.dtype("uint8"))
        for cam in cams:
            ds = g.create_dataset(cam, (len(frames),), dtype=vlen)
            for i, t in enumerate(frames):
                rgb = np.asarray(Image.open(os.path.join(raw, f"{cam}_rgb", f"rgb_{t:04d}.png")).convert("RGB"))
                ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])   # RGB order kept
                ds[i] = buf.reshape(-1)
        f.create_dataset("ee_pos", data=gp[:, :3])
        f.create_dataset("ee_quat_xyzw", data=gp[:, 3:7])
        f.create_dataset("gripper_open", data=np.array([float(obs[t].gripper_open) for t in frames]))
        f.create_dataset("frame_index", data=np.array(frames, np.int32))
        f.create_dataset("instruction", data=instr)
        f.attrs.update(split_frame=s, contact_onset=j["first_contact_onset"], num_frames_total=len(obs),
                       quat_order="xyzw", frame="RLBench world", rate_hz=20.0,
                       targets="measured future TCP poses (no recorded commands)")
    os.replace(tmp, out_path)
    return len(frames), instr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default="data/entry_process/rlbench_select_single", help="processed (contact) root")
    ap.add_argument("--raw-root", default="data/raw/rlbench_single")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--cams", nargs="+", default=["front", "over_shoulder_left", "wrist"],
                    help="X-VLA views in order (default 3 views: exo, exo, ego)")
    ap.add_argument("--val-seeds", type=int, nargs="*", default=[18, 19, 20])
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--qdur", type=float, default=1.0, help="seconds covered by one action chunk")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rows = [r for r in list_contact_episodes(a.out_root, a.raw_root) if not a.tasks or r["task"] in a.tasks]
    metas = {s: {"dataset_name": DATASET_NAME, "datalist": [], "observation_key": [f"images/{c}" for c in a.cams],
                 "language_instruction_key": "instruction", "qdur": a.qdur, "freq": 20.0, "episodes": []}
             for s in ("train", "val")}
    skipped = []
    for r in rows:
        split = "val" if seed_split(r["seed"], a.val_seeds, []) == "val" else "train"
        path = os.path.abspath(os.path.join(a.out, r["task"], f"{r['variation']}_{r['seed']}_{r['episode']}.hdf5"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if a.force or not os.path.exists(path):
            res = export_episode(r["raw"], r["contact_json"], path, a.cams, a.jpeg_quality)
            if res is None:
                skipped.append(r["contact_json"])
                print("skip (no split)", path, flush=True)
                continue
            print(f"{split:5s} {path} ({res[0]} frames) {res[1]}", flush=True)
        metas[split]["datalist"].append(path)
        metas[split]["episodes"].append({k: r[k] for k in ("task", "variation", "seed", "episode")})
    for s, m in metas.items():
        m["skipped_no_split"] = skipped
        json.dump(m, open(os.path.join(a.out, f"meta_{s}.json"), "w"), indent=1)
        print(f"{s}: {len(m['datalist'])} episodes -> {os.path.join(a.out, f'meta_{s}.json')}")


if __name__ == "__main__":
    main()
