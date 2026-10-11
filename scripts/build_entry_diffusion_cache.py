#!/usr/bin/env python3
"""Build the entry-pose diffusion cache (object point cloud, object crops, instruction, entry target) from
RLBench episodes with a contact phase split. One npz per episode + index.json (split by seed).
Detector: --detector oracle (contact object from simulator mask ids) or qwen_sam (deployable: Qwen3-VL box +
SAM 2.1 mask on frame 0; cloud from --cloud-cams, crops from --crop-cams); see entry_vla.data.entry_diffusion_data.

  python scripts/build_entry_diffusion_cache.py --out-root data/entry_process/rlbench_select_single \
      --raw-root data/raw/rlbench_single --cache data/entry_train/diffusion_select_single
Then add frozen VLM features: scripts/embed_vlm_features.py --cache <cache>
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from entry_vla.data.entry_diffusion_data import episode_inputs, episode_inputs_detected  # noqa: E402
from entry_vla.data.rlbench_entry import list_contact_episodes, seed_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default="data/entry_process/rlbench_select_single")
    ap.add_argument("--raw-root", default="data/raw/rlbench_single")
    ap.add_argument("--cache", default="data/entry_train/diffusion_select_single")
    ap.add_argument("--val-seeds", type=int, nargs="*", default=[15, 16, 17])
    ap.add_argument("--test-seeds", type=int, nargs="*", default=[18, 19, 20])
    ap.add_argument("--n-points", type=int, default=2048)
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("--max-object-dist", type=float, default=0.15,
                    help="skip when the nearest frame-0 object is farther from the TCP at contact (m)")
    ap.add_argument("--detector", default="oracle", choices=["oracle", "qwen_sam"])
    ap.add_argument("--cloud-cams", nargs="+", default=["wrist", "front"])
    ap.add_argument("--crop-cams", nargs="+", default=["front"])
    ap.add_argument("--cloud-filter", default="voxel", choices=["voxel", "multiview"])
    ap.add_argument("--voxel-m", type=float, default=0.005)
    ap.add_argument("--vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--segmenter", default="sam2.1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    grounder = segmenter = None
    if a.detector == "qwen_sam":
        from entry_vla.perception.grounding import Qwen3VLGrounder
        from entry_vla.perception.segmentation import make_segmenter
        grounder, segmenter = Qwen3VLGrounder(a.vlm, device=a.device), make_segmenter(a.segmenter, a.device)
    rows = [r for r in list_contact_episodes(a.out_root, a.raw_root) if not a.tasks or r["task"] in a.tasks]
    index = {"format": "entry_vla.entry_diffusion_cache/1", "out_root": a.out_root, "raw_root": a.raw_root,
             "detector": "oracle_contact_object_mask_ids" if a.detector == "oracle" else
             {"name": "qwen3vl_sam2.1_frame0", "vlm": a.vlm, "segmenter": a.segmenter, "cloud_cams": a.cloud_cams,
              "crop_cams": a.crop_cams, "cloud_filter": a.cloud_filter, "voxel_m": a.voxel_m},
             "split_rule": {"by": "seed", "val": a.val_seeds, "test": a.test_seeds},
             "params": {"n_points": a.n_points, "crop_size": a.crop_size, "max_object_dist": a.max_object_dist}, "episodes": [], "skipped": []}
    for i, r in enumerate(rows):
        rel = os.path.join(r["task"], f"{r['variation']}_{r['seed']}_{r['episode']}.npz")
        path = os.path.join(a.cache, rel)
        if a.force or not os.path.exists(path):
            sd = zlib.crc32(f"{r['task']}/{r['seed']}".encode())
            if a.detector == "oracle":
                s = episode_inputs(r["raw"], r["contact_json"], a.n_points, a.crop_size, seed=sd,
                                   max_object_dist=a.max_object_dist)
            else:
                s = episode_inputs_detected(r["raw"], r["contact_json"], grounder, segmenter, cloud_cams=a.cloud_cams,
                                            crop_cams=a.crop_cams, n_points=a.n_points, crop_size=a.crop_size, seed=sd,
                                            cloud_filter=a.cloud_filter, voxel_m=a.voxel_m)
            if s is None:
                index["skipped"].append({**r, "reason": "no split frame, or touched object not found/visible at frame 0"})
                print(f"[{i + 1}/{len(rows)}] skip {rel}", flush=True)
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path + ".tmp.npz", **s)
            os.replace(path + ".tmp.npz", path)
        z = np.load(path)
        index["episodes"].append({"task": r["task"], "variation": r["variation"], "seed": r["seed"],
                                  "episode": r["episode"], "split": seed_split(r["seed"], a.val_seeds, a.test_seeds),
                                  "file": rel, "crop_cams": z["crop_cams"].tolist(),
                                  "object_ids": z["object_ids"].tolist() if "object_ids" in z else None})
        print(f"[{i + 1}/{len(rows)}] {index['episodes'][-1]['split']:5s} {rel} "
              f"dist {z['object_tcp_dist_m'].round(3).tolist() if 'object_tcp_dist_m' in z else '-'} "
              f"crops {z['crop_cams'].tolist()} pts {int(z['n_obj_points_raw'])}", flush=True)
    index["episode_counts"] = {s: sum(e["split"] == s for e in index["episodes"]) for s in ("train", "val", "test")}
    json.dump(index, open(os.path.join(a.cache, "index.json"), "w"), indent=1)
    print(f"{index['episode_counts']}, {len(index['skipped'])} skipped, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
