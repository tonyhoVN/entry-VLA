#!/usr/bin/env python3
"""Build the single-arm entry-pose training cache from RLBench episodes with a contact phase split.

Each episode becomes <cache>/<task>/<variation>_<seed>_<episode>.npz (see entry_vla.data.rlbench_entry);
an index.json records the episode split (by seed: never adjacent frames) and skipped episodes.
Existing npz files are reused unless --force.

  python scripts/build_rlbench_entry_cache.py \
      --out-root data/entry_process/rlbench_select_single --raw-root data/raw/rlbench_single \
      --cache data/entry_train/rlbench_select_single
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from entry_vla.data.rlbench_entry import episode_samples, list_contact_episodes, seed_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default="data/entry_process/rlbench_select_single")
    ap.add_argument("--raw-root", default="data/raw/rlbench_single")
    ap.add_argument("--cache", default="data/entry_train/rlbench_select_single")
    ap.add_argument("--val-seeds", type=int, nargs="*", default=[15, 16, 17])
    ap.add_argument("--test-seeds", type=int, nargs="*", default=[18, 19, 20])
    ap.add_argument("--n-points", type=int, default=4096, help="points stored per sample (training subsamples)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-queries", type=int, default=12)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rows = list_contact_episodes(a.out_root, a.raw_root)
    if a.tasks:
        rows = [r for r in rows if r["task"] in a.tasks]
    index = {"format": "entry_vla.rlbench_entry_cache/1", "out_root": a.out_root, "raw_root": a.raw_root,
             "split_rule": {"by": "seed", "val": a.val_seeds, "test": a.test_seeds},
             "params": {"n_points": a.n_points, "stride": a.stride, "max_queries": a.max_queries},
             "episodes": [], "skipped": []}
    for i, r in enumerate(rows):
        rel = os.path.join(r["task"], f"{r['variation']}_{r['seed']}_{r['episode']}.npz")
        path = os.path.join(a.cache, rel)
        split = seed_split(r["seed"], a.val_seeds, a.test_seeds)
        if a.force or not os.path.exists(path):
            s = episode_samples(r["raw"], r["contact_json"], a.n_points, a.stride, a.max_queries,
                                seed=zlib.crc32(f"{r['task']}/{r['seed']}".encode()))
            if s is None:
                index["skipped"].append({**r, "reason": "split_frame undefined (contact within pre-frames)"})
                print(f"[{i + 1}/{len(rows)}] skip {rel}: no split", flush=True)
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path + ".tmp.npz", **s)
            os.replace(path + ".tmp.npz", path)
        n = int(np.load(path)["state"].shape[0])
        index["episodes"].append({"task": r["task"], "variation": r["variation"], "seed": r["seed"],
                                  "episode": r["episode"], "split": split, "file": rel, "n_samples": n})
        print(f"[{i + 1}/{len(rows)}] {split:5s} {rel} ({n} samples)", flush=True)
    counts = {s: sum(e["n_samples"] for e in index["episodes"] if e["split"] == s) for s in ("train", "val", "test")}
    index["sample_counts"] = counts
    json.dump(index, open(os.path.join(a.cache, "index.json"), "w"), indent=1)
    print(f"{len(index['episodes'])} episodes, {len(index['skipped'])} skipped, samples {counts}, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
