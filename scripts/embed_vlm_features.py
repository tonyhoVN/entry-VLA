#!/usr/bin/env python3
"""Add frozen VLM features to the entry-diffusion cache (scripts/build_entry_diffusion_cache.py).

The same frozen Qwen3-VL model encodes both modalities (last decoder layer, bf16, no gradients):
  text_feat (D,)    bare instruction text (no chat template), mean over its tokens
  crop_feat (V, D)  each object crop alone (one image per prompt), mean over that image's vision tokens
Features are written into each episode npz (float16) with the model id; episodes that already carry
features from the same model are skipped unless --force. Uses the locally cached weights only.

  python scripts/embed_vlm_features.py --cache data/entry_train/diffusion_select_single
"""

import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data/entry_train/diffusion_select_single")
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from entry_vla.models.vlm_features import VLMFeatures

    index = json.load(open(os.path.join(a.cache, "index.json")))
    todo = []
    for e in index["episodes"]:
        z = np.load(os.path.join(a.cache, e["file"]))
        if a.force or "text_feat" not in z or str(z.get("vlm_model", "")) != a.model_id:
            todo.append(e)
    print(f"{len(todo)}/{len(index['episodes'])} episodes need features", flush=True)
    if not todo:
        return
    t0 = time.time()
    vlm = VLMFeatures.load(a.model_id, a.device)

    for i, e in enumerate(todo):
        path = os.path.join(a.cache, e["file"])
        z = dict(np.load(path))
        text = vlm.text(str(z["instruction"]))   # bare instruction: template tokens would dominate the mean
        crop_feat = vlm.crops(list(z["crops"]), a.batch)
        z.update(text_feat=text.astype(np.float16), crop_feat=crop_feat.astype(np.float16),
                 vlm_model=np.array(a.model_id), vlm_feature="last_hidden_state mean (bare instruction tokens | image tokens)")
        np.savez_compressed(path + ".tmp.npz", **z)
        os.replace(path + ".tmp.npz", path)
        print(f"[{i + 1}/{len(todo)}] {e['file']} text {text.shape} crops {len(crop_feat)}", flush=True)
    index["vlm"] = {"model_id": a.model_id, "dim": int(text.shape[0])}
    json.dump(index, open(os.path.join(a.cache, "index.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
