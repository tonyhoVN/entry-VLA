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
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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
    proc = AutoProcessor.from_pretrained(a.model_id, local_files_only=True)
    proc.tokenizer.padding_side = "right"
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model_id, dtype=torch.bfloat16,
                                                            local_files_only=True).to(a.device).eval()
    img_tok = model.config.image_token_id

    def encode(texts, images=None, image_only=False):
        inp = proc(text=texts, images=images, return_tensors="pt", padding=True).to(a.device)
        with torch.no_grad():
            h = model.model(**inp).last_hidden_state.float()
        m = (inp["input_ids"] == img_tok) if image_only else inp["attention_mask"].bool()
        return ((h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)).cpu().numpy()

    def chat(content):
        return proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                        add_generation_prompt=False)

    for i, e in enumerate(todo):
        path = os.path.join(a.cache, e["file"])
        z = dict(np.load(path))
        text = encode([str(z["instruction"])])[0]   # bare instruction: template tokens would dominate the mean
        crops = [Image.fromarray(c) for c in z["crops"]]
        feats = []
        for j in range(0, len(crops), a.batch):
            part = crops[j:j + a.batch]
            feats.append(encode([chat([{"type": "image"}])] * len(part), images=part, image_only=True))
        z.update(text_feat=text.astype(np.float16), crop_feat=np.concatenate(feats).astype(np.float16),
                 vlm_model=np.array(a.model_id), vlm_feature="last_hidden_state mean (bare instruction tokens | image tokens)")
        np.savez_compressed(path + ".tmp.npz", **z)
        os.replace(path + ".tmp.npz", path)
        print(f"[{i + 1}/{len(todo)}] {e['file']} text {text.shape} crops {len(crops)}", flush=True)
    index["vlm"] = {"model_id": a.model_id, "dim": int(text.shape[0])}
    json.dump(index, open(os.path.join(a.cache, "index.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
