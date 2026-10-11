#!/usr/bin/env python3
"""Predict entry poses for a new RLBench episode with the trained entry diffusion model, from frame 0 only.

Pipeline (deployable, no simulator masks, no future frames): Qwen3-VL box + SAM 2.1 mask of the
instruction's target object in front/overhead/wrist -> fused front+overhead object cloud (canonical frame,
centroid c) + crops -> frozen Qwen3-VL text/crop features (standardised with the training statistics
stored in the checkpoint) -> S DDIM samples of [p_C, R]. Candidates are ranked by medoid score (summed
distance to the other samples, 0.1 m per rad) and returned in the world frame (p_W = p_C + c,
quaternion xyzw) for the motion planner.

  python scripts/predict_entry_diffusion.py --ckpt runs/entry_diffusion_close_microwave/best.pt \
      --episode data/raw/rlbench_single/close_microwave/variation0/seed4242/episodes/episode0 --out entry.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
RZ_PI = np.diag([-1.0, -1.0, 1.0])


def geodesic(Ra, Rb):
    c = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.arccos(np.clip(c, -1, 1))


def R_to_quat_xyzw(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episode", required=True, help="raw RLBench episode (frame 0 is used)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--top", type=int, default=8, help="candidates written, best first")
    ap.add_argument("--ddim-steps", type=int, default=None)
    ap.add_argument("--vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--segmenter", default="sam2.1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from entry_vla.data.entry_diffusion_data import detect_object_frame0
    from entry_vla.data.peract2_raw import load_pickle
    from entry_vla.models.entry_diffusion import EntryDiffusion, EntryDiffusionConfig
    from entry_vla.models.vlm_features import VLMFeatures
    from entry_vla.perception.grounding import Qwen3VLGrounder
    from entry_vla.perception.segmentation import make_segmenter

    t0 = time.time()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    targs, fn = ck["args"], ck["feat_norm"]
    cache_index = json.load(open(os.path.join(targs["cache"], "index.json")))
    det_cfg = cache_index.get("detector") if isinstance(cache_index.get("detector"), dict) else {}
    instruction = load_pickle(os.path.join(a.episode, "variation_descriptions.pkl"))[0]
    tr = [e for e in cache_index["episodes"] if e["split"] == "train"]
    grip_open = float(np.mean([float(np.load(os.path.join(targs["cache"], e["file"]))["tgt_grip_open"]) for e in tr]) >= 0.5)

    grounder = Qwen3VLGrounder(a.vlm, device=a.device)
    segmenter = make_segmenter(a.segmenter, a.device)
    s = detect_object_frame0(a.episode, instruction, grounder, segmenter,
                             cloud_cams=det_cfg.get("cloud_cams", ["wrist", "front"]),
                             crop_cams=det_cfg.get("crop_cams", ["front"]),
                             cloud_filter=det_cfg.get("cloud_filter", "multiview"), voxel_m=det_cfg.get("voxel_m", 0.005),
                             n_points=cache_index["params"]["n_points"], seed=a.seed)
    if s is None:
        raise SystemExit("object not detected in frame 0")
    vlm = VLMFeatures(grounder.model, grounder.processor, a.device)     # same frozen Qwen3-VL weights
    text = (vlm.text(instruction) - fn["mu_t"]) / fn["sd_t"]
    cf = vlm.crops(list(s["crops"]))[:targs["max_views"]]
    crop_feat = np.zeros((targs["max_views"], cf.shape[1]), np.float32)
    crop_feat[:len(cf)] = (cf - fn["mu_v"]) / fn["sd_v"]
    mask = np.arange(targs["max_views"]) < len(cf)
    n_store = s["obj_points"].shape[0]
    pts = s["obj_points"][np.linspace(0, n_store - 1, targs["n_points"]).astype(int)]

    dev = torch.device(a.device)
    model = EntryDiffusion(EntryDiffusionConfig(**ck["config"])).to(dev).eval()
    model.load_state_dict(ck["model"])
    T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x)[None], dtype=dt, device=dev)  # noqa: E731
    batch = {"obj_points": T(pts), "centroid": T(s["centroid"]), "text_feat": T(text), "crop_feat": T(crop_feat),
             "crop_mask": T(mask, torch.bool)}
    g = torch.Generator(device=dev).manual_seed(a.seed)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=targs.get("precision") == "bf16"):
        P, R = model.sample(batch, a.samples, a.ddim_steps or targs["ddim_steps"], targs["guidance"], g)
    P, R = P[0].float().cpu().numpy(), R[0].float().cpu().numpy()
    sym = lambda Ra, Rb: np.minimum(geodesic(Ra, Rb), geodesic(Ra, Rb @ RZ_PI))  # noqa: E731
    d = np.linalg.norm(P[:, None] - P[None], axis=-1) + 0.1 * sym(R[:, None], R[None])
    order = np.argsort(d.sum(1))
    c = s["centroid"].astype(np.float64)
    cands = [{"rank": i, "pos_world": (P[k] + c).tolist(), "quat_xyzw": R_to_quat_xyzw(R[k]).tolist(),
              "pos_canonical": P[k].tolist(), "medoid_score": float(d[k].sum())} for i, k in enumerate(order[:a.top])]
    out = {"format": "entry_vla.entry_prediction/1", "episode": os.path.abspath(a.episode), "instruction": instruction,
           "ckpt": os.path.abspath(a.ckpt), "frame": 0, "centroid_world": c.tolist(),
           "detections": json.loads(str(s["detections"])), "n_object_points": int(s["n_obj_points_raw"]),
           "samples": a.samples, "sample_spread_m": float(np.linalg.norm(P - P.mean(0), axis=1).mean()),
           "gripper_open": grip_open, "candidates": cands,
           "note": "entry from Qwen3-VL+SAM2.1 frame-0 object cloud + diffusion; gripper = majority entry state of the training set (no gripper head)",
           "elapsed_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("centroid_world", "n_object_points", "sample_spread_m", "elapsed_s")}),
          "\ntop candidate", cands[0])


if __name__ == "__main__":
    main()
