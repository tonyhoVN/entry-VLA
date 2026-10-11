#!/usr/bin/env python3
"""Offline evaluation of an X-VLA checkpoint on held-out RLBench manipulation-phase episodes.

For every sample of the val meta (X-VLA's own non-training reader: all frames, no augmentation) the
model predicts a chunk of absolute EE6D targets from the images, instruction and current pose; errors
against the demo's measured future trajectory (the training target) are reported per horizon step:
position (m), rotation (rad, geodesic from the 6D layout), gripper accuracy (closed if p > 0.5).
Baseline: hold the current pose. This measures imitation of demo trajectories, NOT task success
(closed-loop RLBench rollouts need a policy service between the xvla and peract2 envs).

  CUDA_VISIBLE_DEVICES=1 python scripts/xvla/eval_xvla_offline.py --xvla-root ../X-VLA \
      --ckpt runs/xvla_close_microwave/ckpt-3000 \
      --metas data/xvla/rlbench_manip_close_microwave/meta_val.json --out runs/xvla_close_microwave/eval_ckpt-3000.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def rot6d_to_R(v):
    """X-VLA layout: v = R[:, :2].reshape(6) (row-major), i.e. columns interleaved."""
    a1, a2 = v[..., 0:5:2], v[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True).clip(1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], -1)


def geodesic(Ra, Rb):
    c = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.arccos(np.clip(c, -1, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xvla-root", default=os.environ.get("XVLA_ROOT", "../X-VLA"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--metas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain-id", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    import xvla_rlbench
    xvla_rlbench.register(a.xvla_root, a.domain_id if a.domain_id is not None else xvla_rlbench.DEFAULT_DOMAIN_ID)
    import torch
    from datasets.dataset import InfiniteDataReader
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    t0 = time.time()
    model = XVLA.from_pretrained(a.ckpt).to(a.device).eval()
    proc = XVLAProcessor.from_pretrained(a.ckpt)
    reader = InfiniteDataReader(a.metas, num_actions=model.num_actions, num_views=3, training=False,
                                action_mode=model.action_mode)
    P, G, Pc, n = [], [], [], 0

    def flush(buf):
        lang = proc.encode_language([b["language_instruction"] for b in buf])
        inp = {"input_ids": lang["input_ids"].to(a.device),
               "image_input": torch.stack([b["image_input"] for b in buf]).to(a.device),
               "image_mask": torch.stack([b["image_mask"] for b in buf]).to(a.device),
               "domain_id": torch.stack([b["domain_id"] for b in buf]).to(a.device),
               "proprio": torch.stack([b["proprio"] for b in buf]).float().to(a.device)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.generate_actions(**inp, steps=a.denoise_steps)
        P.append(pred.float().cpu().numpy())
        G.append(torch.stack([b["action"] for b in buf]).numpy())
        Pc.append(inp["proprio"].cpu().numpy())

    buf = []
    for s in reader:
        buf.append(s)
        n += 1
        if len(buf) == a.batch_size:
            flush(buf)
            buf = []
        if n >= a.max_samples:
            break
    if buf:
        flush(buf)
    P, G, Pc = np.concatenate(P), np.concatenate(G), np.concatenate(Pc)       # (N,H,20), (N,H,20), (N,20)
    pos_e = np.linalg.norm(P[..., :3] - G[..., :3], axis=-1)                   # (N,H)
    rot_e = geodesic(rot6d_to_R(P[..., 3:9]), rot6d_to_R(G[..., 3:9]))
    grip_acc = ((P[..., 9] > 0.5) == (G[..., 9] > 0.5)).astype(float)
    hold = np.broadcast_to(Pc[:, None], G.shape)
    hold_pos = np.linalg.norm(hold[..., :3] - G[..., :3], axis=-1)
    hold_rot = geodesic(rot6d_to_R(hold[..., 3:9]), rot6d_to_R(G[..., 3:9]))
    H = P.shape[1]
    idx = sorted({0, H // 4, H // 2, H - 1})
    res = {"ckpt": a.ckpt, "metas": a.metas, "samples": int(len(P)), "horizon": int(H),
           "model": {"pos_m_mean": float(pos_e.mean()), "rot_rad_mean": float(rot_e.mean()),
                     "grip_acc": float(grip_acc.mean()),
                     "pos_m_at_step": {int(i + 1): float(pos_e[:, i].mean()) for i in idx},
                     "rot_rad_at_step": {int(i + 1): float(rot_e[:, i].mean()) for i in idx}},
           "hold_current_pose": {"pos_m_mean": float(hold_pos.mean()), "rot_rad_mean": float(hold_rot.mean()),
                                 "pos_m_at_step": {int(i + 1): float(hold_pos[:, i].mean()) for i in idx}},
           "note": "offline imitation error vs measured demo future poses; not closed-loop success",
           "elapsed_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
