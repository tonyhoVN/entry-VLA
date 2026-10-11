#!/usr/bin/env python3
"""Train the conditional entry-pose diffusion model (entry_vla.models.entry_diffusion) on the cache from
scripts/build_entry_diffusion_cache.py + scripts/embed_vlm_features.py.

Target canonicalisation: the parallel gripper is symmetric under a pi rotation about the TCP z axis, so
each target rotation is replaced by whichever of {R, R Rz(pi)} is closer to the train chordal mean;
errors are symmetric geodesic angles. VLM features are standardised with train statistics.
Augmentation: point subsampling + jitter, random camera-crop dropout.

Evaluation (val for model selection, test once at the end with the best-val checkpoint), S samples each:
  sample1  error of one random sample          best_of_S  oracle minimum over samples
  medoid   sample with minimum summed distance to the other samples (no oracle)
  baseline task_mean (mean train entry pose of the task, relative to the object centroid)

  python scripts/train_entry_diffusion.py --cache data/entry_train/diffusion_select_single --out runs/entry_diffusion
  python scripts/train_entry_diffusion.py --dry-run
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RZ_PI = np.diag([-1.0, -1.0, 1.0])


def so3_project(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def geodesic(Ra, Rb):
    c = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.arccos(np.clip(c, -1, 1))


def sym_geodesic(Ra, Rb):
    return np.minimum(geodesic(Ra, Rb), geodesic(Ra, Rb @ RZ_PI))


def canonicalize(R, ref):
    flip = geodesic(ref, R @ RZ_PI) < geodesic(ref, R)
    out = R.copy()
    out[flip] = R[flip] @ RZ_PI
    return out


def load(cache, index, max_views):
    out = {}
    for split in ("train", "val", "test"):
        eps = [e for e in index["episodes"] if e["split"] == split]
        rows = {k: [] for k in ("obj_points", "centroid", "text_feat", "crop_feat", "crop_mask", "tgt_pos_C", "tgt_rot", "task")}
        for e in eps:
            z = np.load(os.path.join(cache, e["file"]))
            if "text_feat" not in z:
                raise SystemExit(f"{e['file']} has no VLM features: run scripts/embed_vlm_features.py")
            cf = z["crop_feat"].astype(np.float32)[:max_views]
            pad = np.zeros((max_views, cf.shape[1]), np.float32)
            pad[:len(cf)] = cf
            rows["obj_points"].append(z["obj_points"])
            rows["centroid"].append(z["centroid"])
            rows["text_feat"].append(z["text_feat"].astype(np.float32))
            rows["crop_feat"].append(pad)
            rows["crop_mask"].append(np.arange(max_views) < len(cf))
            rows["tgt_pos_C"].append(z["tgt_pos_C"])
            rows["tgt_rot"].append(z["tgt_rot"])
            rows["task"].append(e["task"])
        out[split] = {k: np.stack(v) if k != "task" else np.array(v) for k, v in rows.items()} if eps else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data/entry_train/diffusion_select_single")
    ap.add_argument("--out", default="runs/entry_diffusion")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--max-views", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--diffusion-steps", type=int, default=100)
    ap.add_argument("--ddim-steps", type=int, default=25)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance weight (1 = off)")
    ap.add_argument("--cond-dropout", type=float, default=0.1)
    ap.add_argument("--view-dropout", type=float, default=0.3, help="drop each crop view with this probability")
    ap.add_argument("--jitter-m", type=float, default=0.002)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--no-centroid", action="store_true")
    ap.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not 1 <= a.steps <= 200_000:
        raise SystemExit("--steps must be in [1, 200000]")
    index = json.load(open(os.path.join(a.cache, "index.json")))
    print(json.dumps({"cache": a.cache, "episodes": index["episode_counts"], "vlm": index.get("vlm"),
                      "detector": index.get("detector")}))
    if a.dry_run:
        return
    import torch
    from entry_vla.models.entry_diffusion import EntryDiffusion, EntryDiffusionConfig

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = load(a.cache, index, a.max_views)
    tr = data["train"]
    R_ref = so3_project(tr["tgt_rot"].sum(0))
    for _ in range(3):
        R_ref = so3_project(canonicalize(tr["tgt_rot"], R_ref).sum(0))
    mu_t, sd_t = tr["text_feat"].mean(0), tr["text_feat"].std(0) + 1e-3
    vm = tr["crop_feat"][tr["crop_mask"]]
    mu_v, sd_v = vm.mean(0), vm.std(0) + 1e-3
    for d in data.values():
        if d is None:
            continue
        d["tgt_rot_c"] = canonicalize(d["tgt_rot"], R_ref).astype(np.float32)
        d["text_feat"] = ((d["text_feat"] - mu_t) / sd_t).astype(np.float32)
        d["crop_feat"] = np.where(d["crop_mask"][..., None], (d["crop_feat"] - mu_v) / sd_v, 0).astype(np.float32)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cfg = EntryDiffusionConfig(vlm_dim=tr["text_feat"].shape[1], n_steps=a.diffusion_steps, cond_dropout=a.cond_dropout,
                               use_vision=not a.no_vision, use_text=not a.no_text, use_centroid=not a.no_centroid)
    model = EntryDiffusion(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    os.makedirs(a.out, exist_ok=True)
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=a.precision == "bf16" and dev.type == "cuda")

    def to_batch(d, idx, train):
        n_store = d["obj_points"].shape[1]
        sel = np.stack([rng.choice(n_store, a.n_points, replace=n_store < a.n_points) for _ in idx]) if train else \
            np.broadcast_to(np.linspace(0, n_store - 1, a.n_points).astype(int), (len(idx), a.n_points))
        pts = np.take_along_axis(d["obj_points"][idx], sel[..., None], 1).copy()
        mask = d["crop_mask"][idx].copy()
        if train:
            pts[..., :3] += rng.normal(0, a.jitter_m, pts[..., :3].shape).astype(np.float32)
            drop = rng.random(mask.shape) < a.view_dropout
            keep_one = mask & ~drop
            mask = np.where(keep_one.any(1, keepdims=True), keep_one, mask)   # never drop every view
        T = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)  # noqa: E731
        return {"obj_points": T(pts), "centroid": T(d["centroid"][idx]), "text_feat": T(d["text_feat"][idx]),
                "crop_feat": T(d["crop_feat"][idx]), "crop_mask": T(mask),
                "tgt_pos_C": T(d["tgt_pos_C"][idx]), "tgt_rot": T(d["tgt_rot_c"][idx])}

    def evaluate(d, seed=123):
        model.eval()
        g = torch.Generator(device=dev).manual_seed(seed)
        P, R = [], []
        for i in range(0, len(d["task"]), 32):
            idx = np.arange(i, min(i + 32, len(d["task"])))
            with torch.autocast(**amp):
                p, r = model.sample(to_batch(d, idx, False), a.samples, a.ddim_steps, a.guidance, g)
            P.append(p.float().cpu().numpy())
            R.append(r.float().cpu().numpy())
        model.train()
        P, R = np.concatenate(P), np.concatenate(R)                        # (B,S,3), (B,S,3,3)
        tp, tR = d["tgt_pos_C"], d["tgt_rot"]
        pe = np.linalg.norm(P - tp[:, None], axis=-1)                     # (B,S)
        re = sym_geodesic(R, tR[:, None])
        cost = pe + 0.1 * re
        dpp = np.linalg.norm(P[:, :, None] - P[:, None], axis=-1) + 0.1 * sym_geodesic(R[:, :, None], R[:, None])
        med = dpp.sum(-1).argmin(1)
        b = np.arange(len(pe))
        best = cost.argmin(1)
        per = {"sample1_pos_m": pe[:, 0], "sample1_rot_rad": re[:, 0], "medoid_pos_m": pe[b, med],
               "medoid_rot_rad": re[b, med], "best_of_S_pos_m": pe[b, best], "best_of_S_rot_rad": re[b, best],
               "sample_spread_m": np.linalg.norm(P - P.mean(1, keepdims=True), axis=-1).mean(1)}
        res = {k: float(v.mean()) for k, v in per.items()}
        means = {t: (tr["tgt_pos_C"][tr["task"] == t].mean(0), so3_project(tr["tgt_rot_c"][tr["task"] == t].sum(0)))
                 for t in np.unique(tr["task"])}
        gp, gR = tr["tgt_pos_C"].mean(0), R_ref
        bp = np.stack([means.get(t, (gp, gR))[0] for t in d["task"]])
        bR = np.stack([means.get(t, (gp, gR))[1] for t in d["task"]])
        bpe, bre = np.linalg.norm(bp - tp, axis=-1), sym_geodesic(bR, tR)
        res["task_mean_pos_m"], res["task_mean_rot_rad"] = float(bpe.mean()), float(bre.mean())
        res["per_task"] = {t: {"medoid_pos_m": float(per["medoid_pos_m"][d["task"] == t].mean()),
                               "medoid_rot_rad": float(per["medoid_rot_rad"][d["task"] == t].mean()),
                               "best_of_S_pos_m": float(per["best_of_S_pos_m"][d["task"] == t].mean()),
                               "task_mean_pos_m": float(bpe[d["task"] == t].mean()),
                               "task_mean_rot_rad": float(bre[d["task"] == t].mean()), "n": int((d["task"] == t).sum())}
                           for t in np.unique(d["task"])}
        return res

    log = open(os.path.join(a.out, "train_log.jsonl"), "w")
    best, t0, n = None, time.time(), len(tr["task"])
    for step in range(1, a.steps + 1):
        idx = rng.choice(n, a.batch_size, replace=n < a.batch_size)
        with torch.autocast(**amp):
            loss = model.loss(to_batch(tr, idx, True))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            v = evaluate(data["val"])
            rec = {"step": step, "loss": float(loss), **{f"val_{k}": round(x, 4) for k, x in v.items() if k != "per_task"},
                   "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            score = v["medoid_pos_m"] + 0.1 * v["medoid_rot_rad"]
            if best is None or score < best[0]:
                best = (score, step)
                torch.save({"model": model.state_dict(), "config": vars(cfg), "args": vars(a), "step": step,
                            "R_ref": R_ref, "feat_norm": {"mu_t": mu_t, "sd_t": sd_t, "mu_v": mu_v, "sd_v": sd_v}},
                           os.path.join(a.out, "best.pt"))
    ck = torch.load(os.path.join(a.out, "best.pt"), map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    report = {"best_step": best[1], "selection": "val medoid_pos_m + 0.1 * medoid_rot_rad",
              "episodes": index["episode_counts"], "detector": index.get("detector"), "vlm": index.get("vlm"),
              "conditions": {"vision": cfg.use_vision, "text": cfg.use_text, "centroid": cfg.use_centroid},
              "val": evaluate(data["val"]), "test": evaluate(data["test"]) if data["test"] is not None else None,
              "notes": "offline agreement with demo entry poses; Euclidean m, symmetric geodesic rad; "
                       "test evaluated once with the best-val checkpoint"}
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    for s in ("val", "test"):
        r = report[s]
        if r:
            print(s, {k: round(r[k], 4) for k in ("medoid_pos_m", "medoid_rot_rad", "best_of_S_pos_m",
                                                  "sample1_pos_m", "task_mean_pos_m", "task_mean_rot_rad")})
    print(f"done in {time.time() - t0:.0f}s -> {a.out}/report.json")


if __name__ == "__main__":
    main()
