#!/usr/bin/env python3
"""Train the single-arm entry-pose model on the RLBench contact-split cache
(scripts/build_rlbench_entry_cache.py). Model selection uses the val split only; the test split is
evaluated once, with the best-val checkpoint, at the end.

Baselines on the same samples:
  stay       current TCP pose (no preparation motion)
  task_mean  mean train entry pose of the same task (rotation: chordal mean)

  python scripts/train_single_entry.py --cache data/entry_train/rlbench_select_single --out runs/single_entry
  python scripts/train_single_entry.py --dry-run     # validate config/cache only, no training
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from entry_vla.geometry.rotations import rot6d_to_matrix  # noqa: E402
from entry_vla.models.text import HashingTextEncoder  # noqa: E402


def load_split(cache, index, split, enc):
    eps = [e for e in index["episodes"] if e["split"] == split]
    if not eps:
        return None
    cols = {k: [] for k in ("points", "state", "text", "tgt_T", "tgt_grip_open", "task")}
    for e in eps:
        z = np.load(os.path.join(cache, e["file"]))
        n = z["state"].shape[0]
        cols["points"].append(z["points"])
        cols["state"].append(z["state"])
        cols["text"].append(np.repeat(enc([str(z["instruction"])]), n, 0))
        cols["tgt_T"].append(z["tgt_T"])
        cols["tgt_grip_open"].append(z["tgt_grip_open"])
        cols["task"] += [e["task"]] * n
    out = {k: (np.concatenate(v) if k != "task" else np.array(v)) for k, v in cols.items()}
    out["n_episodes"] = len(eps)
    return out


def so3_project(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def baselines(train, data):
    """(B,3) positions and (B,3,3) rotations for each baseline."""
    st = data["state"]
    stay = (st[:, :3], np.stack([rot6d_to_matrix(r) for r in st[:, 3:9]]))
    means = {}
    for t in np.unique(train["task"]):
        m = train["task"] == t
        means[t] = (train["tgt_T"][m, :3, 3].mean(0), so3_project(train["tgt_T"][m, :3, :3].sum(0)))
    gp, gR = train["tgt_T"][:, :3, 3].mean(0), so3_project(train["tgt_T"][:, :3, :3].sum(0))
    tm = (np.stack([means.get(t, (gp, gR))[0] for t in data["task"]]),
          np.stack([means.get(t, (gp, gR))[1] for t in data["task"]]))
    return {"stay": stay, "task_mean": tm}


def summarize(per_sample, tasks):
    out = {k: float(np.mean(v)) for k, v in per_sample.items()}
    out["per_task"] = {t: {k: float(np.mean(v[tasks == t])) for k, v in per_sample.items()} for t in np.unique(tasks)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data/entry_train/rlbench_select_single")
    ap.add_argument("--out", default="runs/single_entry")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-points", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-candidates", type=int, default=4)
    ap.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--jitter-m", type=float, default=0.003, help="xyz noise on training points")
    ap.add_argument("--no-symmetric", action="store_true", help="do not treat gripper as pi-symmetric about z")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not 1 <= a.steps <= 100_000:
        raise SystemExit("--steps must be in [1, 100000]")
    index = json.load(open(os.path.join(a.cache, "index.json")))
    counts = index["sample_counts"]
    print(json.dumps({"cache": a.cache, "sample_counts": counts, "skipped": len(index["skipped"])}))
    if a.dry_run:
        return
    import torch
    from entry_vla.models.single_entry import (SingleEntryConfig, SingleEntryNet, SingleLossConfig,
                                               entry_metrics, rot_error, single_entry_loss)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    enc = HashingTextEncoder(64)
    data = {s: load_split(a.cache, index, s, enc) for s in ("train", "val", "test")}
    tr = data["train"]
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    sym = not a.no_symmetric
    model = SingleEntryNet(SingleEntryConfig(num_candidates=a.num_candidates, symmetric_gripper=sym)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    lcfg = SingleLossConfig()
    os.makedirs(a.out, exist_ok=True)
    T = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731

    def batch(d, idx, train):
        n_store = d["points"].shape[1]
        sel = np.stack([rng.choice(n_store, a.n_points, replace=False) for _ in idx]) if train else \
            np.broadcast_to(np.linspace(0, n_store - 1, a.n_points).astype(int), (len(idx), a.n_points))
        pts = np.take_along_axis(d["points"][idx], sel[..., None], 1).copy()
        if train and a.jitter_m > 0:
            pts[..., :3] += rng.normal(0, a.jitter_m, pts[..., :3].shape).astype(np.float32)
        return (T(pts), torch.ones(len(idx), a.n_points, dtype=torch.bool, device=dev), T(d["text"][idx]),
                T(d["state"][idx]), T(d["tgt_T"][idx, :3, 3]), T(d["tgt_T"][idx, :3, :3]), T(d["tgt_grip_open"][idx]))

    def evaluate(d):
        model.eval()
        per = {}
        with torch.no_grad():
            for i in range(0, len(d["state"]), 256):
                idx = np.arange(i, min(i + 256, len(d["state"])))
                pts, msk, txt, st, tp, tR, tg = batch(d, idx, False)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=a.precision == "bf16" and dev.type == "cuda"):
                    pred = model(pts, msk, txt, st)
                for k, v in entry_metrics(pred, tp, tR, tg, sym).items():
                    per.setdefault(k, []).append(v.float().cpu().numpy())
        model.train()
        per = {k: np.concatenate(v) for k, v in per.items()}
        res = {"model": summarize(per, d["task"])}
        for name, (bp, bR) in baselines(tr, d).items():
            tp, tR = T(d["tgt_T"][:, :3, 3]), T(d["tgt_T"][:, :3, :3])
            bp_t, bR_t = T(bp.astype(np.float32)), T(bR.astype(np.float32))
            res[name] = summarize({"top1_pos_m": (bp_t - tp).norm(dim=-1).cpu().numpy(),
                                   "top1_rot_rad": rot_error(bR_t, tR, sym).cpu().numpy()}, d["task"])
        return res

    log = open(os.path.join(a.out, "train_log.jsonl"), "w")
    best, t0, n = None, time.time(), len(tr["state"])
    for step in range(1, a.steps + 1):
        idx = rng.choice(n, a.batch_size, replace=n < a.batch_size)
        pts, msk, txt, st, tp, tR, tg = batch(tr, idx, True)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=a.precision == "bf16" and dev.type == "cuda"):
            pred = model(pts, msk, txt, st)
        loss, _ = single_entry_loss(pred, tp, tR, tg, lcfg, sym)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            v = evaluate(data["val"])["model"]
            rec = {"step": step, "loss": float(loss), "val_top1_pos_m": v["top1_pos_m"], "val_top1_rot_rad": v["top1_rot_rad"],
                   "val_minK_pos_m": v["minK_pos_m"], "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            score = v["top1_pos_m"] + 0.1 * v["top1_rot_rad"]   # 0.1 m per rad
            if best is None or score < best[0]:
                best = (score, step)
                torch.save({"model": model.state_dict(), "config": vars(model.cfg), "args": vars(a), "step": step},
                           os.path.join(a.out, "best.pt"))
    ck = torch.load(os.path.join(a.out, "best.pt"), map_location=dev)
    model.load_state_dict(ck["model"])
    report = {"best_step": best[1], "selection": "val top1_pos_m + 0.1 * top1_rot_rad", "sample_counts": counts,
              "episodes": {s: (data[s] or {}).get("n_episodes", 0) for s in data},
              "val": evaluate(data["val"]), "test": evaluate(data["test"]) if data["test"] is not None else None,
              "notes": "errors: Euclidean metres, geodesic radians (gripper pi-symmetric about TCP z unless --no-symmetric); "
                       "test evaluated once with the best-val checkpoint"}
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    for s in ("val", "test"):
        if report[s]:
            print(s, {k: (round(report[s][k]["top1_pos_m"], 4), round(report[s][k]["top1_rot_rad"], 3))
                      for k in ("model", "stay", "task_mean")})
    print(f"done in {time.time() - t0:.0f}s -> {a.out}/report.json")


if __name__ == "__main__":
    main()
