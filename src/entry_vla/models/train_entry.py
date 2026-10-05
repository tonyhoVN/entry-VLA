"""Entry-proposal training / evaluation (user-run) and a tiny CPU forward check (agent-safe).

``train`` runs optimisation and is never invoked by tests or by the agent.
``forward_check`` builds a tiny randomly initialised model and runs one CPU
forward pass + loss + decode on synthetic tensors (no optimizer step).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

HARD_MAX_STEPS = 200_000


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    m, t, d = cfg["model"], cfg["train"], cfg["data"]
    problems = []
    if not 1 <= int(t["steps"]) <= HARD_MAX_STEPS:
        problems.append(f"train.steps must be in [1, {HARD_MAX_STEPS}]")
    if int(m["num_candidates"]) < 1:
        problems.append("model.num_candidates >= 1")
    if d.get("pose_frame") == "region_point" and not d.get("geometry_source"):
        problems.append("pose_frame=region_point requires data.geometry_source (labelled test-time source)")
    if t["precision"] not in ("fp32", "bf16", "fp16"):
        problems.append("train.precision must be fp32|bf16|fp16")
    root = d["episodes_root"]
    n_eps = len([x for x in os.listdir(root) if os.path.exists(os.path.join(root, x, "meta.json"))]) if os.path.isdir(root) else 0
    return {"problems": problems, "episodes_found": n_eps, "splits_file": d.get("splits"),
            "effective_batch": int(t["batch_size"]) * int(t.get("grad_accum", 1))}


def build_samples(cfg: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    from ..data import episode as E
    from ..data.entry_dataset import EntrySampleConfig, make_samples
    from .text import make_text_encoder
    d = cfg["data"]
    splits = json.load(open(d["splits"]))["assignment"] if d.get("splits") else None
    scfg = EntrySampleConfig(**{k: v for k, v in d.items() if k in EntrySampleConfig.__dataclass_fields__})
    enc = make_text_encoder(cfg.get("text_encoder", {}))
    out = []
    for path in E.list_episodes(d["episodes_root"]):
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        if splits is not None and splits.get(meta["episode_id"]) != split:
            continue
        ep = E.load_episode(path, cameras=scfg.cameras or [c for c, m in meta["cameras"].items() if not m.get("moving")])
        out += make_samples(ep, scfg, enc([meta.get("instruction", "")])[0], seed=int(cfg.get("seed", 0)))
    return out


def _to_torch(batch: Dict[str, np.ndarray], device):
    import torch
    t = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
    model_in = {"points": t["points"].float(), "points_mask": t["points_mask"].bool(), "text": t["text"].float(),
                "state": t["state"].float(), "geometry": t["geometry"].float()}
    tgt = {"pos": t["tgt_pos"].float(), "rot": t["tgt_rot"].float(), "grip_open": t["tgt_grip_open"].float(),
           "role": t["tgt_role"].long()}
    return model_in, tgt


def forward_check(num_candidates: int = 4, n_points: int = 64, batch: int = 2, seed: int = 0) -> Dict[str, Any]:
    """Tiny CPU forward pass with random weights (shape/decoding validation only)."""
    import torch
    from .entry_proposal import EntryModelConfig, EntryProposalNet, LossConfig, collapse_stats, decode_candidates, min_over_k_loss
    torch.manual_seed(seed)
    cfg = EntryModelConfig(num_candidates=num_candidates, point_feat_dim=16, hidden=32, text_dim=8, state_dim=34)
    net = EntryProposalNet(cfg).eval()
    rng = np.random.default_rng(seed)
    b = {"points": torch.tensor(rng.normal(size=(batch, n_points, 6)), dtype=torch.float32),
         "points_mask": torch.ones(batch, n_points, dtype=torch.bool), "text": torch.zeros(batch, 8),
         "state": torch.zeros(batch, 34), "geometry": torch.zeros(batch, cfg.geometry_dim)}
    b["points_mask"][0, n_points // 2:] = False
    with torch.no_grad():
        out = net(b)
        R = out["rot"]
        orth = (R.transpose(-1, -2) @ R - torch.eye(3)).abs().max().item()
        det = torch.linalg.det(R)
        tgt = {"pos": torch.zeros(batch, 2, 3), "rot": torch.eye(3).expand(batch, 2, 3, 3).clone(),
               "grip_open": torch.ones(batch, 2), "role": torch.zeros(batch, 2, dtype=torch.long)}
        loss = min_over_k_loss(out, tgt, LossConfig())
        stats = collapse_stats(out, loss["winner"])
        dec = decode_candidates(out, torch.eye(4).expand(batch, 4, 4))
    return {"pos_shape": list(out["pos"].shape), "rot_shape": list(R.shape), "cand_logits_shape": list(out["cand_logits"].shape),
            "max_orthonormality_err": orth, "min_det": float(det.min()), "loss": float(loss["loss"]),
            "collapse": stats, "decoded_candidates": len(dec[0]), "decoded_T_shape": list(dec[0][0]["T_WE"]["left"].shape),
            "params": sum(p.numel() for p in net.parameters()), "optimizer_steps": 0}


def train(cfg: Dict[str, Any], run_dir: str) -> Dict[str, Any]:  # pragma: no cover - user-run
    import torch
    from ..data.entry_dataset import collate
    from .entry_proposal import EntryModelConfig, EntryProposalNet, LossConfig, collapse_stats, min_over_k_loss
    t = cfg["train"]
    torch.manual_seed(int(cfg.get("seed", 0)))
    samples = build_samples(cfg, "train")
    if t.get("tiny_overfit"):
        samples = samples[: int(t.get("tiny_overfit_samples", 8))]
    if not samples:
        raise RuntimeError("no training samples (check annotations / splits)")
    mcfg = EntryModelConfig(**{**cfg["model"], "state_dim": int(samples[0]["state"].shape[0]),
                               "text_dim": int(samples[0]["text"].shape[0])})
    if t["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {t['device']} but CUDA is unavailable (no silent CPU fallback for training)")
    dev = torch.device(t["device"])
    net = EntryProposalNet(mcfg).to(dev)
    lcfg = LossConfig(**cfg.get("loss", {}))
    opt = torch.optim.AdamW(net.parameters(), lr=float(t["lr"]), weight_decay=float(t.get("weight_decay", 1e-4)))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[t["precision"]]
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    log = open(os.path.join(run_dir, "train_log.jsonl"), "a")
    accum = int(t.get("grad_accum", 1))
    t0 = time.time()
    for step in range(1, int(t["steps"]) + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            idx = rng.choice(len(samples), size=min(int(t["batch_size"]), len(samples)), replace=False)
            x, y = _to_torch(collate([samples[i] for i in idx]), dev)
            with torch.autocast(device_type=dev.type, dtype=dtype, enabled=t["precision"] != "fp32" and dev.type == "cuda"):
                out = net(x)
            out = {k: v.float() for k, v in out.items()}
            out["rot"] = out["rot"].float()
            L = min_over_k_loss(out, y, lcfg)
            (L["loss"] / accum).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), float(t.get("grad_clip", 1.0)))
        opt.step()
        if step % int(t.get("log_every", 50)) == 0 or step == 1:
            rec = {"step": step, "loss": float(L["loss"]), "min_pos_err_m": float(L["min_pos_err_m"]),
                   "min_rot_err_rad": float(L["min_rot_err_rad"]), **collapse_stats(out, L["winner"]),
                   "elapsed_s": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
        if step % int(t.get("save_every", 1000)) == 0 or step == int(t["steps"]):
            torch.save({"model": net.state_dict(), "model_cfg": asdict(mcfg), "data_cfg": cfg["data"],
                        "text_encoder": cfg.get("text_encoder", {}), "step": step},
                       os.path.join(run_dir, f"entry_model_{step:07d}.pt"))
    log.close()
    return {"run_dir": run_dir, "steps": int(t["steps"]), "train_samples": len(samples)}


def evaluate(cfg: Dict[str, Any], checkpoint: str, split: str) -> Dict[str, Any]:  # pragma: no cover - user-run
    """Offline entry-pose errors vs annotated demo entries for the model and the
    nearest-demo / mean-canonical baselines (agreement only; downstream success
    must come from rollouts)."""
    import torch
    from ..data.entry_dataset import collate
    from ..evaluation.metrics import entry_pose_errors
    from .entry_proposal import EntryModelConfig, EntryProposalNet
    ck = torch.load(checkpoint, map_location="cpu")
    net = EntryProposalNet(EntryModelConfig(**ck["model_cfg"])).eval()
    net.load_state_dict(ck["model"])
    test, train = build_samples(cfg, split), build_samples(cfg, "train")
    if not test:
        raise RuntimeError(f"no {split} samples")
    x, y = _to_torch(collate(test), "cpu")
    with torch.no_grad():
        out = net(x)
    true = np.stack([np.eye(4)] * (2 * len(test)))
    true[:, :3, :3] = y["rot"].numpy().reshape(-1, 3, 3); true[:, :3, 3] = y["pos"].numpy().reshape(-1, 3)
    def to_T(pos, rot):
        T = np.repeat(np.eye(4)[None], len(pos), 0); T[:, :3, :3] = rot; T[:, :3, 3] = pos
        return T
    top1 = out["cand_logits"].argmax(1)
    b = torch.arange(len(test))
    res = {"model_top1": entry_pose_errors(to_T(out["pos"][b, top1].reshape(-1, 3).numpy(), out["rot"][b, top1].reshape(-1, 3, 3).numpy()), true)}
    cost = (out["pos"] - y["pos"][:, None]).abs().sum((-1, -2))
    best = cost.argmin(1)
    res["model_oracle_best_of_k"] = entry_pose_errors(to_T(out["pos"][b, best].reshape(-1, 3).numpy(), out["rot"][b, best].reshape(-1, 3, 3).numpy()), true)
    trp = np.stack([s["tgt_pos"] for s in train]); trr = np.stack([s["tgt_rot"] for s in train])
    trg = np.stack([s["geometry"] for s in train]); teg = np.stack([s["geometry"] for s in test])
    nn_idx = np.argmin(((teg[:, None] - trg[None]) ** 2).sum(-1), 1) if np.any(trg) else np.zeros(len(test), int)
    res["nearest_demo"] = entry_pose_errors(to_T(trp[nn_idx].reshape(-1, 3), trr[nn_idx].reshape(-1, 3, 3)), true)
    from ..geometry.rotations import project_to_so3
    R_mean = np.stack([project_to_so3(trr[:, i].mean(0)) for i in range(2)])     # chordal mean per arm
    res["mean_canonical"] = entry_pose_errors(to_T(np.repeat(trp.mean(0)[None], len(test), 0).reshape(-1, 3),
                                                   np.repeat(R_mean[None], len(test), 0).reshape(-1, 3, 3)), true)
    res["n_test"], res["split"] = len(test), split
    res["note"] = "pose agreement with demonstrations is not downstream success"
    return res
