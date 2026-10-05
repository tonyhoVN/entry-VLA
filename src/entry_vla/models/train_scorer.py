"""Outcome-scorer training from probe records (user-run) + inference wrapper for ScoredSelector."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..geometry.rotations import matrix_to_rot6d
from ..geometry.transforms import invert_T
from ..sim.contracts import ARMS

HARD_MAX_STEPS = 200_000


def read_probes(paths: List[str]) -> List[Dict[str, Any]]:
    recs = []
    for p in paths:
        with open(p) as f:
            recs += [json.loads(l) for l in f if l.strip()]
    return recs


def label_version_key(rec: Dict[str, Any]) -> str:
    return json.dumps(rec["label_version"], sort_keys=True)


def candidate_vector(T_WE: Dict[str, np.ndarray], grip: Dict[str, float], q: Optional[Dict[str, Any]],
                     T_FW: np.ndarray) -> np.ndarray:
    T_inv = invert_T(np.asarray(T_FW, dtype=np.float64))
    parts = []
    for a in ARMS:
        T = T_inv @ np.asarray(T_WE[a])
        parts += [T[:3, 3], matrix_to_rot6d(T[:3, :3]), [float(grip[a] > 0.5)]]
    for a in ARMS:
        parts.append(np.zeros(7) if not q or q.get(a) is None else np.asarray(q[a]))
    return np.concatenate(parts).astype(np.float32)


def build_arrays(recs: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Attempted, arrived probes only (transit failures are labelled separately and masked out)."""
    rows = [r for r in recs if r.get("arrival") and not r["arrival"].get("failure") and "cache" in r["arrival"]]
    out: Dict[str, List[Any]] = {k: [] for k in ("points", "points_mask", "text", "state", "history", "candidate", "success",
                                                 "arm_success", "duration_s", "mask", "group")}
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    for r in rows:
        pth = r["pre_travel"]["cache"]
        if pth not in cache:
            with np.load(pth) as z:
                cache[pth] = {k: z[k] for k in z.files}
        c = cache[pth]
        cand = r["candidate"]
        out["points"].append(c["points"]); out["points_mask"].append(c["points_mask"]); out["text"].append(c["text"])
        out["state"].append(c["state"]); out["history"].append(c["history"])
        out["candidate"].append(candidate_vector({a: np.asarray(cand["T_WE"][a]) for a in ARMS}, cand["gripper"],
                                                 r.get("candidate_q"), c["T_FW"]))
        o = r["outcome"]
        out["success"].append(float(o["success"]))
        a_s = o.get("arm_success") or {}
        out["arm_success"].append([np.nan if a_s.get(a) is None else float(a_s[a]) for a in ARMS])
        out["duration_s"].append(float(o.get("skill_sim_time_s", 0.0)))
        out["mask"].append(1.0)
        out["group"].append(r["group"])
    return {k: (np.asarray(v) if k != "group" else v) for k, v in out.items()}


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t, d = cfg["train"], cfg["data"]
    problems = []
    if not 1 <= int(t["steps"]) <= HARD_MAX_STEPS:
        problems.append(f"train.steps must be in [1, {HARD_MAX_STEPS}]")
    files = [p for p in d["probe_files"] if os.path.exists(p)]
    if len(files) != len(d["probe_files"]):
        problems.append(f"missing probe files: {sorted(set(d['probe_files']) - set(files))}")
    return {"problems": problems, "probe_files": d["probe_files"]}


def _split_groups(groups: List[str], frac_val: float, salt: str) -> np.ndarray:
    import hashlib
    u = np.array([int(hashlib.sha256(f"{salt}:{g}".encode()).hexdigest()[:8], 16) / 16**8 for g in groups])
    return u < frac_val


def train(cfg: Dict[str, Any], run_dir: str) -> Dict[str, Any]:  # pragma: no cover - user-run
    import torch
    from .scorer import OutcomeScorer, ScorerConfig, scorer_loss
    t, d = cfg["train"], cfg["data"]
    recs = read_probes(d["probe_files"])
    versions = {label_version_key(r) for r in recs}
    if len(versions) != 1:
        raise RuntimeError(f"probe files mix {len(versions)} label versions (policy/controller/reset/timeout); "
                           "regenerate or train separate scorers")
    arr = build_arrays(recs)
    if len(arr["success"]) == 0:
        raise RuntimeError("no attempted+arrived probes to train on")
    is_val = _split_groups(arr["group"], float(d.get("val_fraction", 0.2)), str(d.get("salt", "scorer")))
    T = {k: torch.as_tensor(v) for k, v in arr.items() if k != "group"}
    scfg = ScorerConfig(**{**cfg.get("model", {}), "text_dim": int(arr["text"].shape[1]),
                           "state_dim": int(arr["state"].shape[1]), "history_dim": int(arr["history"].shape[1])})
    if t["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dev = torch.device(t["device"])
    net = OutcomeScorer(scfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=float(t["lr"]), weight_decay=float(t.get("weight_decay", 1e-4)))
    tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    log = open(os.path.join(run_dir, "train_log.jsonl"), "a")

    def batch(idx):
        b = {k: v[idx].to(dev) for k, v in T.items()}
        b["points"], b["text"], b["state"], b["history"], b["candidate"] = (b["points"].float(), b["text"].float(),
                                                                          b["state"].float(), b["history"].float(),
                                                                          b["candidate"].float())
        b["points_mask"] = b["points_mask"].bool()
        b["arm_success"] = b["arm_success"].float()
        return b
    t0 = time.time()
    for step in range(1, int(t["steps"]) + 1):
        idx = rng.choice(tr_idx, size=min(int(t["batch_size"]), len(tr_idx)), replace=False)
        b = batch(idx)
        L = scorer_loss(net(b), b)
        opt.zero_grad(set_to_none=True); L["loss"].backward(); opt.step()
        if step % int(t.get("log_every", 50)) == 0 or step == int(t["steps"]):
            rec = {"step": step, "loss": float(L["loss"]), "elapsed_s": time.time() - t0}
            if len(va_idx):
                with torch.no_grad():
                    bv = batch(va_idx)
                    ov = net(bv)
                    p = torch.sigmoid(ov["joint_logit"]).cpu().numpy()
                    y = bv["success"].cpu().numpy()
                    rec.update({"val_brier": float(np.mean((p - y) ** 2)), "val_n": int(len(va_idx)),
                                "val_success_rate": float(y.mean())})
            log.write(json.dumps(rec) + "\n"); log.flush()
    log.close()
    from dataclasses import asdict
    path = os.path.join(run_dir, "scorer.pt")
    torch.save({"model": net.state_dict(), "model_cfg": asdict(scfg), "label_version": json.loads(next(iter(versions))),
                "data_cfg": d}, path)
    return {"run_dir": run_dir, "checkpoint": path, "train_n": int(len(tr_idx)), "val_n": int(len(va_idx)),
            "note": "scores are learned estimates, not calibrated safety certificates"}


def make_score_fn(checkpoint: str, sample_cfg, text_emb: np.ndarray, expected_label_version: Dict[str, Any],
                  device: str = "cpu"):  # pragma: no cover - needs trained checkpoint
    """Wrap a trained scorer as ``score_fn(ctx, candidates)`` for ScoredSelector.
    Refuses checkpoints trained under a different label version."""
    import torch
    from ..data.entry_dataset import observation_inputs
    from .scorer import OutcomeScorer, ScorerConfig
    ck = torch.load(checkpoint, map_location=device)
    if json.dumps(ck["label_version"], sort_keys=True) != json.dumps(expected_label_version, sort_keys=True):
        raise RuntimeError("scorer label_version differs from the evaluation policy/controller/protocol; revalidate")
    net = OutcomeScorer(ScorerConfig(**ck["model_cfg"])).to(device).eval()
    net.load_state_dict(ck["model"])

    def score_fn(ctx, cands):
        from ..evaluation.probes import _history
        x = observation_inputs(ctx.obs, sample_cfg, text_emb, np.random.default_rng(0),
                               ctx.region.points if sample_cfg.geometry_source else None)
        n = len(cands)
        b = {k: torch.as_tensor(np.repeat(np.asarray(x[k])[None], n, 0)).to(device) for k in ("points", "points_mask", "text", "state")}
        b["points"], b["points_mask"] = b["points"].float(), b["points_mask"].bool()
        b["text"], b["state"] = b["text"].float(), b["state"].float()
        b["history"] = torch.as_tensor(np.repeat(_history(ctx.obs)[None], n, 0)).float().to(device)
        b["candidate"] = torch.as_tensor(np.stack([candidate_vector(c.T_WE, c.gripper, c.q, x["T_FW"]) for c in cands])).float().to(device)
        with torch.no_grad():
            o = net(b)
        pj = torch.sigmoid(o["joint_logit"]).cpu().numpy()
        pa = torch.sigmoid(o["arm_logits"]).cpu().numpy()
        dur = torch.exp(o["log_duration"]).cpu().numpy()
        return [{"p_success": float(pj[i]), "p_left": float(pa[i, 0]), "p_right": float(pa[i, 1]), "duration_s": float(dur[i])}
                for i in range(n)]
    return score_fn
