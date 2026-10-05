"""Bounded SmolVLA fine-tuning loop (user-run; never executed by the agent).

Adds what ``lerobot-train`` 0.6.1 lacks for this project: gradient accumulation,
on-the-fly sequential chunk encoding from ``target.abs`` and entry_vla
train-only action normalisation. Uses lerobot's dataset, policy factory and
pre-processor (tokenisation, image handling, state normalisation).

UNVERIFIED call details (read in v0.6.1 source but never executed; check on first
user run): ``LeRobotDatasetMetadata`` import path, ``PreTrainedConfig.from_pretrained``,
``lerobot.configs.types.NormalizationMode``, and the ``dataset_stats`` kwarg of
``make_pre_post_processors``. A failure there is a pin/API mismatch to fix here.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import numpy as np

from .smolvla_adapter import ACTION_DIM, compute_action_stats, encode_chunks_from_targets

HARD_MAX_STEPS = 200_000


def validate_train_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Schema/consistency checks only (no imports of lerobot/torch, no data iteration)."""
    p, d, t = cfg["policy"], cfg["dataset"], cfg["train"]
    problems = []
    if not 1 <= int(t["steps"]) <= HARD_MAX_STEPS:
        problems.append(f"train.steps must be in [1, {HARD_MAX_STEPS}]")
    if int(p["n_action_steps"]) > int(p["chunk_size"]):
        problems.append("n_action_steps must be <= chunk_size (SmolVLAConfig constraint)")
    if t["precision"] not in ("fp32", "bf16", "fp16"):
        problems.append("train.precision must be fp32|bf16|fp16")
    if int(t["batch_size"]) < 1 or int(t["grad_accum"]) < 1:
        problems.append("batch_size and grad_accum must be >= 1")
    if not str(t["device"]).startswith(("cuda", "cpu")):
        problems.append("train.device must be cuda:N or cpu")
    if len(p["cameras"]) < 1 or len(p["cameras"]) > 3 + int(p.get("empty_cameras", 0)):
        problems.append("SmolVLA base expects up to 3 cameras (plus empty_cameras slots)")
    info_path = os.path.join(d["root"], "meta", "info.json")
    ds = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        feats = info.get("features", {})
        for k, dim in (("observation.state", 20), ("action", ACTION_DIM), ("target.abs", 20)):
            shp = feats.get(k, {}).get("shape")
            if shp is None or int(shp[0]) != dim:
                problems.append(f"dataset feature {k} shape {shp} != ({dim},)")
        for cam_key in p["cameras"].values():
            if f"observation.images.{cam_key}" not in feats:
                problems.append(f"dataset lacks observation.images.{cam_key}")
        ds = {"fps": info.get("fps"), "total_episodes": info.get("total_episodes"), "codebase_version": info.get("codebase_version")}
    else:
        ds = {"missing": info_path}
    eff = int(t["batch_size"]) * int(t["grad_accum"])
    return {"problems": problems, "dataset": ds, "effective_batch": eff,
            "note": "memory fit on the RTX 4090 is NOT verified; profile only when requested"}


def train(cfg: Dict[str, Any], run_dir: str) -> Dict[str, Any]:  # pragma: no cover - user-run
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors

    p, d, t = cfg["policy"], cfg["dataset"], cfg["train"]
    torch.manual_seed(int(cfg.get("seed", 0)))
    meta = LeRobotDatasetMetadata(d["repo_id"], root=d["root"])
    H = int(p["chunk_size"])
    dts = {"target.abs": [k / meta.fps for k in range(H)], "target.frame": [k / meta.fps for k in range(H)]}
    train_eps = d.get("train_episodes")
    dataset = LeRobotDataset(d["repo_id"], root=d["root"], episodes=train_eps, delta_timestamps=dts)

    pcfg = PreTrainedConfig.from_pretrained(p["checkpoint"])
    pcfg.pretrained_path = p["checkpoint"]
    for k in ("chunk_size", "n_action_steps", "freeze_vision_encoder", "train_expert_only", "train_state_proj",
              "empty_cameras"):
        if k in p:
            setattr(pcfg, k, p[k])
    pcfg.device = t["device"]
    pcfg.use_amp = t["precision"] != "fp32"
    from lerobot.configs.types import NormalizationMode
    pcfg.normalization_mapping = dict(pcfg.normalization_mapping)
    pcfg.normalization_mapping["ACTION"] = NormalizationMode.IDENTITY   # entry_vla normalises chunks itself
    rename = {f"observation.images.{k}": f"observation.images.{k}" for k in p["cameras"].values()}
    policy = make_policy(pcfg, ds_meta=meta, rename_map=rename)
    pre, _ = make_pre_post_processors(policy_cfg=pcfg, pretrained_path=p["checkpoint"], dataset_stats=meta.stats)
    if t.get("activation_checkpointing"):
        vlm = getattr(getattr(getattr(policy, "model", None), "vlm_with_expert", None), "vlm", None)
        if vlm is None or not hasattr(vlm, "gradient_checkpointing_enable"):
            raise RuntimeError("activation checkpointing requested but the VLM module path was not found in this lerobot build")
        vlm.gradient_checkpointing_enable()

    # Train-only action statistics over re-encoded chunks (bounded sample).
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(t["batch_size"]), shuffle=True,
                                         num_workers=int(t.get("num_workers", 4)), drop_last=True)
    chunks, pads = [], []
    for i, b in enumerate(loader):
        c, m = encode_chunks_from_targets(b["observation.state"].numpy(), b["target.abs"].numpy(), b["target.frame"].numpy())
        chunks.append(c); pads.append(m)
        if i + 1 >= int(t.get("stats_batches", 50)):
            break
    stats = compute_action_stats(np.concatenate(chunks), np.concatenate(pads), "train")
    mean, std = torch.tensor(stats["mean"]).float(), torch.tensor(stats["std"]).float()
    with open(os.path.join(run_dir, "action_stats.json"), "w") as f:
        json.dump(stats, f)

    params = [q for q in policy.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(t["lr"]), betas=(0.9, 0.95), weight_decay=float(t.get("weight_decay", 1e-10)))
    warm = int(t.get("warmup_steps", 1000))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(warm, 1)))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[t["precision"]]
    scaler = torch.cuda.amp.GradScaler(enabled=t["precision"] == "fp16")
    accum = int(t["grad_accum"])
    step, it = 0, iter(loader)
    log = open(os.path.join(run_dir, "train_log.jsonl"), "a")
    policy.train()
    t0 = time.time()
    while step < int(t["steps"]):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            raw_state = b["observation.state"].numpy()
            c, pad = encode_chunks_from_targets(raw_state, b["target.abs"].numpy(), b["target.frame"].numpy())
            b = pre(b)
            b["action"] = ((torch.from_numpy(c) - mean) / std).to(t["device"])
            b["action_is_pad"] = torch.from_numpy(pad).to(t["device"])
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=t["precision"] != "fp32" and "cuda" in t["device"]):
                loss, _ = policy.forward(b)
            scaler.scale(loss / accum).backward()
            tot += float(loss.detach())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, float(t.get("grad_clip", 10.0)))
        scaler.step(opt); scaler.update(); sched.step()
        step += 1
        if step % int(t.get("log_every", 50)) == 0:
            log.write(json.dumps({"step": step, "loss": tot / accum, "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}) + "\n")
            log.flush()
        if step % int(t.get("save_every", 2000)) == 0 or step == int(t["steps"]):
            policy.save_pretrained(os.path.join(run_dir, f"checkpoint_{step:07d}"))
    log.close()
    return {"steps": step, "run_dir": run_dir}
