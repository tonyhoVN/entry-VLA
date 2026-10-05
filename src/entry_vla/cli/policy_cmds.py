"""M2 commands: annotate, rollout (prep + frozen policy trials), policy-dataset, train-smolvla."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import yaml

from ..data import episode as E
from ..utils.provenance import make_run_dir, run_manifest, seed_everything, write_json
from .common import add_common, bounded_int, emit, fail, load, module_available


def register(sub) -> None:
    p = sub.add_parser("annotate", help="propose paired skill-start intervals (+overrides, overlays)")
    add_common(p, "configs/annotate.yaml")
    p.add_argument("--max-episodes", type=bounded_int(1, 100000), default=None)
    p.add_argument("--overlays", action="store_true", help="write PNG/GIF overlays and timeline plots")
    p.set_defaults(func=run_annotate)

    p = sub.add_parser("rollout", help="bounded entry-selection trials: prepare -> handoff -> frozen policy")
    add_common(p, "configs/rollout.yaml")
    p.add_argument("--max-trials", type=bounded_int(1, 10000), default=None)
    p.set_defaults(func=run_rollout)

    p = sub.add_parser("policy-dataset", help="build a LeRobot v3 dataset (20D state, target.abs) from episodes")
    add_common(p, "configs/train/smolvla_4090.yaml")
    p.set_defaults(func=run_policy_dataset)

    p = sub.add_parser("train-smolvla", help="fine-tune SmolVLA with chunk re-encoding + grad accumulation (user-run)")
    add_common(p, "configs/train/smolvla_4090.yaml")
    p.set_defaults(func=run_train_smolvla)


def _load_overrides(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def run_annotate(args) -> int:
    from ..data.segmentation import AnnotateConfig, apply_overrides, arm_criteria, propose_paired_start
    cfg = load(args)
    a = cfg["annotate"]
    eps = E.list_episodes(a["episodes_root"])[: args.max_episodes]
    acfg = AnnotateConfig(**(a.get("params") or {}))
    rep: Dict[str, Any] = {"episodes_root": a["episodes_root"], "num_episodes": len(eps), "params": acfg.__dict__}
    if args.dry_run:
        emit({**rep, "dry_run": True})
        return 0
    overrides = _load_overrides(a.get("overrides"))
    out = {}
    for d in eps:
        ep = E.load_episode(d, cameras=acfg.visibility_cameras + list(a.get("overlay_cameras", [])) or [])
        prop = apply_overrides(ep.episode_id, propose_paired_start(ep, acfg), overrides)
        ann = dict(ep.annotations)
        ann.setdefault("skill_starts", [])
        ann["skill_starts"] = [s for s in ann["skill_starts"] if s.get("label") != prop["label"]] + [prop]
        E.save_annotations(d, ann)
        out[ep.episode_id] = {"status": prop["status"], "start": prop.get("proposed_start"),
                              "confidence": prop.get("confidence")}
        if args.overlays and prop.get("interval"):
            from ..data.overlays import save_overlays, save_timeline
            odir = a.get("overlay_dir", os.path.join(a["episodes_root"], "_overlays"))
            save_overlays(ep, prop, odir, list(a.get("overlay_cameras", [])) or list(ep.cameras))
            save_timeline(ep, prop, {k: arm_criteria(ep, k, acfg) for k in ("left", "right")},
                          os.path.join(odir, f"{ep.episode_id}_timeline.png"))
    emit({**rep, "annotations": out})
    return 0


def run_rollout(args) -> int:
    from ..evaluation import factory as F
    from ..evaluation.metrics import summarize_trials
    from ..evaluation.trials import append_jsonl, run_trial
    from ..sim.registry import backend_class, make_backend
    cfg = load(args)
    tr = cfg["trials"]
    n = args.max_trials or int(tr["n"])
    caps = backend_class(cfg["backend"]["name"]).capabilities()
    rep = {"backend": caps.name, "task": cfg["backend"].get("task"), "n_trials": n, "selector": cfg["selector"]["type"],
           "policy": cfg["policy"]["type"], "region": cfg["region"]["mode"]}
    if cfg["policy"]["type"] == "mock_oracle" and caps.name != "mock":
        fail("mock_oracle policy only runs on the mock backend")
    if args.dry_run:
        F.make_prep(cfg); F.make_filter(cfg); F.make_trial_cfg(cfg, caps.name)
        emit({**rep, "dry_run": True, "modules": module_available({"duobench": ["duobench", "rcs"],
                                                                    "peract2": ["rlbench", "pyrep"]}.get(caps.name, []))})
        return 0
    seed_everything(int(cfg["seed"]))
    run_dir = make_run_dir(tr.get("runs_root", "runs"), f"rollout_{caps.name}_{rep['selector']}")
    write_json(os.path.join(run_dir, "manifest.json"), run_manifest(cfg, {"capabilities": caps.as_dict()}))
    be = make_backend(cfg["backend"])
    try:
        be.make(cfg["backend"]["task"])
        arms = F.make_arms(cfg, be)
        policy = F.make_policy(cfg)
        selector = F.make_selector(cfg, int(cfg["seed"]))
        region = F.make_region(cfg, int(cfg["seed"]))
        recs = []
        for i in range(n):
            seed = int(tr.get("seed_start", 1000)) + i
            rec = run_trial(be, policy, selector, region, arms, seed, F.make_trial_cfg(cfg, caps.name), F.make_prep(cfg),
                            F.make_filter(cfg), wrist_T_EC=F.wrist_hand_eye(cfg), trial_id=f"trial{i:05d}")
            append_jsonl(os.path.join(run_dir, "trials.jsonl"), rec)
            recs.append(rec)
    finally:
        be.close()
    summary = summarize_trials(recs, requested=n, time_cap_s=tr.get("time_cap_s"))
    write_json(os.path.join(run_dir, "summary.json"), summary)
    emit({**rep, "run_dir": run_dir, "summary": summary})
    return 0


def run_policy_dataset(args) -> int:
    from ..policies.smolvla_adapter import lerobot_features
    cfg = load(args)
    d, p = cfg["dataset"], cfg["policy"]
    eps = E.list_episodes(d["episodes_root"])
    rep = {"episodes_root": d["episodes_root"], "num_episodes": len(eps), "root": d["root"], "repo_id": d["repo_id"],
           "features": {k: v["shape"] for k, v in lerobot_features(p["cameras"], p["image_hw"]).items()},
           "segment_by_annotation": d.get("segment_label")}
    if args.dry_run:
        emit({**rep, "dry_run": True, "modules": module_available(["lerobot"])})
        return 0
    if os.path.exists(d["root"]):
        fail(f"{d['root']} exists; datasets are immutable (choose a new root)")
    splits = json.load(open(d["splits"])) if d.get("splits") else None
    episodes, segments = [], []
    for path in eps:
        ep = E.load_episode(path, cameras=list(p["cameras"]))
        if splits and splits["assignment"].get(ep.episode_id) != "train":
            continue
        seg = None
        if d.get("segment_label"):
            s = [x for x in ep.annotations.get("skill_starts", []) if x.get("label") == d["segment_label"]
                 and x.get("status") in ("proposed", "manual")]
            if not s:
                continue
            seg = (int(s[0]["proposed_start"]), None)   # local skill segment from the annotated start
        episodes.append(ep); segments.append(seg)
    from ..policies.smolvla_adapter import write_lerobot_dataset
    res = write_lerobot_dataset(episodes, segments, d["repo_id"], d["root"], float(d["fps"]), p["cameras"],
                                p["image_hw"], robot_type=d.get("robot_type", "bimanual_franka"))
    emit({**rep, **res})
    return 0


def run_train_smolvla(args) -> int:
    from ..policies.train_smolvla import train, validate_train_config
    cfg = load(args)
    v = validate_train_config(cfg)
    if args.dry_run:
        emit({**v, "dry_run": True, "modules": module_available(["lerobot", "torch"])})
        return 0 if not v["problems"] else 1
    if v["problems"]:
        fail(f"config problems: {v['problems']}")
    seed_everything(int(cfg["seed"]))
    run_dir = make_run_dir(cfg["train"].get("runs_root", "runs"), "smolvla")
    write_json(os.path.join(run_dir, "manifest.json"), run_manifest(cfg))
    emit(train(cfg, run_dir))
    return 0
