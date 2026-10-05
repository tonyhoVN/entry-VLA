"""M4 commands: collect-probes (frozen-policy outcome labels) and compare (matched selector comparison)."""

from __future__ import annotations

import os

from ..utils.provenance import make_run_dir, run_manifest, seed_everything, write_json
from .common import add_common, bounded_int, emit, fail, load, module_available


def register(sub) -> None:
    p = sub.add_parser("collect-probes", help="bounded candidate probes with a frozen policy (user-run on simulators)")
    add_common(p, "configs/probes.yaml")
    p.add_argument("--max-scenes", type=bounded_int(1, 100000), default=None)
    p.set_defaults(func=run_collect)

    p = sub.add_parser("compare", help="matched comparison of entry selectors on identical seeds")
    add_common(p, "configs/compare.yaml")
    p.add_argument("--max-trials", type=bounded_int(1, 10000), default=None)
    p.set_defaults(func=run_compare)


def _sample_cfg(cfg):
    from ..data.entry_dataset import EntrySampleConfig
    d = cfg.get("inputs") or {}
    return EntrySampleConfig(**{k: v for k, v in d.items() if k in EntrySampleConfig.__dataclass_fields__})


def run_collect(args) -> int:
    import numpy as np
    from ..evaluation import factory as F
    from ..evaluation.probes import ProbeConfig, collect_scene
    from ..evaluation.trials import append_jsonl
    from ..models.text import make_text_encoder
    from ..sim.registry import backend_class, make_backend
    cfg = load(args)
    pc = ProbeConfig(**cfg["probes"])
    if args.max_scenes:
        pc.n_scenes = args.max_scenes
    caps = backend_class(cfg["backend"]["name"]).capabilities()
    rep = {"backend": caps.name, "n_scenes": pc.n_scenes, "attempts_upper_bound": pc.n_scenes * (1 + pc.perturbations_per_scene
           + pc.shell_per_scene), "policy": cfg["policy"]["type"], "base_selector": cfg["selector"]["type"]}
    if cfg["policy"]["type"] == "mock_oracle" and caps.name != "mock":
        fail("mock_oracle policy only runs on the mock backend")
    if args.dry_run:
        _sample_cfg(cfg); F.make_prep(cfg); F.make_filter(cfg)
        emit({**rep, "dry_run": True})
        return 0
    seed_everything(int(cfg["seed"]))
    run_dir = make_run_dir(cfg.get("runs_root", "runs"), f"probes_{caps.name}")
    write_json(os.path.join(run_dir, "manifest.json"), run_manifest(cfg, {"capabilities": caps.as_dict()}))
    be = make_backend(cfg["backend"])
    scfg = _sample_cfg(cfg)
    try:
        be.make(cfg["backend"]["task"])
        policy, arms = F.make_policy(cfg), F.make_arms(cfg, be)
        base, region = F.make_selector(cfg, int(cfg["seed"])), F.make_region(cfg, int(cfg["seed"]))
        text = make_text_encoder(cfg.get("text_encoder", {}))
        rng = np.random.default_rng(int(cfg["seed"]))
        n = 0
        for i in range(pc.n_scenes):
            seed = pc.seed_start + i
            obs = be.reset(seed=seed)
            emb = text([obs.instruction])[0]
            for rec in collect_scene(be, policy, base, region, arms, seed, pc, F.make_trial_cfg(cfg, caps.name),
                                     F.make_prep(cfg), F.make_filter(cfg), scfg, emb, os.path.join(run_dir, "cache"), rng):
                append_jsonl(os.path.join(run_dir, "probes.jsonl"), rec)
                n += 1
    finally:
        be.close()
    emit({**rep, "run_dir": run_dir, "records": n})
    return 0


def run_compare(args) -> int:
    from ..evaluation import factory as F
    from ..evaluation.compare import compare
    from ..evaluation.trials import append_jsonl, run_trial
    from ..sim.registry import backend_class, make_backend
    cfg = load(args)
    caps = backend_class(cfg["backend"]["name"]).capabilities()
    n = args.max_trials or int(cfg["trials"]["n"])
    names = [s["name"] for s in cfg["selectors"]]
    rep = {"backend": caps.name, "selectors": names, "n_trials_each": n, "reference": cfg["reference"],
           "policy": cfg["policy"]["type"]}
    if cfg["reference"] not in names:
        fail("reference must be one of the selectors")
    if cfg["policy"]["type"] == "mock_oracle" and caps.name != "mock":
        fail("mock_oracle policy only runs on the mock backend")
    if args.dry_run:
        emit({**rep, "dry_run": True, "budgets": cfg.get("budgets", {})})
        return 0
    seed_everything(int(cfg["seed"]))
    run_dir = make_run_dir(cfg["trials"].get("runs_root", "runs"), f"compare_{caps.name}")
    write_json(os.path.join(run_dir, "manifest.json"), run_manifest(cfg, {"capabilities": caps.as_dict()}))
    be = make_backend(cfg["backend"])
    out = {}
    try:
        be.make(cfg["backend"]["task"])
        policy, arms = F.make_policy(cfg), F.make_arms(cfg, be)
        for scfg in cfg["selectors"]:
            sub = {**cfg, "selector": scfg}
            if scfg["type"] == "full_episode":
                from ..evaluation.selectors import FullEpisodeSelector
                sel = FullEpisodeSelector()
            else:
                sel = F.make_selector(sub, int(cfg["seed"]))
            region = F.make_region(cfg, int(cfg["seed"]))
            recs = []
            for i in range(n):
                seed = int(cfg["trials"].get("seed_start", 1000)) + i
                rec = run_trial(be, policy, sel, region, arms, seed, F.make_trial_cfg(cfg, caps.name), F.make_prep(cfg),
                                F.make_filter(cfg), wrist_T_EC=F.wrist_hand_eye(cfg), trial_id=f"{scfg['name']}_{i:05d}")
                append_jsonl(os.path.join(run_dir, f"trials_{scfg['name']}.jsonl"), rec)
                recs.append(rec)
            out[scfg["name"]] = recs
    finally:
        be.close()
    res = compare(out, cfg["reference"], n, cfg["trials"].get("time_cap_s"), cfg.get("budgets", {}))
    write_json(os.path.join(run_dir, "comparison.json"), res)
    emit({**rep, "run_dir": run_dir, "comparison": res})
    return 0
