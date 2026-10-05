"""M1 data commands: inspect, export, replay-render, split, check-alignment, make-fixtures."""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from ..data import episode as E
from .common import add_common, bounded_int, emit, fail, load, module_available

FORMATS = ("duobench_raw", "duobench_lerobot", "peract2_dir", "mock")
FORMAT_MODULES = {"duobench_raw": ["pyarrow", "PIL"], "duobench_lerobot": ["pyarrow"], "peract2_dir": ["PIL"],
                  "mock": []}


def register(sub) -> None:
    p = sub.add_parser("inspect", help="inspect ONE source episode (<=10 frames) and report schema/alignment")
    add_common(p, "configs/export.yaml")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frames", type=bounded_int(1, 10), default=10)
    p.add_argument("--handeye", action="store_true", help="DuoBench: wrist hand-eye consistency under export.frames")
    p.set_defaults(func=run_inspect)

    p = sub.add_parser("export", help="convert source demonstrations to entry_vla.episode/1 (user-run for full data)")
    add_common(p, "configs/export.yaml")
    p.add_argument("--max-episodes", type=bounded_int(1, 100000), default=None)
    p.add_argument("--max-frames", type=bounded_int(1, 100000), default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=run_export)

    p = sub.add_parser("replay-render", help="DuoBench: re-render RGB-D from recorded sim_state (user-run, simulator)")
    add_common(p, "configs/export.yaml")
    p.add_argument("--max-episodes", type=bounded_int(1, 100000), default=1)
    p.add_argument("--max-frames", type=bounded_int(1, 100000), default=None)
    p.set_defaults(func=run_replay_render)

    p = sub.add_parser("split", help="group-aware train/val/test(/heldout) split over exported episodes")
    add_common(p, "configs/split.yaml")
    p.set_defaults(func=run_split)

    p = sub.add_parser("check-alignment", help="obs/action alignment diagnostics (offline; --live = 1 reset + 2 steps)")
    add_common(p, "configs/export.yaml")
    p.add_argument("--episode-dir", default=None, help="exported episode to analyse offline")
    p.add_argument("--live", action="store_true", help="bounded one-step diagnostic in the configured backend")
    p.set_defaults(func=run_check_alignment)

    p = sub.add_parser("duobench-frames", help="print T_world_shared / T_shared_armbase from a DuoBench task config")
    add_common(p, "configs/export.yaml")
    p.add_argument("--allow-asset-import", action="store_true",
                   help="required: importing duobench may fetch assets into ~/.duobench / ~/.rcs")
    p.set_defaults(func=run_duobench_frames)

    p = sub.add_parser("make-fixtures", help="write synthetic MOCK episodes (labelled synthetic) for pipeline checks")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=bounded_int(1, 50), default=4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=run_make_fixtures)


def _xcfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    x = cfg.get("export")
    if not x or x.get("format") not in FORMATS:
        fail(f"config needs export.format in {FORMATS}")
    return x


def _preflight(x: Dict[str, Any]) -> Dict[str, Any]:
    rep = {"format": x["format"], "source": x.get("source"), "modules": module_available(FORMAT_MODULES[x["format"]])}
    if x["format"] != "mock":
        rep["source_exists"] = bool(x.get("source")) and os.path.exists(x["source"])
    return rep


def run_inspect(args) -> int:
    cfg = load(args)
    x = _xcfg(cfg)
    rep = _preflight(x)
    if args.dry_run:
        emit({**rep, "dry_run": True})
        return 0
    if x["format"] != "mock" and not rep["source_exists"]:
        fail(f"source {x.get('source')!r} not found. Download selectively (see envs/*/README.md); nothing was fetched")
    fmt = x["format"]
    if fmt == "duobench_raw":
        from ..data import duobench_raw as D
        rep["summary"] = D.parquet_summary(x["source"])
        rows = D.read_episode_rows(x["source"], args.episode, max_frames=args.frames)
        rep["uuid"] = rows[0]["uuid"]
        rep["alignment"] = D.alignment_report(rows)
        rep["instruction"] = rows[0].get("instruction")
        if args.handeye:
            T_ws = (x.get("frames") or {}).get("T_world_shared")
            rep["handeye"] = {a: D.handeye_consistency(rows, np.eye(4) if T_ws is None else np.asarray(T_ws), a)
                              for a in ("left", "right")}
            rep["handeye_assumed_T_world_shared"] = "identity" if T_ws is None else "config"
    elif fmt == "duobench_lerobot":
        from ..data import lerobot_v3 as L
        rep["summary"] = L.inspect(x["source"])
        fr = L.read_episode_frames(x["source"], args.episode, max_frames=args.frames)
        rep["first_state"] = np.asarray(fr["observation.state"].tolist())[0].tolist()
        rep["timestamps"] = np.asarray(fr["timestamp"]).tolist()
    elif fmt == "peract2_dir":
        from ..data import peract2_raw as P
        eps = P.list_episode_dirs(x["source"])
        if not eps:
            fail(f"no episode dirs under {x['source']}/all_variations/episodes")
        rep["num_episode_dirs"] = len(eps)
        rep["episode"] = P.inspect_episode_dir(eps[args.episode], max_frames=args.frames)
    else:
        from ..data.synthetic import record_mock_episode
        ep = record_mock_episode(int(cfg["seed"]))
        rep["episode"] = {"frames": ep.num_frames, "problems": E.validate_episode(ep)}
    emit(rep)
    return 0


def _export_iter(x: Dict[str, Any], max_episodes, max_frames):
    fmt = x["format"]
    task = x["task_id"]
    n = max_episodes or int(x.get("max_episodes", 1))
    off = int(x.get("episode_offset", 0))
    common = {"image_hw": x.get("image_hw"), "cameras": x.get("cameras"), **(x.get("frames") or {}),
              "gripper_stroke_m": x.get("gripper_stroke_m", 0.085)}
    common = {k: v for k, v in common.items() if v is not None}
    if fmt == "duobench_raw":
        from ..data import duobench_raw as D
        for i in range(off, off + n):
            try:
                rows = D.read_episode_rows(x["source"], i, max_frames=max_frames)
            except IndexError:
                return
            yield D.export_rows(rows, task, x["source"], i, common)
    elif fmt == "duobench_lerobot":
        from ..data import lerobot_v3 as L
        for i in range(off, off + n):
            yield L.export_episode(x["source"], i, task, max_frames=max_frames,
                                   decode_videos=bool(x.get("decode_videos", False)))
    elif fmt == "peract2_dir":
        from ..data import peract2_raw as P
        for ep_dir in P.list_episode_dirs(x["source"])[off:off + n]:
            yield P.export_episode_dir(ep_dir, task, common, max_frames=max_frames)
    else:
        from ..data.synthetic import record_mock_episode
        for i in range(off, off + n):
            yield record_mock_episode(seed=i)


def run_export(args) -> int:
    cfg = load(args)
    x = _xcfg(cfg)
    rep = _preflight(x)
    rep["out_root"] = x["out_root"]
    if args.dry_run:
        emit({**rep, "dry_run": True})
        return 0
    if x["format"] != "mock" and not rep["source_exists"]:
        fail(f"source {x.get('source')!r} not found")
    written = []
    for ep in _export_iter(x, args.max_episodes, args.max_frames):
        problems = E.validate_episode(ep)
        if problems:
            fail(f"{ep.episode_id}: {problems}")
        written.append(E.save_episode(ep, x["out_root"], overwrite=args.overwrite))
    emit({**rep, "written": written})
    return 0


def run_replay_render(args) -> int:
    cfg = load(args)
    x = _xcfg(cfg)
    if x["format"] != "duobench_raw":
        fail("replay-render is implemented for duobench_raw only (PerAct2 saves depth already)")
    rep = {**_preflight(x), "modules_sim": module_available(["duobench", "rcs", "gymnasium", "mujoco"]),
           "out_root": x.get("rgbd_out_root", x["out_root"] + "_rgbd")}
    if args.dry_run:
        emit({**rep, "dry_run": True})
        return 0
    if not all(rep["modules_sim"].values()):
        fail(f"simulator modules missing {rep['modules_sim']} (setup blocker; envs/duobench/README.md)")
    from ..data.duobench_replay import replay_render_episode
    out = []
    for i in range(int(x.get("episode_offset", 0)), int(x.get("episode_offset", 0)) + args.max_episodes):
        res = replay_render_episode(x["source"], i, x["task_id"], cfg["backend"], {"image_hw": x.get("image_hw")},
                                    args.max_frames)
        if not res["report"]["within_tolerance"]:
            out.append({"episode_index": i, "rejected": True, "report": res["report"]})
            continue
        out.append({"episode_index": i, "dir": E.save_episode(res["episode"], rep["out_root"]), "report": res["report"]})
    emit({**rep, "episodes": out})
    return 0


def run_split(args) -> int:
    from ..data import splits as S
    cfg = load(args)
    s = cfg["split"]
    eps = E.list_episodes(s["episodes_root"])
    rep = {"episodes_root": s["episodes_root"], "num_episodes": len(eps), "by": s.get("by", "source_demo_id")}
    if args.dry_run:
        emit({**rep, "dry_run": True})
        return 0
    if not eps:
        fail(f"no exported episodes under {s['episodes_root']}")
    metas = []
    for d in eps:
        with open(os.path.join(d, "meta.json")) as f:
            import json
            metas.append(json.load(f))
    res = S.assign_splits(metas, tuple(s.get("fractions", (0.8, 0.1, 0.1))), by=rep["by"],
                          salt=str(s.get("salt", "entry_vla")), heldout=s.get("heldout"))
    S.save_splits(res, s["out"])
    counts = {}
    for v in res["assignment"].values():
        counts[v] = counts.get(v, 0) + 1
    emit({**rep, "out": s["out"], "counts": counts})
    return 0


def run_check_alignment(args) -> int:
    from ..data import alignment as A
    cfg = load(args)
    rep: Dict[str, Any] = {}
    if args.dry_run:
        emit({"episode_dir": args.episode_dir, "live": args.live, "backend": cfg.get("backend", {}).get("name"),
              "dry_run": True})
        return 0
    if args.episode_dir:
        ep = E.load_episode(args.episode_dir, cameras=[])
        rep["declared"] = ep.meta.get("alignment")
        for a in ("left", "right"):
            if f"{a}.cmd_T_WE" in ep.lowdim and f"{a}.T_WE" in ep.lowdim:
                rep[f"{a}_target_lag"] = A.estimate_target_lag(ep.lowdim[f"{a}.T_WE"][:, :3, 3],
                                                               ep.lowdim[f"{a}.cmd_T_WE"][:, :3, 3])
        t = ep.lowdim["time"]
        if len(t) > 1:
            dt = np.diff(t)
            rep["frame_spacing_s"] = {"mean": float(dt.mean()), "min": float(dt.min()), "max": float(dt.max())}
    if args.live:
        from ..sim.registry import make_backend
        be = make_backend(cfg["backend"])
        try:
            be.make(cfg["backend"]["task"])
            rep["live"] = A.live_one_step_diagnostic(be, seed=int(cfg["seed"]))
        finally:
            be.close()
    if not rep:
        fail("pass --episode-dir and/or --live")
    emit(rep)
    return 0


def run_make_fixtures(args) -> int:
    from ..data.synthetic import record_mock_episode
    out = []
    for i in range(args.n):
        ep = record_mock_episode(seed=args.seed + i, split="unassigned")
        out.append(E.save_episode(ep, args.out))
    emit({"written": out, "provenance": "synthetic_mock"})
    return 0


def run_duobench_frames(args) -> int:
    cfg = load(args)
    task = cfg.get("export", {}).get("task_id") or cfg["backend"]["task"]
    if args.dry_run or not args.allow_asset_import:
        emit({"task": task, "modules": module_available(["duobench", "rcs"]), "dry_run": True,
              "note": "pass --allow-asset-import to import duobench (may download assets)"})
        return 0
    from ..sim.duobench import DuoBenchBackend, _pose_matrix
    be = DuoBenchBackend(cfg["backend"])
    t, c = be.build_cfg(task)
    T_ws = _pose_matrix(c.root_frame_to_world) @ _pose_matrix(c.shared_base_frame_to_root_frame)
    armbase = {k: _pose_matrix(v).tolist() for k, v in (c.robot_to_shared_base_frame or {}).items()}
    emit({"task": t, "T_world_shared": T_ws.tolist(), "T_shared_armbase": armbase,
          "note": "paste into export.frames; robot keys must map to left/right; confirm with inspect --handeye"})
    return 0
