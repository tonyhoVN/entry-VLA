"""M0 one-task headless reset/step/RGB-D smoke check for a backend.

Bounded by construction: one reset and at most ``--steps`` (<= 5) hold-pose steps.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..sim.contracts import ARMS, BackendError, BimanualTcpCommand
from ..sim.registry import backend_class, make_backend
from ..sim.validate import check_observation, depth_summary, tcp_projection_report
from ..utils.provenance import write_json
from .common import add_common, bounded_int, emit, fail, load, module_available

REQUIRED_MODULES = {"duobench": ["gymnasium", "duobench", "rcs", "mujoco"],
                    "peract2": ["rlbench", "pyrep"], "mock": []}


def register(sub) -> None:
    p = sub.add_parser("smoke", help="M0 backend smoke check (1 reset + <=5 steps)")
    add_common(p, "configs/smoke.yaml")
    p.add_argument("--steps", type=bounded_int(0, 5), default=5)
    p.add_argument("--out", default=None, help="write report JSON (+ first-frame npz) to this directory")
    p.set_defaults(func=run)


def resolve_task(backend_name: str, task: str, known: list) -> str:
    """Registered task id; DuoBench also accepts display names (Carry-Pot)."""
    if backend_name == "duobench":
        from ..sim.duobench import resolve_task_id
        try:
            return resolve_task_id(task)
        except ValueError as e:
            fail(str(e))
    if task not in known:
        fail(f"task {task!r} not in {backend_name} tasks: {known}")
    return task


def run(args) -> int:
    cfg = load(args)
    bcfg = cfg["backend"]
    name = bcfg["name"]
    cls = backend_class(name)
    caps = cls.capabilities()
    task = bcfg.get("task")
    task = resolve_task(name, task, cls(bcfg).list_tasks())
    report = {"backend": name, "task": task, "capabilities": caps.as_dict(),
              "modules": module_available(REQUIRED_MODULES[name])}
    if args.dry_run:
        report["dry_run"] = True
        emit(report)
        return 0
    missing = [m for m, ok in report["modules"].items() if not ok]
    if missing:
        fail(f"{name}: missing modules {missing}. This is a setup blocker; see envs/{name}/README.md")
    backend = make_backend(bcfg)
    try:
        backend.make(task)
        obs = backend.reset(seed=int(cfg["seed"]))
        report["reset_contract_violations"] = check_observation(obs, require_depth=bool(bcfg.get("require_depth", True)))
        report["depth"] = depth_summary(obs)
        report["tcp_projection"] = tcp_projection_report(obs)
        report["instruction"] = obs.instruction
        hold = BimanualTcpCommand(T_WE={a: obs.arms[a].T_WE.copy() for a in ARMS},
                                  gripper={a: 1.0 for a in ARMS})
        start = {a: obs.arms[a].T_WE[:3, 3].copy() for a in ARMS}
        drift = []
        for _ in range(args.steps):
            res = backend.step(hold, obs)
            obs = res.obs
            drift.append({a: float(np.linalg.norm(obs.arms[a].T_WE[:3, 3] - start[a])) for a in ARMS})
            if res.terminated or res.truncated:
                break
        report["hold_steps"] = len(drift)
        report["hold_drift_m"] = drift
        report["step_contract_violations"] = check_observation(obs)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            write_json(os.path.join(args.out, "smoke_report.json"), report)
            arrays = {}
            for n, c in obs.cameras.items():
                if c.rgb is not None:
                    arrays[f"{n}_rgb"] = c.rgb
                if c.depth is not None:
                    arrays[f"{n}_depth"] = c.depth
            np.savez_compressed(os.path.join(args.out, "last_frame.npz"), **arrays)
    except BackendError as e:
        fail(f"{type(e).__name__}: {e}")
    finally:
        backend.close()
    emit(report)
    ok = not report["reset_contract_violations"] and not report["step_contract_violations"]
    return 0 if ok else 1
