"""Trial runner: reset -> pre-travel decision -> physical preparation -> arrival
verification -> frozen local policy -> labelled outcome.

Each trial yields four distinct records (pre_travel, decision, arrival,
outcome) plus a label version that pins policy checkpoint, controller, reset
protocol and timeout. Labels must be regenerated if any of those change.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..geometry.shells import FilterConfig
from ..policies.base import CommandArbiter, LocalPolicy, run_policy
from ..sim.contracts import ARMS, Observation
from ..sim.preparation import PLANNER_ID, JointSpacePlanner, PrepConfig, execute_preparation
from .failures import classify
from .selectors import PreTravelContext, RegionEstimator, Selector


@dataclass
class TrialConfig:
    max_policy_steps: int = 100
    n_action_steps: int = 4
    pause_steps: int = 0
    reset_protocol: str = "reset(seed)"
    controller: str = ""
    history_reset: str = "policy.reset() at handoff"
    success_key: str = "success"            # info/oracle key used for LABELLING only
    min_clearance_m: Optional[float] = None  # early-stop if arm/arm clearance drops below (labels collision)


def label_version(policy: LocalPolicy, cfg: TrialConfig, backend_name: str, planner_id: str) -> Dict[str, Any]:
    return {"policy": policy.describe(), "controller": cfg.controller, "backend": backend_name,
            "reset_protocol": cfg.reset_protocol, "timeout_steps": cfg.max_policy_steps,
            "n_action_steps": cfg.n_action_steps, "history_reset": cfg.history_reset, "planner": planner_id,
            "pause_steps": cfg.pause_steps}


def _success_fn(key: str) -> Callable[[Observation, Dict[str, Any]], bool]:
    def f(obs: Observation, info: Dict[str, Any]) -> bool:
        v = info.get(key, obs.oracle.get(key, False))
        return bool(v) and float(v) > 0
    return f


def run_trial(backend, policy: LocalPolicy, selector: Selector, region: RegionEstimator, arms, seed: int,
              trial_cfg: TrialConfig, prep_cfg: PrepConfig, filter_cfg: FilterConfig,
              wrist_T_EC: Optional[Dict[str, np.ndarray]] = None,
              arrival_verifier: Optional[Callable[[Observation], Dict[str, Any]]] = None,
              trial_id: Optional[str] = None) -> Dict[str, Any]:
    t_wall = time.perf_counter()
    trial_id = trial_id or f"{selector.name}_seed{seed}"
    obs = backend.reset(seed=seed)
    rec: Dict[str, Any] = {"trial_id": trial_id, "seed": seed, "selector": selector.name,
                           "label_version": label_version(policy, trial_cfg, backend.name,
                                                          PLANNER_ID)}
    reg = region(obs)
    rec["pre_travel"] = {"t_s": obs.time, "q": {a: obs.arms[a].q.tolist() for a in ARMS},
                         "T_WE": {a: obs.arms[a].T_WE.tolist() for a in ARMS},
                         "region_source": reg.source, "region_points": {a: reg.points[a].tolist() for a in ARMS}}
    ctx = PreTravelContext(obs=obs, region=reg, arms=arms, filter_cfg=filter_cfg, wrist_T_EC=wrist_T_EC)
    t_dec = time.perf_counter()
    dec = selector.select(ctx)
    chosen = dec.pop("chosen")
    rec["decision"] = {**dec, "decision_time_s": time.perf_counter() - t_dec,
                       "chosen": chosen.as_record() if chosen else None}
    if chosen is None:
        rec["outcome"] = {"label": classify(None, None, False, rejected="rejected_pre_travel"), "success": False}
        rec["wall_time_s"] = time.perf_counter() - t_wall
        return rec
    arbiter = CommandArbiter()
    planner = JointSpacePlanner(arms, prep_cfg)
    ho = execute_preparation(backend, obs, chosen.T_WE, planner, arbiter, prep_cfg, gripper_goal=chosen.gripper,
                             pause_steps=trial_cfg.pause_steps)
    rec["arrival"] = ho.as_dict()
    if ho.failure:
        rec["outcome"] = {"label": classify(ho.failure, None, False), "success": False,
                          "prep_sim_time_s": ho.prep_sim_time_s}
        rec["wall_time_s"] = time.perf_counter() - t_wall
        return rec
    obs = ho.obs
    if arrival_verifier is not None:
        v = arrival_verifier(obs)          # uses ACTUAL endpoint observations
        rec["arrival"]["verifier"] = v
        if not v.get("accept", True):
            rec["outcome"] = {"label": "rejected_at_arrival", "success": False, "prep_sim_time_s": ho.prep_sim_time_s}
            rec["wall_time_s"] = time.perf_counter() - t_wall
            return rec
    invalid_fn = None
    if trial_cfg.min_clearance_m is not None:
        from ..geometry.franka import dual_arm_clearance

        def invalid_fn(o: Observation) -> Optional[str]:
            c = dual_arm_clearance(arms["left"], o.arms["left"].q, arms["right"], o.arms["right"].q)
            return f"collision:clearance={c:.3f}" if c < trial_cfg.min_clearance_m else None
    ro = run_policy(backend, policy, obs, arbiter, trial_cfg.max_policy_steps, trial_cfg.n_action_steps,
                    _success_fn(trial_cfg.success_key), invalid_fn)
    rec["outcome"] = {"label": classify(None, ro.terminated_by, ro.success), "success": ro.success,
                      "terminated_by": ro.terminated_by, "policy_steps": ro.steps, "skill_sim_time_s": ro.sim_time_s,
                      "prep_sim_time_s": ho.prep_sim_time_s, "policy_queries": ro.policy_queries,
                      "mean_query_latency_s": ro.mean_query_latency_s}
    rec["wall_time_s"] = time.perf_counter() - t_wall
    return rec


def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=_default) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)
