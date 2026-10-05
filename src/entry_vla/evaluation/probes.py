"""Frozen-policy candidate probe collection (M4; user-run on simulators).

For each scene seed: reset, cache pre-travel inputs, generate bounded candidate
perturbations (+ optional shell samples), filter them geometrically (no
deliberately unsafe attempts), then for every valid candidate: reset the same
seed, verify the reset reproduces the scene, physically prepare, hand off to
the frozen policy and label the outcome. Pre-travel context, candidate,
actual arrival and final outcome are stored as distinct records; arrival
inputs (including post-arrival wrist views) are cached separately.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..data.entry_dataset import EntrySampleConfig, observation_inputs
from ..geometry.rotations import so3_exp
from ..geometry.shells import FilterConfig
from ..policies.base import CommandArbiter, LocalPolicy, run_policy
from ..sim.contracts import ARMS, Observation
from ..sim.preparation import PLANNER_ID, JointSpacePlanner, PrepConfig, execute_preparation
from .failures import classify
from .selectors import DistanceShellSelector, EntryCandidate, PreTravelContext, RegionEstimator, Selector, validate_candidates
from .trials import TrialConfig, _success_fn, label_version


@dataclass
class ProbeConfig:
    n_scenes: int = 10
    seed_start: int = 5000
    perturbations_per_scene: int = 8
    max_pos_m: float = 0.08            # bounded translation perturbation (per arm)
    max_rot_rad: float = 0.35          # bounded rotation perturbation (per arm, TCP frame)
    shell_per_scene: int = 0
    shell_radius: float = 0.20
    reset_check_tol_m: float = 1e-3    # region points must match across resets of the same seed
    arm_success_keys: Optional[Dict[str, str]] = None   # oracle keys for per-arm labels (factorized scorer)


def perturb(c: EntryCandidate, rng: np.random.Generator, max_pos: float, max_rot: float) -> EntryCandidate:
    T = {}
    for a in ARMS:
        d = rng.normal(size=3); d *= rng.uniform(0, max_pos) / max(np.linalg.norm(d), 1e-9)
        w = rng.normal(size=3); w *= rng.uniform(0, max_rot) / max(np.linalg.norm(w), 1e-9)
        M = c.T_WE[a].copy()
        M[:3, :3] = M[:3, :3] @ so3_exp(w)
        M[:3, 3] = M[:3, 3] + d
        T[a] = M
    return EntryCandidate(T, dict(c.gripper), dict(c.roles), source=f"perturb({c.source})")


def _node(d: Dict[str, Any], key: str):
    for p in key.split("."):
        d = d.get(p) if isinstance(d, dict) else None
    return d


def collect_scene(backend, policy: LocalPolicy, base: Selector, region: RegionEstimator, arms, seed: int,
                  pcfg: ProbeConfig, tcfg: TrialConfig, prep: PrepConfig, fcfg: FilterConfig, scfg: EntrySampleConfig,
                  text_emb: np.ndarray, cache_dir: str, rng: np.random.Generator) -> List[Dict[str, Any]]:
    os.makedirs(cache_dir, exist_ok=True)
    obs0 = backend.reset(seed=seed)
    reg0 = region(obs0)
    lv = label_version(policy, tcfg, backend.name, PLANNER_ID)
    pre = observation_inputs(obs0, scfg, text_emb, np.random.default_rng(seed),
                             reg0.points if scfg.geometry_source else None)
    pre_path = os.path.join(cache_dir, f"scene{seed}_pretravel.npz")
    np.savez_compressed(pre_path, **pre, history=_history(obs0))
    ctx = PreTravelContext(obs=obs0, region=reg0, arms=arms, filter_cfg=fcfg)
    seeds = base.propose(ctx)
    if not seeds:
        return [{"scene_seed": seed, "label_version": lv, "outcome": {"label": "rejected_pre_travel", "success": False}}]
    cands = [seeds[0]] + [perturb(seeds[0], rng, pcfg.max_pos_m, pcfg.max_rot_rad) for _ in range(pcfg.perturbations_per_scene)]
    if pcfg.shell_per_scene:
        cands += DistanceShellSelector(n=pcfg.shell_per_scene, radius=pcfg.shell_radius,
                                       seed=int(rng.integers(1 << 31))).propose(ctx)
    cands = validate_candidates(ctx, cands)
    recs = []
    for j, c in enumerate(cands):
        rec: Dict[str, Any] = {"probe_id": f"scene{seed}_c{j}", "scene_seed": seed, "group": f"scene{seed}",
                               "label_version": lv, "pre_travel": {"cache": pre_path, "region_source": reg0.source,
                                                                   "region_points": {a: reg0.points[a].tolist() for a in ARMS}},
                               "candidate": c.as_record(), "candidate_q": None if c.q is None else
                               {a: (None if c.q[a] is None else c.q[a].tolist()) for a in ARMS}}
        if not c.valid:
            rec["outcome"] = {"label": "rejected_pre_travel", "success": False, "not_attempted": True}
            recs.append(rec)
            continue
        obs = backend.reset(seed=seed)
        reg = region(obs)
        if max(np.linalg.norm(reg.points[a] - reg0.points[a]) for a in ARMS) > pcfg.reset_check_tol_m:
            rec["outcome"] = {"label": "reset_rejected", "success": False}
            recs.append(rec)
            continue
        arb = CommandArbiter()
        ho = execute_preparation(backend, obs, c.T_WE, JointSpacePlanner(arms, prep), arb, prep, gripper_goal=c.gripper,
                                 pause_steps=tcfg.pause_steps)
        rec["arrival"] = ho.as_dict()
        if ho.failure:
            rec["outcome"] = {"label": classify(ho.failure, None, False), "success": False,
                              "prep_sim_time_s": ho.prep_sim_time_s}
            recs.append(rec)
            continue
        arr = observation_inputs(ho.obs, EntrySampleConfig(**{**scfg.__dict__, "allow_moving_cameras": True,
                                                              "cameras": []}), text_emb, np.random.default_rng(seed + j),
                                 reg0.points if scfg.geometry_source else None)
        arr_path = os.path.join(cache_dir, f"{rec['probe_id']}_arrival.npz")
        np.savez_compressed(arr_path, **arr, history=_history(ho.obs))
        rec["arrival"]["cache"] = arr_path
        ro = run_policy(backend, policy, ho.obs, arb, tcfg.max_policy_steps, tcfg.n_action_steps, _success_fn(tcfg.success_key))
        arm_s = {a: None for a in ARMS}
        if pcfg.arm_success_keys:
            for a in ARMS:
                v = _node(ro.final_obs.oracle, pcfg.arm_success_keys[a])
                arm_s[a] = None if v is None else bool(v)
        rec["outcome"] = {"label": classify(None, ro.terminated_by, ro.success), "success": ro.success,
                          "arm_success": arm_s, "skill_sim_time_s": ro.sim_time_s, "prep_sim_time_s": ho.prep_sim_time_s,
                          "policy_queries": ro.policy_queries, "policy_steps": ro.steps}
        recs.append(rec)
    return recs


def _history(obs: Observation) -> np.ndarray:
    """Contact/history features available at that time: measured widths + closed flags."""
    w = [obs.arms[a].gripper_width or 0.0 for a in ARMS]
    return np.array(w + [float(x < 0.02) for x in w], np.float32)
