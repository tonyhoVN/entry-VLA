"""Build arms / policies / selectors / region estimators from config sections."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np

from ..geometry.franka import FRANKA_HAND_TCP, PandaArm
from ..geometry.shells import FilterConfig
from ..sim.contracts import ARMS
from ..sim.preparation import PrepConfig
from .selectors import (DistanceShellSelector, FixedCanonicalSelector, ModelSelector, NearestDemoSelector,
                        RegionEstimator, ScoredSelector, Selector, VisibilityGeometricSelector)
from .trials import TrialConfig


def _T(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(4, 4)


def make_arms(cfg: Dict[str, Any], backend) -> Dict[str, PandaArm]:
    """``robot.arms: from_backend`` (mock) or explicit ``{left: {T_WB, T_flange_tcp}, right: ...}``.

    For DuoBench, T_WB = T_world_shared @ T_shared_armbase (``entry-vla duobench-frames``);
    the stock scene uses FR3 arms (Panda DH used here, kinematic equivalence UNVERIFIED)
    and Robotiq 2F-85 TCP offsets (must be supplied; the Franka-hand default is wrong there)."""
    spec = (cfg.get("robot") or {}).get("arms", "from_backend")
    if spec == "from_backend":
        if not hasattr(backend, "arms"):
            raise ValueError(f"backend {backend.name} does not expose arm models; set robot.arms explicitly")
        return backend.arms
    arms = {}
    for a in ARMS:
        s = spec[a]
        if s.get("T_WB") is None:
            raise ValueError(f"robot.arms.{a}.T_WB is required")
        tcp = _T(s["T_flange_tcp"]) if s.get("T_flange_tcp") is not None else FRANKA_HAND_TCP.copy()
        arms[a] = PandaArm(T_WB=_T(s["T_WB"]), T_flange_tcp=tcp)
    return arms


def make_region(cfg: Dict[str, Any], seed: int) -> RegionEstimator:
    r = cfg["region"]
    return RegionEstimator(r["mode"], r.get("oracle_points"), float(r.get("sigma_m", 0.0)), seed=seed)


def make_policy(cfg: Dict[str, Any]):
    p = cfg["policy"]
    t = p["type"]
    if t == "mock_oracle":
        from ..policies.base import MockOraclePolicy
        return MockOraclePolicy(horizon=int(p.get("horizon", 8)))
    if t == "smolvla":
        from ..policies.smolvla_adapter import SmolVLAPolicyAdapter
        with open(p["action_stats"]) as f:
            stats = json.load(f)
        return SmolVLAPolicyAdapter(p["checkpoint"], p["cameras"], p["image_hw"], stats, device=p.get("device", "cuda:0"))
    if t == "remote":
        import os
        from ..policies.service import RemotePolicy
        key = os.environ.get(p.get("authkey_env", "ENTRY_VLA_POLICY_AUTHKEY"), "").encode()
        if not key:
            raise ValueError("set the policy service authkey env var (default ENTRY_VLA_POLICY_AUTHKEY)")
        return RemotePolicy((p.get("host", "127.0.0.1"), int(p["port"])), key, p["cameras"],
                            send_depth=bool(p.get("send_depth", False)), send_oracle=False)
    raise ValueError(f"unknown policy type {t!r} (expert replay is diagnostic-only and not selectable here)")


def make_selector(cfg: Dict[str, Any], seed: int, scorer=None, proposer=None, demo_library=None) -> Selector:
    s = cfg["selector"]
    t = s["type"]
    if t == "fixed_canonical":
        sel: Selector = FixedCanonicalSelector({a: _T(s["T_OE"][a]) for a in ARMS})
    elif t == "distance_shell":
        sel = DistanceShellSelector(n=int(s.get("n", 32)), radius=float(s.get("radius", 0.20)), seed=seed)
    elif t == "visibility_geometric":
        sel = VisibilityGeometricSelector(np.asarray(s["K"]), s["image_wh"], n=int(s.get("n", 32)),
                                          radius=float(s.get("radius", 0.20)), seed=seed)
    elif t == "nearest_demo":
        if not demo_library:
            raise ValueError("nearest_demo needs a demo-entry library (annotations)")
        sel = NearestDemoSelector(demo_library, k=int(s.get("k", 5)))
    elif t == "supervised":
        if proposer is None:
            raise ValueError("supervised selector needs a trained proposal model")
        sel = ModelSelector(proposer)
    else:
        raise ValueError(f"unknown selector {t!r}")
    if s.get("scorer"):
        if scorer is None:
            raise ValueError("selector.scorer set but no scorer loaded")
        sel = ScoredSelector(sel, scorer, mode=s["scorer"], min_success=float(s.get("min_success", 0.0)),
                             duration_weight=float(s.get("duration_weight", 0.0)))
    return sel


def make_prep(cfg: Dict[str, Any]) -> PrepConfig:
    p = dict(cfg.get("prep") or {})
    p.pop("planner", None)
    return PrepConfig(**p)


def make_planner(cfg: Dict[str, Any], arms):
    """``prep.planner``: ``jointspace`` (default, simulator planner, NOT MoveIt) or ``moveit`` (moveit_py, opt-in)."""
    kind = (cfg.get("prep") or {}).get("planner", "jointspace")
    if kind == "jointspace":
        from ..sim.preparation import JointSpacePlanner
        return JointSpacePlanner(arms, make_prep(cfg))
    if kind == "moveit":
        from ..sim.moveit_planner import MoveItPlanner
        m = cfg.get("moveit") or {}
        return MoveItPlanner(arms, group=m.get("group", "both_arms"), joint_order=m.get("joint_order"))
    raise ValueError(f"unknown planner {kind!r}")


def make_filter(cfg: Dict[str, Any]) -> FilterConfig:
    f = dict(cfg.get("filter") or {})
    if "max_approach_angle_from_down_deg" in f:
        f["max_approach_angle_from_down"] = float(np.deg2rad(f.pop("max_approach_angle_from_down_deg")))
    return FilterConfig(**f)


def make_trial_cfg(cfg: Dict[str, Any], backend_name: str) -> TrialConfig:
    t = dict(cfg.get("trial") or {})
    t.setdefault("controller", backend_name)
    return TrialConfig(**t)


def wrist_hand_eye(cfg: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    h = (cfg.get("robot") or {}).get("wrist_T_EC")
    return None if not h else {a: _T(h[a]) for a in ARMS}
