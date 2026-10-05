"""Stopped-entry preparation: plan, physically execute, settle, verify, hand off.

Planner identity: ``entry_vla.jointspace_ik_interp/v1`` -- damped-least-squares IK
to the entry pose, joint-space linear interpolation with a velocity limit,
and capsule-model clearance checks (arm/arm, arm/table). It is NOT MoveIt
and does not model object collisions; real-system claims require the
MoveIt integration (``planner: moveit``, not implemented here).

Main trials must physically execute preparation. ``diagnostic_teleport`` is a
separately labelled intervention and only works on backends with verified
snapshot support; it never teleports held objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..geometry.franka import PandaArm, dual_arm_clearance, table_clearance
from ..geometry.rotations import geodesic_angle
from .contracts import ARMS, BimanualTcpCommand, Observation, UnsupportedCapabilityError

PLANNER_ID = "entry_vla.jointspace_ik_interp/v2"   # v2: Cartesian via-point retries


@dataclass
class PrepConfig:
    max_joint_speed: float = 0.8          # rad/s for interpolation
    control_dt: float = 0.1               # s per command (backend control period)
    min_pair_clearance: float = 0.02      # m, capsule model
    table_z: float = 0.0
    min_table_clearance: float = 0.01
    pos_tol: float = 0.01                 # arrival tolerance (m)
    rot_tol: float = 0.05                 # rad
    settle_vel_tol: float = 0.02          # rad/s joint speed for "stopped"
    settle_steps: int = 3                 # consecutive stopped steps required
    max_settle_steps: int = 30
    max_exec_steps: int = 300
    waypoint_mode: str = "dense"          # "dense" | "final_only" (backend plans internally, e.g. PerAct2)
    via_lift_m: float = 0.15              # retry: lift TCPs by this much before transit
    via_outward_m: float = 0.10           # retry: also move TCPs away from the arms' midline
    arrival_order: str = "simultaneous"   # "simultaneous" | "left_first" | "right_first" (scheduling ablation)
    gripper_during_transit: float = 1.0   # open


@dataclass
class Plan:
    ok: bool
    reason: str = ""
    q_path: Dict[str, np.ndarray] = field(default_factory=dict)
    T_path: Dict[str, np.ndarray] = field(default_factory=dict)
    min_pair_clearance: Optional[float] = None
    min_table_clearance: Optional[float] = None
    planning_time_s: float = 0.0
    planner: str = PLANNER_ID


@dataclass
class HandoffRecord:
    arrived: bool
    stopped: bool
    failure: Optional[str]
    exec_steps: int
    settle_steps: int
    prep_sim_time_s: float
    arrival_pos_err: Dict[str, float]
    arrival_rot_err: Dict[str, float]
    arrival_joint_speed: Dict[str, Optional[float]]
    gripper_width: Dict[str, Optional[float]]
    min_pair_clearance: Optional[float]
    pause_added: bool
    intervention: Optional[str] = None
    planner: str = PLANNER_ID
    plan_time_s: float = 0.0
    route: str = ""
    obs: Optional[Observation] = None

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("obs", None)
        return d


class JointSpacePlanner:
    """Direct joint-space interpolation; on clearance failure, retries through
    Cartesian via-points (lift, then lift+outward) and reports which route was used."""

    planner_id = PLANNER_ID

    def __init__(self, arms: Dict[str, PandaArm], cfg: PrepConfig):
        self.arms = arms
        self.cfg = cfg

    def _segment(self, q0: Dict[str, np.ndarray], q1: Dict[str, np.ndarray]):
        span = max(np.max(np.abs(q1[a] - q0[a])) for a in ARMS)
        n = max(1, int(np.ceil(span / (self.cfg.max_joint_speed * self.cfg.control_dt))))
        s = np.linspace(0, 1, n + 1)[1:, None]
        path = {a: q0[a][None] + s * (q1[a] - q0[a])[None] for a in ARMS}
        min_pair, min_table = np.inf, np.inf
        for k in range(n):
            min_pair = min(min_pair, dual_arm_clearance(self.arms["left"], path["left"][k], self.arms["right"],
                                                        path["right"][k]))
            for a in ARMS:
                min_table = min(min_table, table_clearance(self.arms[a], path[a][k], self.cfg.table_z))
        return path, min_pair, min_table

    def _ik_all(self, T: Dict[str, np.ndarray], seed: Dict[str, np.ndarray]):
        out = {}
        for a in ARMS:
            q, ok, info = self.arms[a].ik(T[a], q_init=seed[a])
            if not ok:
                return None, f"{a}:ik_failed(pos_err={info['pos_err']:.3g})"
            out[a] = q
        return out, ""

    def _vias(self, q_start: Dict[str, np.ndarray], T_goal: Dict[str, np.ndarray]) -> List[List[Dict[str, np.ndarray]]]:
        T0 = {a: self.arms[a].fk(q_start[a]) for a in ARMS}
        mid = 0.5 * (self.arms["left"].T_WB[:3, 3] + self.arms["right"].T_WB[:3, 3])
        axis = self.arms["left"].T_WB[:3, 3] - self.arms["right"].T_WB[:3, 3]
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        sign = {"left": 1.0, "right": -1.0}

        def shifted(Ts, dz, out):
            r = {}
            for a in ARMS:
                T = Ts[a].copy()
                T[2, 3] += dz
                T[:3, 3] += sign[a] * out * axis
                r[a] = T
            return r
        lift, out = self.cfg.via_lift_m, self.cfg.via_outward_m
        return [[shifted(T0, lift, 0.0), shifted(T_goal, lift, 0.0)],
                [shifted(T0, lift, out), shifted(T_goal, lift, out)]]

    def plan(self, q_start: Dict[str, np.ndarray], T_goal: Dict[str, np.ndarray]) -> Plan:
        import time
        t0 = time.perf_counter()
        q_goal, why = self._ik_all(T_goal, q_start)
        if q_goal is None:
            return Plan(False, why, planning_time_s=time.perf_counter() - t0)
        if self.cfg.arrival_order != "simultaneous":
            return self._sequential(q_start, q_goal, t0)
        routes = [[]] + self._vias(q_start, T_goal)
        last_reason, last_pair, last_table = "", None, None
        for r_i, vias in enumerate(routes):
            qs = [q_start]
            ok = True
            for T_via in vias:
                qv, why = self._ik_all(T_via, qs[-1])
                if qv is None:
                    ok, last_reason = False, f"via:{why}"
                    break
                qs.append(qv)
            if not ok:
                continue
            qs.append(q_goal)
            paths, min_pair, min_table = [], np.inf, np.inf
            for qa, qb in zip(qs[:-1], qs[1:]):
                pth, mp, mt = self._segment(qa, qb)
                paths.append(pth)
                min_pair, min_table = min(min_pair, mp), min(min_table, mt)
            last_pair, last_table = float(min_pair), float(min_table)
            if min_pair < self.cfg.min_pair_clearance:
                last_reason = "pair_clearance"
                continue
            if min_table < self.cfg.min_table_clearance:
                last_reason = "table_clearance"
                continue
            q_path = {a: np.concatenate([p[a] for p in paths]) for a in ARMS}
            T_path = {a: np.stack([self.arms[a].fk(q) for q in q_path[a]]) for a in ARMS}
            plan = Plan(True, q_path=q_path, T_path=T_path, min_pair_clearance=last_pair,
                        min_table_clearance=last_table, planning_time_s=time.perf_counter() - t0)
            plan.reason = ["direct", "via_lift", "via_lift_outward"][r_i]
            return plan
        return Plan(False, last_reason or "no_route", min_pair_clearance=last_pair, min_table_clearance=last_table,
                    planning_time_s=time.perf_counter() - t0)

    def _sequential(self, q_start, q_goal, t0) -> Plan:
        """One arm travels while the other holds still, then the other arm travels."""
        import time
        order = {"left_first": ("left", "right"), "right_first": ("right", "left")}.get(self.cfg.arrival_order)
        if order is None:
            raise ValueError(f"unknown arrival_order {self.cfg.arrival_order!r}")
        first, second = order
        q_mid = {first: q_goal[first], second: q_start[second]}
        paths, min_pair, min_table = [], np.inf, np.inf
        for qa, qb in ((q_start, q_mid), (q_mid, q_goal)):
            pth, mp, mt = self._segment(qa, qb)
            paths.append(pth)
            min_pair, min_table = min(min_pair, mp), min(min_table, mt)
        dt = time.perf_counter() - t0
        if min_pair < self.cfg.min_pair_clearance:
            return Plan(False, "sequential:pair_clearance", min_pair_clearance=float(min_pair), planning_time_s=dt)
        if min_table < self.cfg.min_table_clearance:
            return Plan(False, "sequential:table_clearance", min_table_clearance=float(min_table), planning_time_s=dt)
        q_path = {a: np.concatenate([p_[a] for p_ in paths]) for a in ARMS}
        T_path = {a: np.stack([self.arms[a].fk(q) for q in q_path[a]]) for a in ARMS}
        plan = Plan(True, q_path=q_path, T_path=T_path, min_pair_clearance=float(min_pair),
                    min_table_clearance=float(min_table), planning_time_s=dt)
        plan.reason = f"sequential:{self.cfg.arrival_order}"
        return plan


def _speed(obs: Observation, a: str) -> Optional[float]:
    dq = obs.arms[a].dq
    return None if dq is None else float(np.max(np.abs(dq)))


def execute_preparation(backend, obs: Observation, T_goal: Dict[str, np.ndarray], planner: JointSpacePlanner,
                        arbiter, cfg: PrepConfig, gripper_goal: Optional[Dict[str, float]] = None,
                        pause_steps: int = 0) -> HandoffRecord:
    """Physically drive both arms to ``T_goal``, settle, flush, and release to the policy.

    ``pause_steps`` adds an explicit hold after settling (recorded as ``pause_added``
    so experiments can test whether a pause changes the task)."""
    grip = gripper_goal or {a: cfg.gripper_during_transit for a in ARMS}
    plan = planner.plan({a: obs.arms[a].q for a in ARMS}, T_goal)
    t_start = obs.time
    if not plan.ok:
        return HandoffRecord(False, False, f"planning:{plan.reason}", 0, 0, 0.0, {}, {}, {}, {},
                             plan.min_pair_clearance, False, plan_time_s=plan.planning_time_s, obs=obs)
    arbiter.acquire("planner")
    steps = 0
    try:
        path = plan.T_path if cfg.waypoint_mode == "dense" else {a: plan.T_path[a][-1:] for a in ARMS}
        n = len(path["left"])
        for k in range(n):
            arbiter.check("planner")
            res = backend.step(BimanualTcpCommand({a: path[a][k] for a in ARMS}, grip), obs)
            obs, steps = res.obs, steps + 1
            if res.terminated:
                return _record(False, False, "transit:env_terminated", steps, 0, obs, T_goal, plan, t_start, pause_steps)
        # Hold the goal until arrived and stopped (bounded settling).
        hold = BimanualTcpCommand(dict(T_goal), grip)
        stopped_run, settle = 0, 0
        while settle < cfg.max_settle_steps and steps < cfg.max_exec_steps:
            pe = max(np.linalg.norm(obs.arms[a].T_WE[:3, 3] - T_goal[a][:3, 3]) for a in ARMS)
            re = max(float(geodesic_angle(obs.arms[a].T_WE[:3, :3], T_goal[a][:3, :3])) for a in ARMS)
            sp = [_speed(obs, a) for a in ARMS]
            stopped = all(s is not None and s < cfg.settle_vel_tol for s in sp)
            if pe < cfg.pos_tol and re < cfg.rot_tol and stopped:
                stopped_run += 1
                if stopped_run >= cfg.settle_steps:
                    break
            else:
                stopped_run = 0
            res = backend.step(hold, obs)
            obs, steps, settle = res.obs, steps + 1, settle + 1
        for _ in range(pause_steps):
            obs = backend.step(hold, obs).obs
            steps += 1
        # Cancel old targets / flush: command the *measured* pose so no stale goal remains queued.
        flush = BimanualTcpCommand({a: obs.arms[a].T_WE.copy() for a in ARMS}, grip)
        obs = backend.step(flush, obs).obs
        steps += 1
        arrived = all(np.linalg.norm(obs.arms[a].T_WE[:3, 3] - T_goal[a][:3, 3]) < cfg.pos_tol and
                      float(geodesic_angle(obs.arms[a].T_WE[:3, :3], T_goal[a][:3, :3])) < cfg.rot_tol for a in ARMS)
        stopped = stopped_run >= cfg.settle_steps
        failure = None if arrived and stopped else ("transit:not_arrived" if not arrived else "transit:not_stopped")
        return _record(arrived, stopped, failure, steps, settle, obs, T_goal, plan, t_start, pause_steps)
    finally:
        arbiter.release("planner")


def _record(arrived, stopped, failure, steps, settle, obs, T_goal, plan, t_start, pause_steps) -> HandoffRecord:
    return HandoffRecord(
        arrived=arrived, stopped=stopped, failure=failure, exec_steps=steps, settle_steps=settle,
        prep_sim_time_s=float(obs.time - t_start),
        arrival_pos_err={a: float(np.linalg.norm(obs.arms[a].T_WE[:3, 3] - T_goal[a][:3, 3])) for a in ARMS},
        arrival_rot_err={a: float(geodesic_angle(obs.arms[a].T_WE[:3, :3], T_goal[a][:3, :3])) for a in ARMS},
        arrival_joint_speed={a: _speed(obs, a) for a in ARMS},
        gripper_width={a: obs.arms[a].gripper_width for a in ARMS},
        min_pair_clearance=plan.min_pair_clearance, pause_added=pause_steps > 0,
        plan_time_s=plan.planning_time_s, route=plan.reason, obs=obs)


def diagnostic_teleport(backend, q_goal: Dict[str, np.ndarray]) -> Observation:
    """LABELLED diagnostic intervention: set joint configuration directly.
    Only for backends with supported snapshots; never used in main trials and
    never moves held objects (refuses if the mock reports an attachment)."""
    if backend.capabilities().state_snapshot != "supported":
        raise UnsupportedCapabilityError(f"{backend.name}: teleport requires supported snapshots")
    snap = backend.get_state_snapshot()
    if snap.get("attached"):
        raise UnsupportedCapabilityError("refusing to teleport while an object is attached (inconsistent grasp)")
    for a in ARMS:
        snap["q"][a] = np.asarray(q_goal[a], dtype=np.float64).copy()
        snap["dq"][a] = np.zeros(7)
    return backend.restore_state_snapshot(snap)
