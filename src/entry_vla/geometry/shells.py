"""Task-conditioned partial-shell candidate sampling and geometric filtering.

A shell candidate is a *proposal*, not a usable grasp: camera-facing or
region-facing orientation does not establish that the downstream policy can
succeed from there. Candidates are filtered for orientation, table
intersection, IK feasibility/branch, arm clearance and paired clearance;
outcome scoring happens later with a frozen policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .franka import PandaArm, dual_arm_clearance, table_clearance
from .rotations import rot_z
from .transforms import look_at, make_T, tcp_from_camera_pose


@dataclass
class ShellSpec:
    """Spherical shell section around an interaction-region centre (world).

    Angles are radians. ``azimuth`` is about world +z measured from +x;
    ``elevation`` is above the horizontal plane.
    """

    center: np.ndarray
    radius: float = 0.20                # baseline parameter, not ground truth
    radius_jitter: float = 0.03
    azimuth_range: tuple[float, float] = (-np.pi, np.pi)
    elevation_range: tuple[float, float] = (np.deg2rad(20), np.deg2rad(80))
    roll_range: tuple[float, float] = (0.0, 0.0)  # about the approach axis

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)


def sample_shell_tcp_poses(spec: ShellSpec, n: int, rng: np.random.Generator,
                           T_EC: np.ndarray | None = None) -> np.ndarray:
    """Sample ``n`` TCP poses ``T_WE`` on the shell, TCP +z pointing at the centre.

    If ``T_EC`` (wrist camera pose in TCP frame) is given, the *camera* is
    aimed at the centre instead and the TCP pose is recovered with
    ``T_WE = T_WC @ inv(T_EC)``.
    """
    out = np.empty((n, 4, 4))
    for i in range(n):
        az = rng.uniform(*spec.azimuth_range)
        el = rng.uniform(*spec.elevation_range)
        r = spec.radius + rng.uniform(-spec.radius_jitter, spec.radius_jitter)
        offset = r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        eye = spec.center + offset
        T_look = look_at(eye, spec.center)            # +z toward centre
        roll = rng.uniform(*spec.roll_range)
        T_look = T_look @ make_T(rot_z(roll))
        out[i] = tcp_from_camera_pose(T_look, T_EC) if T_EC is not None else T_look
    return out


@dataclass
class FilterConfig:
    max_approach_angle_from_down: float = np.deg2rad(75)  # TCP z vs world -z
    table_z: float = 0.0
    min_table_clearance: float = 0.02
    min_tcp_height: float = 0.01
    min_pair_clearance: float = 0.02
    max_joint_jump: float = 2.0      # rad, IK branch consistency vs seed (inf-norm)
    ik_iters: int = 150


@dataclass
class FilterResult:
    accepted: bool
    reasons: list = field(default_factory=list)
    q: np.ndarray | None = None
    table_clearance: float | None = None


def filter_single_arm(T_WE: np.ndarray, arm: PandaArm, q_seed: np.ndarray, cfg: FilterConfig) -> FilterResult:
    reasons = []
    z_axis = T_WE[:3, 2]
    ang = np.arccos(np.clip(-z_axis[2], -1, 1))
    if ang > cfg.max_approach_angle_from_down:
        reasons.append("orientation")
    if T_WE[2, 3] - cfg.table_z < cfg.min_tcp_height:
        reasons.append("tcp_below_table")
    q, ok, _ = arm.ik(T_WE, q_init=q_seed, max_iters=cfg.ik_iters)
    if not ok:
        reasons.append("ik_failed")
    elif np.max(np.abs(q - q_seed)) > cfg.max_joint_jump:
        reasons.append("ik_branch_jump")
    tc = None
    if ok:
        tc = table_clearance(arm, q, cfg.table_z)
        if tc < cfg.min_table_clearance:
            reasons.append("arm_table_intersection")
    return FilterResult(accepted=not reasons, reasons=reasons, q=q if ok else None, table_clearance=tc)


def filter_pair(T_WE_left: np.ndarray, T_WE_right: np.ndarray, left: PandaArm, right: PandaArm,
                q_seed_l: np.ndarray, q_seed_r: np.ndarray, cfg: FilterConfig) -> dict:
    """Filter a paired candidate. Independent per-arm acceptance does not
    establish joint compatibility, so the pair clearance is checked too."""
    rl = filter_single_arm(T_WE_left, left, q_seed_l, cfg)
    rr = filter_single_arm(T_WE_right, right, q_seed_r, cfg)
    reasons = [f"left:{r}" for r in rl.reasons] + [f"right:{r}" for r in rr.reasons]
    clearance = None
    if rl.q is not None and rr.q is not None:
        clearance = dual_arm_clearance(left, rl.q, right, rr.q)
        if clearance < cfg.min_pair_clearance:
            reasons.append("pair_clearance")
    return {"accepted": not reasons, "reasons": reasons, "q_left": rl.q, "q_right": rr.q,
            "pair_clearance": clearance}
