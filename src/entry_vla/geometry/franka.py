"""Franka Emika Panda kinematics (analytic FK, damped least-squares IK).

Parameters are the published Franka modified (Craig) DH table and joint
limits from the Franka Control Interface documentation
(https://frankaemika.github.io/docs/control_parameters.html). The default
hand TCP offset (0.1034 m along flange z, -45 deg about z) is the documented
``F_T_EE`` for the standard Franka Hand. DuoBench's published scene uses
Robotiq 2F-85 grippers, whose TCP offset differs: always pass the TCP
transform of the gripper actually mounted (``T_flange_tcp``).

This is a kinematic model only: it ignores the simulator's actual link
meshes. Use the backend's own kinematics/collision checker where available;
the clearance helpers here are conservative capsule approximations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rotations import rot_z, so3_log
from .transforms import invert_T, make_T

# (a, d, alpha) per joint, modified DH; theta = q_i. Final row is the flange.
PANDA_MDH = np.array([
    [0.0, 0.333, 0.0],
    [0.0, 0.0, -np.pi / 2],
    [0.0, 0.316, np.pi / 2],
    [0.0825, 0.0, np.pi / 2],
    [-0.0825, 0.384, -np.pi / 2],
    [0.0, 0.0, np.pi / 2],
    [0.088, 0.0, np.pi / 2],
])
PANDA_FLANGE_D = 0.107
PANDA_Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
PANDA_Q_READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
FRANKA_HAND_TCP = make_T(rot_z(-np.pi / 4), [0.0, 0.0, 0.1034])
# Approximate link capsule radius used for conservative clearance checks.
PANDA_LINK_RADIUS = 0.08


def _mdh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


@dataclass
class PandaArm:
    """One arm: base pose in world (``T_WB``) and TCP offset on the flange."""

    T_WB: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_flange_tcp: np.ndarray = field(default_factory=lambda: FRANKA_HAND_TCP.copy())
    q_min: np.ndarray = field(default_factory=lambda: PANDA_Q_MIN.copy())
    q_max: np.ndarray = field(default_factory=lambda: PANDA_Q_MAX.copy())

    def link_frames(self, q: np.ndarray) -> list[np.ndarray]:
        """World transforms of joint frames 1..7, flange and TCP (9 frames)."""
        q = np.asarray(q, dtype=np.float64).reshape(7)
        T = np.asarray(self.T_WB, dtype=np.float64).copy()
        frames = []
        for i in range(7):
            a, d, alpha = PANDA_MDH[i]
            T = T @ _mdh(a, d, alpha, q[i])
            frames.append(T.copy())
        T = T @ make_T(p=[0.0, 0.0, PANDA_FLANGE_D])
        frames.append(T.copy())
        frames.append(T @ self.T_flange_tcp)
        return frames

    def fk(self, q: np.ndarray) -> np.ndarray:
        """``T_WE`` (TCP in world)."""
        return self.link_frames(q)[-1]

    def within_limits(self, q: np.ndarray, margin: float = 0.0) -> bool:
        q = np.asarray(q)
        return bool(np.all(q >= self.q_min + margin) and np.all(q <= self.q_max - margin))

    def jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """6x7 geometric-style Jacobian [v; w] in world frame by finite differences."""
        T0 = self.fk(q)
        J = np.zeros((6, 7))
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = eps
            T1 = self.fk(np.asarray(q) + dq)
            J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
            J[3:, i] = so3_log(T1[:3, :3] @ T0[:3, :3].T) / eps
        return J

    def ik(
        self,
        T_WE_target: np.ndarray,
        q_init: np.ndarray | None = None,
        max_iters: int = 200,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        damping: float = 1e-2,
        step_limit: float = 0.2,
    ) -> tuple[np.ndarray, bool, dict]:
        """Damped least-squares IK with joint-limit clamping.

        Returns ``(q, success, info)``. Different ``q_init`` seeds can reach
        different IK branches; callers that care about branch consistency
        should seed from the current configuration.
        """
        q = np.array(PANDA_Q_READY if q_init is None else q_init, dtype=np.float64)
        info = {"iters": 0, "pos_err": np.inf, "rot_err": np.inf}
        for it in range(max_iters):
            T = self.fk(q)
            e_p = T_WE_target[:3, 3] - T[:3, 3]
            e_r = so3_log(T_WE_target[:3, :3] @ T[:3, :3].T)
            info.update(iters=it, pos_err=float(np.linalg.norm(e_p)), rot_err=float(np.linalg.norm(e_r)))
            if info["pos_err"] < pos_tol and info["rot_err"] < rot_tol:
                return q, self.within_limits(q), info
            J = self.jacobian(q)
            e = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > step_limit:
                dq *= step_limit / n
            q = np.clip(q + dq, self.q_min, self.q_max)
        return q, False, info

    def capsules(self, q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Line segments between successive frame origins (base..TCP)."""
        pts = [np.asarray(self.T_WB)[:3, 3]] + [F[:3, 3] for F in self.link_frames(q)]
        return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1) if np.linalg.norm(pts[i + 1] - pts[i]) > 1e-6]


def segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Minimum distance between segments [p1,q1] and [p2,q2]."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e < 1e-12:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def dual_arm_clearance(left: PandaArm, q_l: np.ndarray, right: PandaArm, q_r: np.ndarray,
                       link_radius: float = PANDA_LINK_RADIUS) -> float:
    """Approximate surface clearance (m) between two arms (capsule model;
    negative = predicted overlap). Base segments are excluded."""
    cl = left.capsules(q_l)[1:]
    cr = right.capsules(q_r)[1:]
    dmin = min(segment_distance(a, b, c, d) for a, b in cl for c, d in cr)
    return dmin - 2 * link_radius


def table_clearance(arm: PandaArm, q: np.ndarray, table_z: float, from_frame: int = 3) -> float:
    """Lowest distal frame origin above a horizontal table plane (m).

    Only frames from ``from_frame`` onward are checked (proximal links sit
    on the mount).
    """
    frames = arm.link_frames(q)[from_frame:]
    return float(min(F[2, 3] for F in frames) - table_z)


def arm_from_mount(T_W_mount: np.ndarray, T_flange_tcp: np.ndarray | None = None) -> PandaArm:
    return PandaArm(T_WB=np.asarray(T_W_mount, dtype=np.float64),
                    T_flange_tcp=FRANKA_HAND_TCP.copy() if T_flange_tcp is None else T_flange_tcp)


def base_from_world(arm: PandaArm, T_WX: np.ndarray) -> np.ndarray:
    """Express a world pose in the arm base frame: ``T_BX = inv(T_WB) T_WX``."""
    return invert_T(arm.T_WB) @ T_WX
