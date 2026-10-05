"""Bimanual Cartesian state/action contract for the local policy.

State (20D) = for arm in (left, right): position(3) [m], rotation6D(6), measured
gripper width(1) [m].

Action chunk (H x 14) = for arm in (left, right): body-relative translation(3)
[m], relative rotation vector(3) [rad], desired gripper command(1).

For each sequential chunk step ``k`` (per arm)::

    delta_p[k]   = R_prev.T @ (p_next - p_prev)
    delta_rot[k] = Log(R_prev.T @ R_next)

where the first ``prev`` is the *measured* query pose and every subsequent
``prev`` is the preceding *desired* waypoint. Decoding starts from the same
measured query pose. These decoupled increments are NOT the coupled SE(3)
logarithm.

Target provenance must be declared: ``"commanded"`` (raw command targets)
vs ``"measured_future"`` (derived from future measured poses). They are never
silently mixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.rotations import matrix_to_rot6d, rot6d_to_matrix, so3_exp, so3_log

ARMS = ("left", "right")
STATE_DIM = 20
ACTION_DIM_PER_ARM = 7
ACTION_DIM = 14
TARGET_SOURCES = ("commanded", "measured_future")

STATE_NAMES = [f"{arm}.{n}" for arm in ARMS for n in
               ("x", "y", "z", "r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5", "gripper_width")]
ACTION_NAMES = [f"{arm}.{n}" for arm in ARMS for n in
                ("dx", "dy", "dz", "drx", "dry", "drz", "gripper_cmd")]


def encode_state(T_left: np.ndarray, T_right: np.ndarray, width_left: float, width_right: float) -> np.ndarray:
    """Measured TCP poses (in the declared policy frame) + widths -> (20,)."""
    parts = []
    for T, w in ((T_left, width_left), (T_right, width_right)):
        parts += [T[:3, 3], matrix_to_rot6d(T[:3, :3]), [w]]
    return np.concatenate(parts).astype(np.float32)


def decode_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    s = np.asarray(state, dtype=np.float64).reshape(STATE_DIM)
    out = []
    for i in range(2):
        b = s[10 * i: 10 * (i + 1)]
        T = np.eye(4)
        T[:3, :3] = rot6d_to_matrix(b[3:9])
        T[:3, 3] = b[0:3]
        out.append((T, float(b[9])))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def encode_arm_chunk(T_query: np.ndarray, T_targets: np.ndarray, gripper_cmds: np.ndarray) -> np.ndarray:
    """Encode one arm: measured query pose + (H, 4, 4) desired waypoints -> (H, 7)."""
    T_targets = np.asarray(T_targets, dtype=np.float64)
    H = T_targets.shape[0]
    out = np.zeros((H, ACTION_DIM_PER_ARM))
    p_prev, R_prev = T_query[:3, 3], T_query[:3, :3]
    for k in range(H):
        p_next, R_next = T_targets[k, :3, 3], T_targets[k, :3, :3]
        out[k, 0:3] = R_prev.T @ (p_next - p_prev)
        out[k, 3:6] = so3_log(R_prev.T @ R_next)
        out[k, 6] = gripper_cmds[k]
        p_prev, R_prev = p_next, R_next
    return out


def decode_arm_chunk(T_query: np.ndarray, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`encode_arm_chunk`: (H, 7) -> (H, 4, 4) waypoints, (H,) gripper."""
    chunk = np.asarray(chunk, dtype=np.float64)
    H = chunk.shape[0]
    Ts = np.zeros((H, 4, 4))
    p, R = T_query[:3, 3].copy(), T_query[:3, :3].copy()
    for k in range(H):
        p = p + R @ chunk[k, 0:3]
        R = R @ so3_exp(chunk[k, 3:6])
        Ts[k] = np.eye(4)
        Ts[k, :3, :3], Ts[k, :3, 3] = R, p
    return Ts, chunk[:, 6].copy()


def encode_bimanual_chunk(T_query_l, T_query_r, T_targets_l, T_targets_r, grip_l, grip_r) -> np.ndarray:
    return np.concatenate([encode_arm_chunk(T_query_l, T_targets_l, grip_l),
                           encode_arm_chunk(T_query_r, T_targets_r, grip_r)], axis=1).astype(np.float32)


def decode_bimanual_chunk(T_query_l, T_query_r, chunk: np.ndarray):
    chunk = np.asarray(chunk).reshape(-1, ACTION_DIM)
    Tl, gl = decode_arm_chunk(T_query_l, chunk[:, :7])
    Tr, gr = decode_arm_chunk(T_query_r, chunk[:, 7:])
    return Tl, Tr, gl, gr


@dataclass
class ChunkTargetSpec:
    """Declares where chunk targets came from (kept with every exported sample)."""

    source: str                        # one of TARGET_SOURCES
    horizon: int
    frame: str = "W"                   # frame the poses are expressed in before encoding
    gripper_semantics: str = "width_m" # or "open_close_binary" (1=open) ; adapter-declared
    control_assumption: str = ""       # e.g. "measured pose at t+k approximates commanded target at t+k-1"

    def __post_init__(self) -> None:
        if self.source not in TARGET_SOURCES:
            raise ValueError(f"target source must be one of {TARGET_SOURCES}, got {self.source!r}")
        if self.source == "measured_future" and not self.control_assumption:
            raise ValueError("measured_future targets require a documented control_assumption")


def build_training_chunks(
    T_meas_l: np.ndarray, T_meas_r: np.ndarray,
    T_tgt_l: np.ndarray, T_tgt_r: np.ndarray,
    grip_l: np.ndarray, grip_r: np.ndarray,
    spec: ChunkTargetSpec,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-frame (T, H, 14) chunks + (T, H) pad mask for one episode.

    * ``T_meas_*`` (T, 4, 4): measured TCP poses (query anchors).
    * ``T_tgt_*`` (T, 4, 4): the per-step target sequence. For
      ``commanded`` it is the command issued at step t; for
      ``measured_future`` the caller passes the measured poses and target
      ``t+k`` is taken from frame ``t+1+k`` (documented assumption).
    * Terminal steps beyond the episode are padded by repeating the last
      target (zero motion) and masked out (``mask=False``).
    """
    T = T_meas_l.shape[0]
    H = spec.horizon
    shift = 1 if spec.source == "measured_future" else 0
    valid = np.ones(T, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    chunks = np.zeros((T, H, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((T, H), dtype=bool)
    for t in range(T):
        idx = np.arange(t + shift, t + shift + H)
        ok = (idx < T)
        ok[ok] &= valid[idx[ok]]
        # Stop the chunk at the first invalid step; everything after is padding.
        if not ok.all():
            first_bad = int(np.argmin(ok))
            ok[first_bad:] = False
        last = idx[ok][-1] if ok.any() else None
        if last is None:
            # No valid future: anchor-only zero chunk (hold), fully masked.
            tl = np.repeat(T_meas_l[t][None], H, 0)
            tr = np.repeat(T_meas_r[t][None], H, 0)
            gl = np.full(H, grip_l[min(t, T - 1)])
            gr = np.full(H, grip_r[min(t, T - 1)])
        else:
            src = np.where(ok, idx, last)
            tl, tr = T_tgt_l[src], T_tgt_r[src]
            gl, gr = grip_l[src], grip_r[src]
        chunks[t] = encode_bimanual_chunk(T_meas_l[t], T_meas_r[t], tl, tr, gl, gr)
        mask[t] = ok
    return chunks, mask
