"""Rotation utilities with explicit quaternion conventions.

Conventions
-----------
* Rotation matrices are 3x3 (or ``(..., 3, 3)``) proper orthonormal matrices.
* Quaternions are NEVER passed around without a declared order. External
  boundaries name the order explicitly: ``"xyzw"`` (RLBench/PyRep, SciPy) or
  ``"wxyz"`` (MuJoCo). Internally we use rotation matrices and the continuous
  6D representation (first two columns of R, Zhou et al. 2019).
* Rotation vectors (axis * angle, radians) are used for relative rotations in
  action chunks: ``Log(R) -> rotvec``, ``Exp(rotvec) -> R``.

Conversions use ``scipy.spatial.transform.Rotation`` (scalar-last ``xyzw``).
Only APIs available in SciPy 1.10 are used (the Python 3.8 simulator env), so
canonical quaternion signs and ``scalar_first`` are handled here.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

QUAT_ORDERS = ("xyzw", "wxyz")
_EPS = 1e-12


def _check_order(order: str) -> None:
    if order not in QUAT_ORDERS:
        raise ValueError(f"quaternion order must be one of {QUAT_ORDERS}, got {order!r}")


def _to_xyzw(q: np.ndarray, order: str) -> np.ndarray:
    _check_order(order)
    q = np.asarray(q, dtype=np.float64)
    return q if order == "xyzw" else np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _from_xyzw(q: np.ndarray, order: str) -> np.ndarray:
    _check_order(order)
    return q if order == "xyzw" else np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def _batched(x: np.ndarray, tail: int):
    """Flatten leading dims for SciPy (1-D stacks only); return (flat, leading shape)."""
    x = np.asarray(x, dtype=np.float64)
    lead = x.shape[:-tail]
    return x.reshape((-1,) + x.shape[-tail:]), lead


def quat_to_matrix(q: np.ndarray, order: str) -> np.ndarray:
    """Unit quaternion (any norm > 0, normalised here) -> rotation matrix."""
    flat, lead = _batched(_to_xyzw(q, order), 1)
    if np.any(np.linalg.norm(flat, axis=-1) < _EPS):
        raise ValueError("zero-norm quaternion")
    return Rotation.from_quat(flat).as_matrix().reshape(lead + (3, 3))


def matrix_to_quat(R: np.ndarray, order: str) -> np.ndarray:
    """Rotation matrix -> unit quaternion with non-negative scalar part."""
    flat, lead = _batched(R, 2)
    q = Rotation.from_matrix(flat).as_quat()
    q[q[:, 3] < 0] *= -1.0
    return _from_xyzw(q.reshape(lead + (4,)), order)


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector (rad) -> rotation matrix. Supports batches."""
    flat, lead = _batched(rotvec, 1)
    return Rotation.from_rotvec(flat).as_matrix().reshape(lead + (3, 3))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (rad), angle in [0, pi]. Supports batches."""
    flat, lead = _batched(R, 2)
    return Rotation.from_matrix(flat).as_rotvec().reshape(lead + (3,))


def geodesic_angle(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Angle (rad) of R_a^T R_b, batched over leading dims."""
    rel = np.swapaxes(np.asarray(R_a, dtype=np.float64), -1, -2) @ np.asarray(R_b, dtype=np.float64)
    flat, lead = _batched(rel, 2)
    ang = Rotation.from_matrix(flat).magnitude().reshape(lead)
    return ang if lead else float(ang)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns of R, flattened column-major: [r00, r10, r20, r01, r11, r21]."""
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt reconstruction of a valid SO(3) matrix from a 6D vector."""
    r6 = np.asarray(r6, dtype=np.float64)
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), _EPS)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2p / np.maximum(np.linalg.norm(a2p, axis=-1, keepdims=True), _EPS)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """Nearest rotation (Frobenius) via SVD, with det=+1 enforced."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def is_rotation(R: np.ndarray, atol: float = 1e-6) -> bool:
    R = np.asarray(R, dtype=np.float64)
    return bool(np.allclose(R @ R.T, np.eye(3), atol=atol) and abs(np.linalg.det(R) - 1.0) < atol)


def rot_x(a: float) -> np.ndarray:
    return Rotation.from_euler("x", float(a)).as_matrix()


def rot_y(a: float) -> np.ndarray:
    return Rotation.from_euler("y", float(a)).as_matrix()


def rot_z(a: float) -> np.ndarray:
    return Rotation.from_euler("z", float(a)).as_matrix()


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed rotation drawn from ``rng``."""
    return Rotation.random(random_state=rng).as_matrix()
