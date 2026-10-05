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
"""

from __future__ import annotations

import numpy as np

QUAT_ORDERS = ("xyzw", "wxyz")
_EPS = 1e-12


def _check_order(order: str) -> None:
    if order not in QUAT_ORDERS:
        raise ValueError(f"quaternion order must be one of {QUAT_ORDERS}, got {order!r}")


def quat_to_wxyz(q: np.ndarray, order: str) -> np.ndarray:
    _check_order(order)
    q = np.asarray(q, dtype=np.float64)
    if order == "wxyz":
        return q.copy()
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def quat_from_wxyz(q_wxyz: np.ndarray, order: str) -> np.ndarray:
    _check_order(order)
    q = np.asarray(q_wxyz, dtype=np.float64)
    if order == "wxyz":
        return q.copy()
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def quat_to_matrix(q: np.ndarray, order: str) -> np.ndarray:
    """Unit quaternion (any norm > 0, normalised here) -> rotation matrix."""
    w, x, y, z = np.moveaxis(quat_to_wxyz(q, order), -1, 0)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if np.any(n < _EPS):
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return R.reshape(R.shape[:-1] + (3, 3))


def matrix_to_quat(R: np.ndarray, order: str) -> np.ndarray:
    """Rotation matrix -> unit quaternion with non-negative scalar part."""
    R = np.asarray(R, dtype=np.float64)
    batch = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)
    out = np.empty((Rf.shape[0], 4))
    for i, m in enumerate(Rf):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
        q = np.array([w, x, y, z])
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        out[i] = q
    return quat_from_wxyz(out.reshape(batch + (4,)), order)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vector (rad) -> rotation matrix. Supports batches."""
    rv = np.asarray(rotvec, dtype=np.float64)
    if rv.ndim > 1:
        return np.stack([so3_exp(r) for r in rv.reshape(-1, 3)]).reshape(rv.shape[:-1] + (3, 3))
    theta = np.linalg.norm(rv)
    K = skew(rv)
    if theta < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(theta) / theta * K + (1 - np.cos(theta)) / theta**2 * K @ K


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (rad), angle in [0, pi]. Supports batches."""
    R = np.asarray(R, dtype=np.float64)
    if R.ndim > 2:
        return np.stack([so3_log(m) for m in R.reshape(-1, 3, 3)]).reshape(R.shape[:-2] + (3,))
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * vee
    if np.pi - theta < 1e-4:
        # Near pi the antisymmetric part vanishes; recover the axis from R + I.
        B = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(B), 0.0, None))
        k = int(np.argmax(axis))
        axis[k] = np.sqrt(B[k, k])
        for j in range(3):
            if j != k:
                axis[j] = B[k, j] / axis[k]
        axis /= np.linalg.norm(axis)
        # Fix sign using the (small) antisymmetric part when available.
        if np.dot(axis, vee) < 0:
            axis = -axis
        return axis * theta
    return theta / (2.0 * np.sin(theta)) * vee


def geodesic_angle(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Angle (rad) of R_a^T R_b, batched over leading dims."""
    R_a = np.asarray(R_a, dtype=np.float64)
    R_b = np.asarray(R_b, dtype=np.float64)
    rel = np.swapaxes(R_a, -1, -2) @ R_b
    tr = np.trace(rel, axis1=-2, axis2=-1)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


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
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return quat_to_matrix(q, "wxyz")
