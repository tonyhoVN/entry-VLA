"""Homogeneous SE(3) transforms.

Naming: ``T_AB`` maps coordinates expressed in frame B to frame A, i.e.
``p_A = T_AB @ p_B``. Composition chains inner indices: ``T_AC = T_AB @ T_BC``.
Frames used across the project:

* ``W``  shared world frame
* ``BL``/``BR`` left/right arm base frames
* ``EL``/``ER`` (or ``E``) TCP frames
* ``C``  a camera frame (OpenCV convention: +z forward, +x right, +y down)
* ``O``  object / interaction-region reference frame
"""

from __future__ import annotations

import numpy as np

from .rotations import matrix_to_quat, quat_to_matrix


def make_T(R: np.ndarray | None = None, p: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = np.asarray(p, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse; supports ``(..., 4, 4)`` batches."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, p)
    out[..., 3, 3] = 1.0
    return out


def apply_T(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform ``(N, 3)`` points (or a single point) by ``T``."""
    pts = np.asarray(points, dtype=np.float64)
    return pts @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def pose_to_T(pos: np.ndarray, quat: np.ndarray, order: str) -> np.ndarray:
    """Position + quaternion with declared ``order`` ("xyzw" or "wxyz") -> T."""
    return make_T(quat_to_matrix(quat, order), pos)


def T_to_pose(T: np.ndarray, order: str) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64)
    return T[:3, 3].copy(), matrix_to_quat(T[:3, :3], order)


def is_valid_T(T: np.ndarray, atol: float = 1e-6) -> bool:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1], atol=atol):
        return False
    R = T[:3, :3]
    return bool(np.allclose(R @ R.T, np.eye(3), atol=atol) and abs(np.linalg.det(R) - 1) < atol)


def tcp_from_camera_pose(T_WC: np.ndarray, T_EC: np.ndarray) -> np.ndarray:
    """Given a desired (wrist) camera pose ``T_WC`` and the fixed hand-eye
    transform ``T_EC`` (camera expressed in TCP frame), return ``T_WE``.

    ``T_WC = T_WE @ T_EC``  =>  ``T_WE = T_WC @ inv(T_EC)``.
    """
    return np.asarray(T_WC) @ invert_T(T_EC)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world transform (OpenCV camera: +z toward target, +y down)."""
    eye = np.asarray(eye, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - eye
    z /= np.linalg.norm(z)
    up = np.asarray(up, dtype=np.float64)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:  # looking along up: pick any perpendicular
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(x) < 1e-8:
            x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.stack([x, y, z], axis=1), eye)


# Converting OpenGL-style camera frames (+x right, +y up, -z forward; used by
# MuJoCo and CoppeliaSim viewers) to OpenCV frames (+x right, +y down, +z fwd).
T_GL_CV = make_T(np.diag([1.0, -1.0, -1.0]))


def opengl_to_opencv_camera(T_W_Cgl: np.ndarray) -> np.ndarray:
    """``T_W_Ccv = T_W_Cgl @ T_Cgl_Ccv``; the flip matrix is its own inverse."""
    return np.asarray(T_W_Cgl) @ T_GL_CV
