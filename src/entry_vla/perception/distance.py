"""Object point clouds from segmented RGB-D views and distance-to-object fields.

The field is the *unsigned* distance from a query point to the nearest fused
object surface point (a point-sampled distance field). It is not signed: a
camera-only cloud has no reliable inside/outside, and the gripper approaching
from outside never needs the sign. Distances carry the sampling error of the
cloud (pixel footprint at the object's depth).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..geometry.camera import backproject_to_world


def object_points(depth: np.ndarray, valid: np.ndarray, mask: np.ndarray, K: np.ndarray, T_WC: np.ndarray,
                  erode: int = 1) -> np.ndarray:
    """World points of masked pixels with valid depth. ``erode`` shrinks the mask
    by that many pixels to drop depth-edge mixing at object boundaries."""
    m = mask & valid
    for _ in range(max(0, erode)):
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        m = e
    return backproject_to_world(depth, K, T_WC, m) if m.any() else np.zeros((0, 3))


def fuse_voxel(clouds: Sequence[np.ndarray], voxel_size: float = 0.005) -> np.ndarray:
    """Concatenate per-camera world clouds and voxel-grid downsample them with Open3D.
    No outlier or median filtering: a task may involve several objects, and all segmented
    surface points are kept (one averaged point per occupied voxel)."""
    pts = [np.asarray(c, dtype=np.float64).reshape(-1, 3) for c in clouds]
    pts = np.concatenate(pts) if pts else np.zeros((0, 3))
    if len(pts) == 0 or voxel_size <= 0:
        return pts
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return np.asarray(pcd.voxel_down_sample(voxel_size).points)


def distance_to_points(queries: np.ndarray, points: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Nearest-neighbour distance from each query (Q, 3) to the cloud (N, 3)."""
    queries = np.atleast_2d(np.asarray(queries, dtype=np.float64))
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.full(len(queries), np.inf)
    out = np.empty(len(queries))
    for i in range(0, len(points), chunk):
        d = np.linalg.norm(queries[:, None, :] - points[None, i:i + chunk, :], axis=-1).min(axis=1)
        out = d if i == 0 else np.minimum(out, d)
    return out


def first_within(distances: Sequence[float], radius: float, before: Optional[int] = None) -> Optional[int]:
    """First index whose distance is <= radius (optionally only before index ``before``)."""
    end = len(distances) if before is None else min(before, len(distances))
    for i in range(end):
        if distances[i] <= radius:
            return i
    return None


def radius_sweep(distances: Sequence[float], radii: Iterable[float]) -> Dict[str, Optional[int]]:
    return {f"{r:.2f}": first_within(distances, r) for r in radii}
