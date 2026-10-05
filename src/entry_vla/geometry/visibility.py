"""Target visibility from a camera: frustum, range and depth-test occlusion.

Visibility is a *weak readiness signal* used for annotation and candidate
filtering. It is never a success label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import project_world


@dataclass
class VisibilityResult:
    in_frustum_fraction: float
    unoccluded_fraction: float     # among in-frustum points
    visible_fraction: float        # in frustum AND not occluded
    mean_pixel_margin: float       # distance of visible points from image border (px)

    @property
    def visible(self) -> bool:
        return self.visible_fraction > 0


def target_visibility(
    target_points_W: np.ndarray,
    K: np.ndarray,
    T_WC: np.ndarray,
    width: int,
    height: int,
    depth_image: np.ndarray | None = None,
    near: float = 0.01,
    far: float = 10.0,
    occlusion_tol: float = 0.01,
    border_px: float = 0.0,
) -> VisibilityResult:
    """Fraction of target sample points visible in a view.

    If ``depth_image`` (metric, same camera) is given, a point is occluded when
    the rendered depth at its pixel is more than ``occlusion_tol`` metres in
    front of it.
    """
    pts = np.asarray(target_points_W, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return VisibilityResult(0.0, 0.0, 0.0, 0.0)
    uv, z = project_world(pts, K, T_WC)
    inside = (
        (z > near) & (z < far)
        & (uv[:, 0] >= border_px) & (uv[:, 0] <= width - 1 - border_px)
        & (uv[:, 1] >= border_px) & (uv[:, 1] <= height - 1 - border_px)
    )
    unocc = inside.copy()
    if depth_image is not None and inside.any():
        ui = np.clip(np.round(uv[inside, 0]).astype(int), 0, width - 1)
        vi = np.clip(np.round(uv[inside, 1]).astype(int), 0, height - 1)
        d = np.asarray(depth_image)[vi, ui]
        occluded = np.isfinite(d) & (d < z[inside] - occlusion_tol)
        unocc[np.where(inside)[0][occluded]] = False
    n_in = int(inside.sum())
    margin = 0.0
    if unocc.any():
        u, v = uv[unocc, 0], uv[unocc, 1]
        margin = float(np.mean(np.minimum.reduce([u, v, width - 1 - u, height - 1 - v])))
    return VisibilityResult(
        in_frustum_fraction=n_in / len(pts),
        unoccluded_fraction=(float(unocc.sum()) / n_in) if n_in else 0.0,
        visible_fraction=float(unocc.sum()) / len(pts),
        mean_pixel_margin=margin,
    )
