"""Pinhole cameras, depth decoding, backprojection and consistent image ops.

Conventions
-----------
* Camera frames follow OpenCV: +x right, +y down, +z forward (optical axis).
  Simulator cameras that use OpenGL frames must be converted with
  :func:`entry_vla.geometry.transforms.opengl_to_opencv_camera`.
* Pixel coordinates: integer ``(u, v)`` = (column, row) are pixel *centres*.
  A principal point at the image centre is therefore ``((W-1)/2, (H-1)/2)``.
* Depth images are planar z-depth in metres (float32), NOT ray length.
* ``K`` may carry a negative focal length after a single-axis image flip;
  every function here handles that case (a flip is a reflection, so it cannot
  be folded into a proper camera rotation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transforms import apply_T, invert_T

# Supported raw depth encodings. Never guess: exporters must declare one.
DEPTH_ENCODINGS = (
    "metric_m",              # already float metres -> passthrough
    "uint16_mm",             # integer millimetres -> /1000
    "linear_buffer_01",      # [0,1] linear between near/far (CoppeliaSim/PyRep depth buffer)
    "rgb24_linear_buffer",   # RLBench PNG: 24-bit RGB-packed linear buffer (scale 2**24-1)
    "opengl_zbuffer_01",     # non-linear OpenGL z-buffer in [0,1] (raw MuJoCo/GL readback)
)

RLBENCH_DEPTH_SCALE = 2**24 - 1  # rlbench.backend.const.DEPTH_SCALE (verify per pinned fork)


@dataclass
class CameraIntrinsics:
    K: np.ndarray        # (3, 3)
    width: int
    height: int

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64)
        if self.K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {self.K.shape}")

    @classmethod
    def from_fovy(cls, fovy_rad: float, width: int, height: int) -> "CameraIntrinsics":
        """Square-pixel pinhole with principal point at the image centre
        (MuJoCo-style vertical field of view)."""
        fy = 0.5 * height / np.tan(0.5 * fovy_rad)
        K = np.array([[fy, 0.0, (width - 1) / 2.0], [0.0, fy, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
        return cls(K, width, height)


def decode_depth(
    raw: np.ndarray,
    encoding: str,
    near: float | None = None,
    far: float | None = None,
    rgb_scale: float = RLBENCH_DEPTH_SCALE,
) -> np.ndarray:
    """Convert a raw depth image to float32 metres using a *declared* encoding.

    ``uint16_mm`` divides by 1000 and must only be used for integer mm data;
    applying it to already-metric depth is rejected.
    """
    if encoding not in DEPTH_ENCODINGS:
        raise ValueError(f"unknown depth encoding {encoding!r}; expected one of {DEPTH_ENCODINGS}")
    raw = np.asarray(raw)
    if encoding == "metric_m":
        if not np.issubdtype(raw.dtype, np.floating):
            raise TypeError(f"metric_m depth must be floating point, got {raw.dtype}")
        return raw.astype(np.float32)
    if encoding == "uint16_mm":
        if raw.dtype != np.uint16:
            raise TypeError(f"uint16_mm depth must have dtype uint16, got {raw.dtype} (already metric?)")
        return (raw.astype(np.float32) / 1000.0).astype(np.float32)
    if near is None or far is None:
        raise ValueError(f"encoding {encoding!r} requires near and far clipping planes")
    if encoding == "rgb24_linear_buffer":
        if raw.ndim != 3 or raw.shape[-1] != 3:
            raise ValueError("rgb24_linear_buffer expects an (H, W, 3) image")
        packed = raw[..., 0].astype(np.float64) * 65536 + raw[..., 1].astype(np.float64) * 256 + raw[..., 2]
        buf = packed / rgb_scale
        return (near + buf * (far - near)).astype(np.float32)
    buf = raw.astype(np.float64)
    if encoding == "linear_buffer_01":
        return (near + buf * (far - near)).astype(np.float32)
    # opengl_zbuffer_01: z_ndc = 2b-1;  z = 2nf / (f + n - z_ndc (f - n))
    z_ndc = 2.0 * buf - 1.0
    return (2.0 * near * far / (far + near - z_ndc * (far - near))).astype(np.float32)


def valid_depth_mask(depth: np.ndarray, near: float | None = None, far: float | None = None,
                     far_margin: float = 1e-3) -> np.ndarray:
    """Finite, positive depths strictly inside the clip range (far-plane hits = background)."""
    d = np.asarray(depth)
    m = np.isfinite(d) & (d > 0)
    if near is not None:
        m &= d >= near
    if far is not None:
        m &= d < far * (1.0 - far_margin)
    return m


def pixel_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    return u, v


def backproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Planar depth (H, W) -> (N, 3) points in the camera frame for masked pixels."""
    depth = np.asarray(depth, dtype=np.float64)
    H, W = depth.shape
    u, v = pixel_grid(W, H)
    if mask is None:
        mask = valid_depth_mask(depth)
    z = depth[mask]
    x = (u[mask] - K[0, 2]) / K[0, 0] * z
    y = (v[mask] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=-1)


def backproject_to_world(depth: np.ndarray, K: np.ndarray, T_WC: np.ndarray,
                         mask: np.ndarray | None = None) -> np.ndarray:
    return apply_T(T_WC, backproject(depth, K, mask))


def project(points_C: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points -> pixel coords (N, 2) and z (N,)."""
    P = np.asarray(points_C, dtype=np.float64).reshape(-1, 3)
    z = P[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * P[:, 0] / z + K[0, 2]
        v = K[1, 1] * P[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1), z


def project_world(points_W: np.ndarray, K: np.ndarray, T_WC: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return project(apply_T(invert_T(T_WC), points_W), K)


def render_point_depth(points_W: np.ndarray, K: np.ndarray, T_WC: np.ndarray, width: int,
                       height: int) -> np.ndarray:
    """Z-buffer a point set into a depth image (inf = empty). Used by tests and
    visibility checks; not a substitute for simulator rendering."""
    uv, z = project_world(points_W, K, T_WC)
    depth = np.full((height, width), np.inf)
    ok = z > 0
    ui = np.round(uv[ok, 0]).astype(int)
    vi = np.round(uv[ok, 1]).astype(int)
    zi = z[ok]
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    np.minimum.at(depth, (vi[inside], ui[inside]), zi[inside])
    return depth


# ---------------------------------------------------------------- image ops
# Every geometric image op returns the adjusted intrinsics so RGB, depth,
# masks and K are always transformed together.

def flip_images(images: dict, K: np.ndarray, horizontal: bool = False, vertical: bool = False):
    """Flip every array in ``images`` (H, W, ...) and return (images, K')."""
    K = np.asarray(K, dtype=np.float64).copy()
    any_img = next(iter(images.values()))
    H, W = any_img.shape[:2]
    out = {}
    for k, im in images.items():
        if im.shape[:2] != (H, W):
            raise ValueError(f"image {k!r} has size {im.shape[:2]}, expected {(H, W)}")
        if horizontal:
            im = im[:, ::-1]
        if vertical:
            im = im[::-1]
        out[k] = np.ascontiguousarray(im)
    if horizontal:  # u' = W-1-u
        K[0, 0], K[0, 2] = -K[0, 0], (W - 1) - K[0, 2]
    if vertical:    # v' = H-1-v
        K[1, 1], K[1, 2] = -K[1, 1], (H - 1) - K[1, 2]
    return out, K


def _resize_nearest(im: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    H, W = im.shape[:2]
    rows = np.clip(np.floor((np.arange(new_h) + 0.5) * H / new_h).astype(int), 0, H - 1)
    cols = np.clip(np.floor((np.arange(new_w) + 0.5) * W / new_w).astype(int), 0, W - 1)
    return im[rows][:, cols]


def _resize_bilinear(im: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    H, W = im.shape[:2]
    y = np.clip((np.arange(new_h) + 0.5) * H / new_h - 0.5, 0, H - 1)
    x = np.clip((np.arange(new_w) + 0.5) * W / new_w - 0.5, 0, W - 1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    wy, wx = (y - y0)[:, None], (x - x0)[None, :]
    f = im.astype(np.float64)
    if f.ndim == 3:
        wy, wx = wy[..., None], wx[..., None]
    top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
    bot = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
    out = top * (1 - wy) + bot * wy
    if np.issubdtype(im.dtype, np.integer):
        out = np.clip(np.round(out), np.iinfo(im.dtype).min, np.iinfo(im.dtype).max)
    return out.astype(im.dtype)


def resize_images(rgb: np.ndarray | None, depth: np.ndarray | None, mask: np.ndarray | None,
                  K: np.ndarray, new_size: tuple[int, int]):
    """Resize RGB (bilinear) and depth/mask (nearest, no depth blending across
    edges) to ``new_size=(H', W')``; returns (rgb, depth, mask, K')."""
    new_h, new_w = new_size
    ref = next(a for a in (rgb, depth, mask) if a is not None)
    H, W = ref.shape[:2]
    sx, sy = new_w / W, new_h / H
    K2 = np.asarray(K, dtype=np.float64).copy()
    # Pixel-centre convention: u' + 0.5 = (u + 0.5) * s
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (K2[0, 2] + 0.5) * sx - 0.5
    K2[1, 2] = (K2[1, 2] + 0.5) * sy - 0.5
    r = _resize_bilinear(rgb, new_h, new_w) if rgb is not None else None
    d = _resize_nearest(depth, new_h, new_w) if depth is not None else None
    m = _resize_nearest(mask, new_h, new_w) if mask is not None else None
    return r, d, m, K2


def normalize_negative_focal(K: np.ndarray, T_WC: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rewrite a camera whose K has BOTH focal lengths negative (e.g. PyRep's
    ``VisionSensor.get_intrinsic_matrix``) as an equivalent positive-focal
    OpenCV camera: negating fx and fy equals rotating the camera frame by pi
    about its optical axis, ``T_WC' = T_WC @ Rz(pi)``. A single negative focal
    is a reflection and is rejected (keep the flipped K instead)."""
    K = np.asarray(K, dtype=np.float64)
    if K[0, 0] > 0 and K[1, 1] > 0:
        return K.copy(), np.asarray(T_WC, dtype=np.float64).copy()
    if not (K[0, 0] < 0 and K[1, 1] < 0):
        raise ValueError("exactly one negative focal length: image is mirrored, cannot normalise to a rotation")
    K2 = K.copy()
    K2[0, 0], K2[1, 1] = -K[0, 0], -K[1, 1]
    Rz_pi = np.diag([-1.0, -1.0, 1.0, 1.0])
    return K2, np.asarray(T_WC, dtype=np.float64) @ Rz_pi


def crop_intrinsics(K: np.ndarray, top: int, left: int) -> np.ndarray:
    K2 = np.asarray(K, dtype=np.float64).copy()
    K2[0, 2] -= left
    K2[1, 2] -= top
    return K2
