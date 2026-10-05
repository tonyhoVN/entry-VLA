import numpy as np
import pytest

from entry_vla.geometry import camera as C
from entry_vla.geometry.rotations import rot_z
from entry_vla.geometry.transforms import apply_T, invert_T, look_at, make_T


def _plane_scene(W=64, H=48):
    intr = C.CameraIntrinsics.from_fovy(np.deg2rad(60), W, H)
    T_WC = look_at([0.0, -0.6, 0.8], [0.0, 0.0, 0.0])
    # analytic depth of the z=0 plane for every pixel
    u, v = C.pixel_grid(W, H)
    rays_C = np.stack([(u - intr.K[0, 2]) / intr.K[0, 0], (v - intr.K[1, 2]) / intr.K[1, 1], np.ones_like(u)], -1)
    rays_W = rays_C @ T_WC[:3, :3].T
    o = T_WC[:3, 3]
    s = -o[2] / rays_W[..., 2]          # ray param where it hits z=0, z-depth = s (since ray_C z = 1)
    return intr, T_WC, s.astype(np.float32)


def test_backproject_reproject_known_plane():
    intr, T_WC, depth = _plane_scene()
    pts = C.backproject_to_world(depth, intr.K, T_WC)
    assert np.allclose(pts[:, 2], 0.0, atol=1e-5)          # all points lie on the table plane
    uv, z = C.project_world(pts, intr.K, T_WC)
    u, v = C.pixel_grid(intr.width, intr.height)
    assert np.allclose(uv[:, 0], u.ravel(), atol=1e-3)
    assert np.allclose(uv[:, 1], v.ravel(), atol=1e-3)


def test_depth_unit_conversions():
    mm = np.array([[1000, 250]], dtype=np.uint16)
    assert np.allclose(C.decode_depth(mm, "uint16_mm"), [[1.0, 0.25]])
    with pytest.raises(TypeError):
        C.decode_depth(mm.astype(np.float32), "uint16_mm")   # already metric -> refuse /1000
    with pytest.raises(TypeError):
        C.decode_depth(mm, "metric_m")
    lin = np.array([[0.0, 0.5, 1.0]])
    assert np.allclose(C.decode_depth(lin, "linear_buffer_01", near=0.1, far=2.1), [[0.1, 1.1, 2.1]])
    with pytest.raises(ValueError):
        C.decode_depth(lin, "linear_buffer_01")


def test_rlbench_rgb24_depth_decoding():
    near, far = 0.01, 4.5
    buf = np.array([[0.0, 0.25, 0.999]])
    ints = np.floor(buf * C.RLBENCH_DEPTH_SCALE + 0.5).astype(np.uint32)
    rgb = np.stack([(ints >> 16) & 255, (ints >> 8) & 255, ints & 255], -1).astype(np.uint8)
    d = C.decode_depth(rgb, "rgb24_linear_buffer", near=near, far=far)
    assert np.allclose(d, near + buf * (far - near), atol=1e-5)


def test_opengl_zbuffer_linearisation():
    near, far = 0.05, 5.0
    z_true = np.array([0.1, 1.0, 4.0])
    z_ndc = (far + near - 2 * near * far / z_true) / (far - near)
    buf = (z_ndc + 1) / 2
    assert np.allclose(C.decode_depth(buf, "opengl_zbuffer_01", near=near, far=far), z_true, rtol=1e-5)


def test_resize_keeps_backprojection_consistent():
    intr, T_WC, depth = _plane_scene(64, 48)
    rgb = np.zeros((48, 64, 3), np.uint8)
    _, d2, _, K2 = C.resize_images(rgb, depth, None, intr.K, (24, 32))
    pts = C.backproject_to_world(d2, K2, T_WC)
    assert np.allclose(pts[:, 2], 0.0, atol=2e-2)   # nearest-sampled depth, half-res


def test_flip_consistency_with_negative_focal():
    intr, T_WC, depth = _plane_scene()
    imgs, K2 = C.flip_images({"depth": depth, "mask": depth > 0}, intr.K, horizontal=True, vertical=True)
    pts = C.backproject_to_world(imgs["depth"], K2, T_WC)
    assert np.allclose(pts[:, 2], 0.0, atol=1e-5)
    assert K2[0, 0] < 0 and K2[1, 1] < 0


def test_normalize_negative_focal_matches():
    intr, T_WC, depth = _plane_scene()
    imgs, K2 = C.flip_images({"depth": depth}, intr.K, horizontal=True, vertical=True)
    K3, T3 = C.normalize_negative_focal(K2, T_WC)
    assert K3[0, 0] > 0 and K3[1, 1] > 0
    a = C.backproject_to_world(imgs["depth"], K2, T_WC)
    b = C.backproject_to_world(imgs["depth"], K3, T3)
    assert np.allclose(a, b, atol=1e-6)
    with pytest.raises(ValueError):
        C.normalize_negative_focal(np.diag([-1.0, 1.0, 1.0]), T_WC)


def test_valid_mask_far_plane():
    d = np.array([0.0, 0.5, 2.0, np.nan, 1.9999])
    m = C.valid_depth_mask(d, near=0.1, far=2.0)
    assert m.tolist() == [False, True, False, False, False]
