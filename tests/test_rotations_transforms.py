import numpy as np
import pytest

from entry_vla.geometry import rotations as R
from entry_vla.geometry import transforms as T


def test_quat_orders_roundtrip(rng):
    for _ in range(20):
        M = R.random_rotation(rng)
        for order in R.QUAT_ORDERS:
            q = R.matrix_to_quat(M, order)
            assert np.allclose(R.quat_to_matrix(q, order), M, atol=1e-9)
    # identity in both conventions
    assert np.allclose(R.matrix_to_quat(np.eye(3), "xyzw"), [0, 0, 0, 1])
    assert np.allclose(R.matrix_to_quat(np.eye(3), "wxyz"), [1, 0, 0, 0])


def test_quat_order_mismatch_detected():
    q_xyzw = np.array([0.0, 0.0, np.sin(0.5), np.cos(0.5)])  # 1 rad about z
    assert np.allclose(R.quat_to_matrix(q_xyzw, "xyzw"), R.rot_z(1.0))
    assert not np.allclose(R.quat_to_matrix(q_xyzw, "wxyz"), R.rot_z(1.0))
    with pytest.raises(ValueError):
        R.quat_to_matrix(q_xyzw, "zyxw")


def test_so3_exp_log_roundtrip_incl_near_pi(rng):
    for theta in [0.0, 1e-9, 1e-4, 0.5, 2.0, np.pi - 1e-3, np.pi - 1e-6]:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rv = axis * theta
        M = R.so3_exp(rv)
        assert R.is_rotation(M)
        assert np.allclose(R.so3_exp(R.so3_log(M)), M, atol=1e-6)


def test_rot6d_roundtrip_and_projection(rng):
    M = np.stack([R.random_rotation(rng) for _ in range(10)])
    assert np.allclose(R.rot6d_to_matrix(R.matrix_to_rot6d(M)), M, atol=1e-9)
    noisy = R.matrix_to_rot6d(M) + 0.1 * rng.normal(size=(10, 6))
    for m in R.rot6d_to_matrix(noisy):
        assert R.is_rotation(m)
    assert R.is_rotation(R.project_to_so3(M[0] + 0.05 * rng.normal(size=(3, 3))))


def test_transform_convention_and_inverse(rng):
    # T_AB maps B coords to A: p_A = T_AB p_B
    T_AB = T.make_T(R.rot_z(np.pi / 2), [1.0, 0, 0])
    p_B = np.array([1.0, 0, 0])
    assert np.allclose(T.apply_T(T_AB, p_B), [1.0, 1.0, 0])
    for _ in range(5):
        X = T.make_T(R.random_rotation(rng), rng.normal(size=3))
        assert np.allclose(T.invert_T(X) @ X, np.eye(4), atol=1e-12)
        assert T.is_valid_T(X)


def test_tcp_from_camera_pose(rng):
    T_WE = T.make_T(R.random_rotation(rng), rng.normal(size=3))
    T_EC = T.make_T(R.random_rotation(rng), [0.05, 0.0, 0.03])
    T_WC = T_WE @ T_EC
    assert np.allclose(T.tcp_from_camera_pose(T_WC, T_EC), T_WE, atol=1e-12)


def test_look_at_points_z_at_target():
    T_WC = T.look_at([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    z = T_WC[:3, 2]
    assert np.allclose(z, -np.ones(3) / np.sqrt(3))
    assert R.is_rotation(T_WC[:3, :3])
    # straight-down view still well defined
    assert R.is_rotation(T.look_at([0, 0, 1.0], [0, 0, 0])[:3, :3])


def test_pose_quat_boundary():
    T1 = T.pose_to_T([1, 2, 3], [0, 0, 0, 1], "xyzw")
    T2 = T.pose_to_T([1, 2, 3], [1, 0, 0, 0], "wxyz")
    assert np.allclose(T1, T2)
    p, q = T.T_to_pose(T1, "xyzw")
    assert np.allclose(q, [0, 0, 0, 1])
