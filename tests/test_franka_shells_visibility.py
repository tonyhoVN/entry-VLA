import numpy as np

from entry_vla.geometry import franka as F
from entry_vla.geometry.camera import CameraIntrinsics, render_point_depth
from entry_vla.geometry.rotations import rot_x, rot_z
from entry_vla.geometry.shells import FilterConfig, ShellSpec, filter_pair, sample_shell_tcp_poses
from entry_vla.geometry.transforms import invert_T, look_at, make_T
from entry_vla.geometry.visibility import target_visibility


def test_panda_ready_pose_reasonable():
    arm = F.PandaArm()
    T = arm.fk(F.PANDA_Q_READY)
    # ready pose: TCP in front of base, pointing down
    assert 0.25 < T[0, 3] < 0.45 and abs(T[1, 3]) < 1e-6 and 0.3 < T[2, 3] < 0.6
    assert T[2, 2] < -0.99


def test_ik_recovers_fk_pose():
    arm = F.PandaArm(T_WB=make_T(rot_z(0.3), [0.1, 0.2, 0.0]))
    q_true = F.PANDA_Q_READY + np.array([0.2, 0.1, -0.1, 0.2, 0.1, -0.1, 0.2])
    target = arm.fk(q_true)
    q, ok, info = arm.ik(target, q_init=F.PANDA_Q_READY)
    assert ok, info
    assert np.allclose(arm.fk(q)[:3, 3], target[:3, 3], atol=1e-3)


def test_tilted_mount_fk_consistent():
    T_mount = make_T(rot_x(np.pi / 4), [0, 0.3, 0.5])
    arm = F.arm_from_mount(T_mount)
    q = F.PANDA_Q_READY
    assert np.allclose(F.base_from_world(arm, arm.fk(q)), F.PandaArm().fk(q), atol=1e-9)


def test_dual_arm_clearance_sign():
    left = F.PandaArm(T_WB=make_T(p=[0, 0.5, 0]))
    right = F.PandaArm(T_WB=make_T(p=[0, -0.5, 0]))
    far = F.dual_arm_clearance(left, F.PANDA_Q_READY, right, F.PANDA_Q_READY)
    close_right = F.PandaArm(T_WB=make_T(p=[0, 0.45, 0]))  # 5 cm apart
    near = F.dual_arm_clearance(left, F.PANDA_Q_READY, close_right, F.PANDA_Q_READY)
    assert far > 0.5 and near < 0


def test_shell_samples_point_at_center(rng):
    spec = ShellSpec(center=[0.5, 0.0, 0.1], radius=0.2, radius_jitter=0.0)
    Ts = sample_shell_tcp_poses(spec, 16, rng)
    for T in Ts:
        d = spec.center - T[:3, 3]
        assert np.isclose(np.linalg.norm(d), 0.2)
        assert np.allclose(T[:3, 2], d / np.linalg.norm(d))
    # camera-aimed variant: camera z points at centre
    T_EC = make_T(rot_z(0.1), [0.05, 0.0, -0.03])
    for T in sample_shell_tcp_poses(spec, 4, rng, T_EC=T_EC):
        T_WC = T @ T_EC
        d = spec.center - T_WC[:3, 3]
        assert np.allclose(T_WC[:3, 2], d / np.linalg.norm(d), atol=1e-9)


def test_filter_pair_rejects_unreachable_and_accepts_fk_pose():
    left = F.PandaArm(T_WB=make_T(p=[0, 0.4, 0]))
    right = F.PandaArm(T_WB=make_T(p=[0, -0.4, 0]))
    cfg = FilterConfig(table_z=0.0)
    q0 = F.PANDA_Q_READY
    res = filter_pair(left.fk(q0), right.fk(q0), left, right, q0, q0, cfg)
    assert res["accepted"], res["reasons"]
    far = make_T(p=[3.0, 0, 0.3]) @ make_T(rot_x(np.pi))
    res = filter_pair(far, right.fk(q0), left, right, q0, q0, cfg)
    assert not res["accepted"] and "left:ik_failed" in res["reasons"]


def test_visibility_occlusion():
    intr = CameraIntrinsics.from_fovy(np.deg2rad(60), 64, 64)
    T_WC = look_at([0, 0, 1.0], [0, 0, 0], up=(0, 1, 0))
    target = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    v = target_visibility(target, intr.K, T_WC, 64, 64)
    assert v.visible_fraction == 1.0
    occluder = np.array([[0.0, 0.0, 0.5]])
    depth = render_point_depth(np.vstack([target, occluder]), intr.K, T_WC, 64, 64)
    v2 = target_visibility(target, intr.K, T_WC, 64, 64, depth_image=depth)
    assert v2.visible_fraction == 0.5
    behind = target_visibility(np.array([[0, 0, 2.0]]), intr.K, T_WC, 64, 64)
    assert behind.visible_fraction == 0.0
