"""Offline adapter tests with FAKE native payloads shaped like the pinned
schemas. These are mock-only: they do not prove simulator compatibility."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from entry_vla.geometry.camera import CameraIntrinsics
from entry_vla.geometry.rotations import matrix_to_quat, rot_x, rot_z
from entry_vla.geometry.transforms import invert_T, look_at, make_T
from entry_vla.sim.contracts import ARMS, BimanualTcpCommand
from entry_vla.sim.duobench import DuoBenchBackend, resolve_task_id
from entry_vla.sim.mock import MockBackend
from entry_vla.sim.peract2 import (camera_from_misc, decode_native_action, encode_native_action,
                                   class_name_for)
from entry_vla.sim.validate import check_observation, tcp_projection_report

REPO = os.path.join(os.path.dirname(__file__), "..")


def test_duobench_task_resolution():
    assert resolve_task_id("Carry-Pot") == "carry_pot"
    assert resolve_task_id("duobench/bin_sort") == "bin_sort"
    with pytest.raises(ValueError):
        resolve_task_id("CarryPot")


def _fake_rcs_obs(T_shared_E, T_W_C, H=12, W=16, depth_mm=800):
    intr = CameraIntrinsics.from_fovy(np.deg2rad(60), W, H)
    K34 = np.hstack([intr.K, np.zeros((3, 1))])
    arm = {}
    for a in ARMS:
        p = T_shared_E[a][:3, 3]
        q = matrix_to_quat(T_shared_E[a][:3, :3], "xyzw")
        arm[a] = {"tquat": np.concatenate([p, q]), "joints": np.zeros(7), "gripper": np.array([0.5], np.float32)}
    frame = {"rgb": {"data": np.zeros((H, W, 3), np.uint8), "intrinsics": K34, "extrinsics": invert_T(T_W_C)},
             "depth": {"data": np.full((H, W, 1), depth_mm, np.uint16), "intrinsics": K34,
                       "extrinsics": invert_T(T_W_C)}}
    return {**arm, "frames": {"head": frame}}


def test_duobench_normalisation_frames_and_depth():
    be = DuoBenchBackend({"name": "duobench"})
    be.T_W_shared = make_T(rot_z(0.3), [0.1, 0.0, 0.8])     # what make() derives from cfg poses
    T_shared_E = {"left": make_T(rot_x(np.pi), [0.4, 0.2, 0.1]), "right": make_T(rot_x(np.pi), [0.4, -0.2, 0.1])}
    T_W_C = look_at([1.5, 0.0, 1.5], [0.4, 0.0, 0.8])
    obs = be.normalize_observation(_fake_rcs_obs(T_shared_E, T_W_C), {"instruction": "carry", "success": 0.0,
                                                                       "left": {"is_grasped": False}})
    assert np.allclose(obs.arms["left"].T_WE, be.T_W_shared @ T_shared_E["left"])
    cam = obs.cameras["head"]
    assert np.allclose(cam.T_WC, T_W_C)
    assert cam.depth.dtype == np.float32 and np.allclose(cam.depth, 0.8)
    assert np.isclose(obs.arms["left"].gripper_width, 0.5 * 0.085)
    assert "left.is_grasped" in obs.oracle and "success" in obs.oracle
    assert check_observation(obs, require_depth=True) == []
    # command conversion goes back to the shared base frame, xyzw
    cmd = BimanualTcpCommand(T_WE={a: obs.arms[a].T_WE for a in ARMS}, gripper={"left": 1.0, "right": 0.0})
    act = be.convert_command(cmd, obs)
    assert np.allclose(act["left"]["tquat"][:3], T_shared_E["left"][:3, 3])
    assert act["right"]["gripper"] == [0.0]


def test_duobench_relative_mode_rejects_absolute_command():
    be = DuoBenchBackend({"name": "duobench", "relative_to": "LAST_STEP"})
    cmd = BimanualTcpCommand(T_WE={a: np.eye(4) for a in ARMS}, gripper={a: 1.0 for a in ARMS})
    from entry_vla.sim.contracts import UnsupportedCapabilityError
    with pytest.raises(UnsupportedCapabilityError):
        be.convert_command(cmd, None)


def test_peract2_native_action_layout_right_first():
    TL = make_T(rot_x(np.pi), [0.3, 0.3, 1.0])
    TR = make_T(rot_x(np.pi), [0.3, -0.3, 1.1])
    a = encode_native_action(BimanualTcpCommand({"left": TL, "right": TR}, {"left": 1.0, "right": 0.0}),
                             {"left": 0.0, "right": 1.0})
    assert a.shape == (18,)
    assert np.allclose(a[:3], TR[:3, 3]) and np.allclose(a[9:12], TL[:3, 3])
    assert np.isclose(np.linalg.norm(a[3:7]), 1.0)
    assert a[7] == 0.0 and a[8] == 1.0 and a[16] == 1.0
    dec = decode_native_action(a)
    assert np.allclose(dec["left"]["T_WE"], TL) and not dec["right"]["gripper_open"]
    assert class_name_for("bimanual_handover_item_easy") == "BimanualHandoverItemEasy"


def _rlbench_pointcloud(depth, extrinsics, intrinsics):
    """Re-implementation of rlbench VisionSensor.pointcloud_from_depth_and_camera_params
    semantics (pixel (u,v) * depth through inv(K [R^T | -R^T C])) for comparison."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pc = np.stack([u * depth, v * depth, depth], -1).reshape(-1, 3)
    cam = (np.linalg.inv(intrinsics) @ pc.T).T
    return cam @ extrinsics[:3, :3].T + extrinsics[:3, 3]


def test_peract2_camera_negative_focal_matches_rlbench_pointcloud():
    H = W = 8
    f = -W / (2 * np.tan(np.deg2rad(30)))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])    # PyRep-style
    T_WC = look_at([1.0, 0.0, 1.5], [0.0, 0.0, 0.7])
    misc = {"front_camera_intrinsics": K, "front_camera_extrinsics": T_WC,
            "front_camera_near": 0.01, "front_camera_far": 4.5}
    depth01 = np.full((H, W), 0.25, np.float32)
    cam = camera_from_misc("front", misc, None, depth01, "linear_buffer_01")
    assert cam.K[0, 0] > 0
    from entry_vla.geometry.camera import backproject_to_world
    ours = backproject_to_world(cam.depth, cam.K, cam.T_WC, mask=np.ones((H, W), bool))
    ref = _rlbench_pointcloud(cam.depth.astype(np.float64), T_WC, K)
    assert np.allclose(ours, ref, atol=1e-6)


def test_mock_backend_contract_snapshot_and_tracking():
    be = MockBackend({"name": "mock"})
    be.make("mock_bimanual_reach")
    obs = be.reset(seed=1)
    assert check_observation(obs, require_depth=True) == []
    rep = tcp_projection_report(obs)
    assert rep["front"]["left"]["in_image"]
    snap = be.get_state_snapshot()
    target = {a: MockBackend.downward_pose(obs.oracle["handles"][a] + [0, 0, 0.1]) for a in ARMS}
    cmd = BimanualTcpCommand(target, {a: 1.0 for a in ARMS})
    for _ in range(40):
        obs = be.step(cmd, obs).obs
    for a in ARMS:
        assert np.linalg.norm(obs.arms[a].T_WE[:3, 3] - target[a][:3, 3]) < 5e-3
    obs2 = be.restore_state_snapshot(snap)
    assert obs2.step == 0
    obs3 = MockBackend({"name": "mock"}); obs3.make("mock_bimanual_reach")
    assert np.allclose(obs3.reset(seed=1).oracle["bar_center"], obs2.oracle["bar_center"])


def test_smoke_cli_on_mock(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    out = subprocess.run([sys.executable, "-m", "entry_vla.cli.main", "smoke", "--config",
                          os.path.join(REPO, "configs", "smoke.yaml"), "backend=mock", "--steps", "3",
                          "--out", str(tmp_path)], capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    rep = json.loads(out.stdout)
    assert rep["hold_steps"] == 3 and rep["reset_contract_violations"] == []
    assert (tmp_path / "last_frame.npz").exists()


def test_smoke_cli_rejects_more_than_five_steps():
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    out = subprocess.run([sys.executable, "-m", "entry_vla.cli.main", "smoke", "--config",
                          os.path.join(REPO, "configs", "smoke.yaml"), "backend=mock", "--steps", "6"],
                         capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode != 0


@pytest.mark.sim
def test_duobench_live_smoke():
    """Opt-in: requires the duobench env with assets present (1 reset + 5 steps)."""
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    out = subprocess.run([sys.executable, "-m", "entry_vla.cli.main", "smoke", "--config",
                          os.path.join(REPO, "configs", "smoke.yaml"), "backend=duobench"],
                         capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr


@pytest.mark.sim
def test_peract2_live_smoke():
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    out = subprocess.run([sys.executable, "-m", "entry_vla.cli.main", "smoke", "--config",
                          os.path.join(REPO, "configs", "smoke.yaml"), "backend=peract2"],
                         capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr
