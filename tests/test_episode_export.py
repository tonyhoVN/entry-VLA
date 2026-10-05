import numpy as np
import pytest

from entry_vla.data.episode import Episode, load_episode, save_episode, validate_episode
from entry_vla.data.synthetic import record_mock_episode


@pytest.fixture(scope="module")
def mock_ep():
    return record_mock_episode(seed=3)


def test_mock_episode_valid_and_roundtrip(tmp_path, mock_ep):
    assert validate_episode(mock_ep) == []
    d = save_episode(mock_ep, str(tmp_path))
    ep2 = load_episode(d)
    assert validate_episode(ep2) == []
    assert np.allclose(ep2.lowdim["left.T_WE"], mock_ep.lowdim["left.T_WE"])
    assert ep2.cameras["front"]["depth"].dtype == np.float32
    with pytest.raises(FileExistsError):
        save_episode(mock_ep, str(tmp_path))        # exported episodes are immutable by default


def test_validation_catches_missing_and_leaks(mock_ep):
    bad = Episode(meta=dict(mock_ep.meta), lowdim=dict(mock_ep.lowdim), cameras={}, oracle={})
    del bad.lowdim["left.dq"]
    bad.lowdim["oracle.stage"] = np.zeros(mock_ep.num_frames)
    probs = validate_episode(bad)
    assert any("left.dq" in p for p in probs) and any("oracle" in p for p in probs)
    bad.mark_unavailable("left.dq", "not recorded")
    del bad.lowdim["oracle.stage"]
    assert validate_episode(bad) == []


def test_validation_requires_per_frame_moving_camera(mock_ep):
    cams = {k: dict(v) for k, v in mock_ep.cameras.items()}
    cams["left_wrist"]["T_WC"] = cams["left_wrist"]["T_WC"][0]
    ep = Episode(meta=mock_ep.meta, lowdim=mock_ep.lowdim, cameras=cams)
    assert any("moving camera" in p for p in validate_episode(ep))


def test_peract2_fake_dir_stub_unpickle_and_export(tmp_path):
    pytest.importorskip("PIL")
    from entry_vla.data import peract2_raw as P
    ep_dir = P.write_fake_episode_dir(str(tmp_path / "bimanual_handover_item_easy"), T=3, size=8)
    rep = P.inspect_episode_dir(ep_dir)
    assert rep["num_frames"] == 3 and "front" in rep["cameras"]
    ep = P.export_episode_dir(ep_dir, "bimanual_handover_item_easy", {"cameras": ["front", "wrist_left"]})
    assert validate_episode(ep) == []
    assert np.allclose(ep.cameras["front"]["depth"], 0.01 + 0.25 * (4.5 - 0.01), atol=1e-4)
    assert ep.meta["cameras"]["wrist_left"]["moving"] and not ep.meta["cameras"]["front"]["moving"]
    assert ep.cameras["front"]["K"][0, 0] > 0
    assert np.isclose(ep.lowdim["left.gripper_width"][0], 0.08)
    assert "not human teleoperation" in ep.meta["provenance"]
    # gripper_pose quaternion xyzw (1,0,0,0) is pi about x -> TCP z points down
    assert np.allclose(ep.lowdim["left.T_WE"][0][:3, 2], [0, 0, -1])
    import importlib.util
    import sys
    if importlib.util.find_spec("rlbench") is None:
        assert "rlbench" not in sys.modules       # fake modules were removed after pickling


def test_stub_unpickler_refuses_unknown_globals(tmp_path):
    import os
    import pickle
    from entry_vla.data.peract2_raw import load_pickle
    p = os.path.join(tmp_path, "evil.pkl")
    with open(p, "wb") as f:
        pickle.dump(os.getcwd, f)            # references posix.getcwd
    with pytest.raises(pickle.UnpicklingError):
        load_pickle(p)


def _write_fake_duobench_parquet(path, T=6):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("PIL")
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.full((12, 16, 3), 90, np.uint8)).save(buf, format="JPEG")
    from entry_vla.geometry.camera import CameraIntrinsics
    from entry_vla.geometry.transforms import invert_T, look_at
    K = np.hstack([CameraIntrinsics.from_fovy(1.0, 16, 12).K, np.zeros((3, 1))]).ravel().tolist()
    E = invert_T(look_at([1.0, 0, 1.0], [0.4, 0, 0.0])).ravel().tolist()
    rows = []
    for k in range(T):
        arm = lambda y: {"tquat": [0.4, y, 0.2 + 0.01 * k, 1.0, 0.0, 0.0, 0.0], "joints": [0.1 * k] * 7,
                         "xyzrpy": [0.0] * 6, "gripper": [0.5]}
        env_a = lambda: {"tquat": [0.0, 0.0, 0.01 * k, 0, 0, 0, 1], "gripper": [0.5]}
        rows.append({
            "obs": {"left": arm(0.2), "right": arm(-0.2),
                    "frames": {"head": {"rgb": {"data": buf.getvalue(), "intrinsics": K, "extrinsics": E}}}},
            "info": {"left": {"absolute_action": [0.4, 0.2, 0.2, 1, 0, 0, 0], "gripper_width": 0.5, "is_grasped": False,
                              "sim_state": [0.0] * 4},
                     "right": {"absolute_action": [0.4, -0.2, 0.2, 1, 0, 0, 0], "gripper_width": 0.5, "is_grasped": False},
                     "obj_on_stove": False},
            "env_action": {"left": env_a(), "right": env_a()},
            "action": {"left": {"tquat": [0.0, 0.0, 0.01 * (k - 1), 0, 0, 0, 1], "gripper": [0.5]},
                       "right": {"tquat": [0.0, 0.0, 0.01 * (k - 1), 0, 0, 0, 1], "gripper": [0.5]}},
            "uuid": "abcdef0123456789", "step": k, "timestamp": 100.0 + k / 30.0, "instruction": "carry the pot",
            "success": False,
        })
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=4)


def test_duobench_raw_export_from_fake_parquet(tmp_path):
    from entry_vla.data import duobench_raw as D
    p = str(tmp_path / "fake.parquet")
    _write_fake_duobench_parquet(p)
    rows = D.read_episode_rows(p, 0, max_frames=10)
    assert len(rows) == 6
    rep = D.alignment_report(rows)
    assert np.isclose(rep["dt_mean_s"], 1 / 30)
    assert all(rep["action_equals_prev_env_action"])
    with pytest.raises(ValueError):
        D.export_rows(rows, "carry_pot", p, 0, {"cameras": ["head"]})
    ep = D.export_rows(rows, "carry_pot", p, 0, {"cameras": ["head"], "T_world_shared": np.eye(4).tolist()})
    assert validate_episode(ep) == []
    assert ep.meta["alignment"]["obs_vs_action"] == "obs_after_action"
    assert "cam.head.depth" in ep.meta["unavailable"]
    assert np.isclose(ep.meta["recording_rate_hz"], 30.0)
