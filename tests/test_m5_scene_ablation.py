import glob
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

from entry_vla.geometry.franka import PANDA_Q_READY
from entry_vla.sim.custom_scene import check_scene_compat, load_spec
from entry_vla.sim.contracts import ARMS
from entry_vla.sim.mock import MockBackend
from entry_vla.sim.preparation import JointSpacePlanner, PrepConfig, execute_preparation
from entry_vla.policies.base import CommandArbiter
from entry_vla.utils.config import load_config

REPO = os.path.join(os.path.dirname(__file__), "..")


def test_tilted_spec_geometry():
    spec = load_spec(yaml.safe_load(open(os.path.join(REPO, "configs/scene/tilted_dual_franka.yaml"))))
    m = spec.mounts()
    # bases tilted by 45 deg: base z-axis makes 45 deg with world z
    for a in ARMS:
        assert np.isclose(np.degrees(np.arccos(m[a][2, 2])), 45.0)
    g = spec.gravity_in_base()
    assert np.allclose(np.linalg.norm(g["left"]), 9.81) and not np.allclose(g["left"], [0, 0, -9.81])
    arms = spec.arms()
    T = arms["left"].fk(PANDA_Q_READY)
    q, ok, _ = arms["left"].ik(T, q_init=PANDA_Q_READY + 0.1)
    assert ok
    assert [c.name for c in spec.cameras] == ["ego", "exo", "left_wrist", "right_wrist"]
    assert set(spec.camera_intrinsics()) == {"ego", "exo", "left_wrist", "right_wrist"}
    with pytest.raises(ValueError):
        load_spec({"scene_id": "duobench_stock_fr3duo_robotiq"})


def test_scene_compat_guard():
    meta = {"episode_id": "e", "backend": "duobench"}
    check_scene_compat(meta, "duobench_stock_fr3duo_robotiq")
    with pytest.raises(ValueError):
        check_scene_compat(meta, "tilted_dual_franka_v1")
    check_scene_compat(meta, "tilted_dual_franka_v1",
                       {"duobench_stock_fr3duo_robotiq->tilted_dual_franka_v1": {"validated": True, "report": "x"}})


def test_sequential_arrival_order_moves_one_arm_at_a_time():
    be = MockBackend({"name": "mock"}); be.make("mock_bimanual_reach")
    obs = be.reset(seed=2)
    goal = {a: MockBackend.downward_pose(obs.oracle["handles"][a] + [0, 0, 0.12]) for a in ARMS}
    cfg = PrepConfig(control_dt=be.dt, arrival_order="left_first")
    plan = JointSpacePlanner(be.arms, cfg).plan({a: obs.arms[a].q for a in ARMS}, goal)
    assert plan.ok and plan.reason == "sequential:left_first"
    qr = plan.q_path["right"]
    n_still = int(np.sum(np.all(np.isclose(qr, obs.arms["right"].q), axis=1)))
    assert n_still >= 1                     # right arm holds while the left arm travels
    ho = execute_preparation(be, obs, goal, JointSpacePlanner(be.arms, cfg), CommandArbiter(), cfg)
    assert ho.arrived and ho.route == "sequential:left_first"


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(REPO, "configs/ablation/*.yaml"))))
def test_ablation_conditions_apply(path):
    spec = yaml.safe_load(open(path))
    base = os.path.join(REPO, spec["base"])
    for name in spec["conditions"]:
        out = subprocess.run([sys.executable, os.path.join(REPO, "scripts/ablation_args.py"), path, name],
                             capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, out.stderr
        ov = out.stdout.split()
        cfg = load_config(base, ov)
        for o in ov:
            k = o.split("=", 1)[0]
            node = cfg
            for p in k.split("."):
                assert p in node, (name, k)
                node = node[p]
