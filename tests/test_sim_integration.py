"""OPT-IN simulator integration tests (``pytest --run-sim -m sim``). Never part of default runs.

Each test is bounded (one reset, a handful of steps or one short preparation).
Environment variables select data where needed:
  ENTRY_VLA_DUOBENCH_PARQUET  one duobench_raw parquet file (for restore tests)
  ENTRY_VLA_DUOBENCH_TASK     task id (default carry_pot)
  ENTRY_VLA_PERACT2_TASK      task id (default bimanual_handover_item_easy)
"""

import os

import numpy as np
import pytest

from entry_vla.sim.contracts import ARMS, BimanualTcpCommand
from entry_vla.utils.config import load_config

REPO = os.path.join(os.path.dirname(__file__), "..")
pytestmark = pytest.mark.sim


def _backend(name):
    from entry_vla.sim.registry import make_backend
    cfg = load_config(os.path.join(REPO, "configs/smoke.yaml"), [f"backend={name}"])["backend"]
    if name == "duobench":
        cfg["task"] = os.environ.get("ENTRY_VLA_DUOBENCH_TASK", cfg["task"])
    else:
        cfg["task"] = os.environ.get("ENTRY_VLA_PERACT2_TASK", cfg["task"])
    be = make_backend(cfg)
    be.make(cfg["task"])
    return be


@pytest.mark.parametrize("name", ["duobench", "peract2"])
def test_task_predicates_available(name):
    be = _backend(name)
    try:
        obs = be.reset(seed=0)
        pred = be.task_predicates(obs)
        assert isinstance(pred, dict) and pred
    finally:
        be.close()


@pytest.mark.parametrize("name", ["duobench", "peract2"])
def test_one_step_alignment_diagnostic(name):
    from entry_vla.data.alignment import live_one_step_diagnostic
    be = _backend(name)
    try:
        rep = live_one_step_diagnostic(be, seed=0)
        assert rep["steps_executed"] == 2
    finally:
        be.close()


def test_duobench_short_preparation():
    """Physically executes a 3 cm lift with the jointspace planner's waypoints through RCS IK."""
    from entry_vla.geometry.franka import PandaArm
    be = _backend("duobench")
    try:
        obs = be.reset(seed=0)
        goal = {a: obs.arms[a].T_WE.copy() for a in ARMS}
        for a in ARMS:
            goal[a][2, 3] += 0.03
        steps = 0
        for _ in range(30):
            obs = be.step(BimanualTcpCommand(goal, {a: 1.0 for a in ARMS}), obs).obs
            steps += 1
            if all(np.linalg.norm(obs.arms[a].T_WE[:3, 3] - goal[a][:3, 3]) < 0.01 for a in ARMS):
                break
        assert all(np.linalg.norm(obs.arms[a].T_WE[:3, 3] - goal[a][:3, 3]) < 0.01 for a in ARMS), steps
    finally:
        be.close()


def test_duobench_attached_state_restoration():
    """Restore a recorded mid-grasp sim_state and compare grasp flags + TCP pose with the recording."""
    path = os.environ.get("ENTRY_VLA_DUOBENCH_PARQUET")
    if not path or not os.path.exists(path):
        pytest.skip("set ENTRY_VLA_DUOBENCH_PARQUET to one duobench_raw parquet file")
    from entry_vla.data import duobench_raw as D
    rows = D.read_episode_rows(path, 0, max_frames=None)
    grasped = [i for i, r in enumerate(rows) if D.get(r, "info.left.is_grasped") and D.get(r, "info.right.is_grasped")]
    if not grasped:
        pytest.skip("episode has no bimanual grasp frame")
    r = rows[grasped[len(grasped) // 2]]
    be = _backend("duobench")
    try:
        be.reset(seed=0)
        hold = {a: {"tquat": list(D.get(r, f"obs.{a}.tquat")), "gripper": list(D.get(r, f"env_action.{a}.gripper"))}
                for a in ARMS}
        obs = be.restore_state_snapshot({"sim_state": np.asarray(D.get(r, "info.left.sim_state")),
                                         "sim_state_schema": D.get(r, "info.left.sim_state_schema")}, hold_action=hold)
        for a in ARMS:
            tq = np.asarray(D.get(r, f"obs.{a}.tquat"))
            assert np.linalg.norm((be.T_W_shared @ np.eye(4))[:3, :3] @ tq[:3] + be.T_W_shared[:3, 3]
                                  - obs.arms[a].T_WE[:3, 3]) < 0.01
            assert obs.oracle.get(f"{a}.is_grasped") is True, "attachment/grasp state not reproduced by restore"
    finally:
        be.close()
