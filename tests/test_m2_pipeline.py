"""M2 pipeline tests on the MOCK backend (mock-only coverage; no simulator claims)."""
import numpy as np
import pytest

from entry_vla.data.segmentation import AnnotateConfig, apply_overrides, arm_criteria, propose_paired_start
from entry_vla.data.synthetic import record_mock_episode
from entry_vla.evaluation.metrics import summarize_trials, wilson_interval
from entry_vla.evaluation.selectors import (DistanceShellSelector, FixedCanonicalSelector, NearestDemoSelector,
                                            RegionEstimator, demo_library_from_annotations)
from entry_vla.evaluation.trials import TrialConfig, run_trial
from entry_vla.geometry.rotations import rot_x
from entry_vla.geometry.shells import FilterConfig
from entry_vla.geometry.transforms import make_T
from entry_vla.policies.base import CommandArbiter, MockOraclePolicy
from entry_vla.sim.contracts import ARMS
from entry_vla.sim.mock import MockBackend
from entry_vla.sim.preparation import JointSpacePlanner, PrepConfig, diagnostic_teleport, execute_preparation

CANON = {"left": make_T(rot_x(np.pi), [0.0, 0.15, 0.10]), "right": make_T(rot_x(np.pi), [0.0, -0.15, 0.10])}
REGION = {"left": "handles.left", "right": "handles.right"}


@pytest.fixture
def be():
    b = MockBackend({"name": "mock"})
    b.make("mock_bimanual_reach")
    yield b
    b.close()


def _prep_cfg(b):
    return PrepConfig(control_dt=b.dt, max_joint_speed=1.0, table_z=0.0)


def test_preparation_physically_arrives_and_stops(be):
    obs = be.reset(seed=2)
    goal = {a: MockBackend.downward_pose(obs.oracle["handles"][a] + [0, 0, 0.12]) for a in ARMS}
    arb = CommandArbiter()
    ho = execute_preparation(be, obs, goal, JointSpacePlanner(be.arms, _prep_cfg(be)), arb, _prep_cfg(be))
    assert ho.arrived and ho.stopped and ho.failure is None, ho.as_dict()
    assert max(ho.arrival_pos_err.values()) < 0.01
    assert all(v is not None and v < 0.02 for v in ho.arrival_joint_speed.values())
    assert arb.owner == {"left": None, "right": None}       # released for the policy


def test_preparation_reports_planning_failure(be):
    obs = be.reset(seed=2)
    goal = {a: MockBackend.downward_pose(np.array([3.0, 0, 0.3])) for a in ARMS}
    ho = execute_preparation(be, obs, goal, JointSpacePlanner(be.arms, _prep_cfg(be)), CommandArbiter(), _prep_cfg(be))
    assert ho.failure.startswith("planning:")


def test_command_arbiter_single_owner():
    arb = CommandArbiter()
    arb.acquire("planner")
    with pytest.raises(RuntimeError):
        arb.acquire("policy")
    arb.release("planner")
    arb.acquire("policy")
    with pytest.raises(RuntimeError):
        arb.check("planner")


def test_teleport_is_labelled_and_refuses_attached(be):
    obs = be.reset(seed=0)
    o2 = diagnostic_teleport(be, {a: obs.arms[a].q + 0.05 for a in ARMS})
    assert not np.allclose(o2.arms["left"].q, obs.arms["left"].q)
    be.state["attached"] = True
    with pytest.raises(Exception):
        diagnostic_teleport(be, {a: obs.arms[a].q for a in ARMS})


def _run(be, selector, seed, **kw):
    return run_trial(be, MockOraclePolicy(), selector, RegionEstimator("oracle", REGION), be.arms, seed,
                     TrialConfig(max_policy_steps=60, n_action_steps=4, **kw), _prep_cfg(be), FilterConfig(table_z=0.0))


def test_trial_canonical_entry_succeeds_with_distinct_records(be):
    rec = _run(be, FixedCanonicalSelector(CANON), seed=1)
    assert {"pre_travel", "decision", "arrival", "outcome", "label_version"} <= set(rec)
    assert rec["pre_travel"]["region_source"] == "oracle"
    assert rec["outcome"]["label"] == "success", rec["outcome"]
    assert rec["label_version"]["policy"]["checkpoint_id"] == "mock-v1"


def test_trial_bad_entry_is_manipulation_failure_not_dropped(be):
    # geometrically valid but outside the mock policy's approach cone -> must fail and be counted
    side = {"left": make_T(rot_x(np.pi), [0.15, 0.15, 0.06]), "right": make_T(rot_x(np.pi), [0.15, -0.15, 0.06])}
    rec = _run(be, FixedCanonicalSelector(side), seed=1)
    assert rec["outcome"]["label"] != "success", rec["outcome"]
    s = summarize_trials([rec, _run(be, FixedCanonicalSelector(CANON), seed=2)], requested=3, time_cap_s=20.0)
    assert s["requested_trials"] == 3 and s["label_counts"]["missing_record"] == 1
    assert abs(s["success_rate"] - 1 / 3) < 1e-9


def test_shell_selector_runs(be):
    rec = _run(be, DistanceShellSelector(n=8, radius=0.12, seed=0), seed=3)
    assert rec["decision"]["n_candidates"] == 8
    assert rec["outcome"]["label"] in ("success", "manipulation_timeout", "rejected_pre_travel",
                                       "transit_planning_failure", "transit_execution_failure")


def test_summaries_refuse_pooling_backends():
    r = {"label_version": {"backend": "duobench"}, "outcome": {"label": "success", "success": True}, "seed": 0}
    r2 = {"label_version": {"backend": "peract2"}, "outcome": {"label": "success", "success": True}, "seed": 0}
    with pytest.raises(ValueError):
        summarize_trials([r, r2])
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi


def test_annotation_paired_start_and_overrides():
    ep = record_mock_episode(seed=4)
    cfg = AnnotateConfig(radius=0.20, visibility_cameras=["front"], min_lead_s=0.2)
    ann = propose_paired_start(ep, cfg)
    assert ann["status"] == "proposed", ann
    s = ann["proposed_start"]
    assert s < min(ann["onset"].values())
    for a in ARMS:
        assert ann["criteria_at_start"][a]["dist_m"] <= 0.20
    assert ann["provenance"]["uses_oracle"] is False
    rej = apply_overrides(ep.episode_id, ann, {ep.episode_id: {"reject": True, "note": "bad"}})
    assert rej["status"] == "rejected" and rej["provenance"]["source"] == "manual_override"
    man = apply_overrides(ep.episode_id, ann, {ep.episode_id: {"proposed_start": s + 1}})
    assert man["status"] == "manual" and man["proposed_start"] == s + 1
    # nearest-demo library from accepted annotation
    lib = demo_library_from_annotations([ep], {ep.episode_id: ann})
    assert len(lib) == 1 and NearestDemoSelector(lib).name == "nearest_demo"


def test_annotation_rejects_when_no_onset():
    ep = record_mock_episode(seed=5)
    for a in ARMS:
        ep.lowdim[f"{a}.gripper_width"] = np.full(ep.num_frames, 0.08)
    assert propose_paired_start(ep, AnnotateConfig())["status"] == "rejected"


def test_smolvla_frames_match_reference_chunks():
    from entry_vla.policies.cartesian_chunk import ChunkTargetSpec, build_training_chunks
    from entry_vla.policies.smolvla_adapter import encode_chunks_from_targets, episode_frames
    ep = record_mock_episode(seed=6)
    frames = list(episode_frames(ep, {"front": "camera1"}, (48, 64)))
    T, H = len(frames), 6
    st = np.stack([f["observation.state"] for f in frames])
    ta = np.stack([f["target.abs"] for f in frames])
    fi = np.stack([f["target.frame"] for f in frames])[:, 0]
    # emulate LeRobot delta_timestamps stacking with boundary clamping
    idx = np.minimum(np.arange(T)[:, None] + np.arange(H)[None], T - 1)
    ch, pad = encode_chunks_from_targets(st, ta[idx], fi[idx])
    ref, mask = build_training_chunks(ep.lowdim["left.T_WE"], ep.lowdim["right.T_WE"], ep.lowdim["left.cmd_T_WE"],
                                      ep.lowdim["right.cmd_T_WE"], ep.lowdim["left.cmd_gripper"],
                                      ep.lowdim["right.cmd_gripper"], ChunkTargetSpec("commanded", H))
    assert np.allclose(ch[~pad], ref[:T][~pad], atol=1e-4)
    assert (~pad == mask[:T]).all()
    assert frames[0]["action"].shape == (14,) and np.allclose(frames[0]["action"], ch[0, 0], atol=1e-5)
