"""M4 on the MOCK backend: probe labels, scorer arrays, matched comparison (mock-only coverage)."""
import numpy as np
import pytest

from entry_vla.data.entry_dataset import EntrySampleConfig
from entry_vla.evaluation.compare import check_matched, compare, paired_success_diff
from entry_vla.evaluation.probes import ProbeConfig, collect_scene
from entry_vla.evaluation.selectors import FixedCanonicalSelector, FullEpisodeSelector, RegionEstimator
from entry_vla.evaluation.trials import TrialConfig, run_trial
from entry_vla.geometry.rotations import rot_x
from entry_vla.geometry.shells import FilterConfig
from entry_vla.geometry.transforms import make_T
from entry_vla.models.train_scorer import build_arrays
from entry_vla.policies.base import MockOraclePolicy
from entry_vla.sim.mock import MockBackend
from entry_vla.sim.preparation import PrepConfig

CANON = {"left": make_T(rot_x(np.pi), [0.0, 0.15, 0.10]), "right": make_T(rot_x(np.pi), [0.0, -0.15, 0.10])}
REG = RegionEstimator("oracle", {"left": "handles.left", "right": "handles.right"})


@pytest.fixture
def be():
    b = MockBackend({"name": "mock"})
    b.make("mock_bimanual_reach")
    return b


def test_probe_records_and_scorer_arrays(be, tmp_path):
    pc = ProbeConfig(n_scenes=1, perturbations_per_scene=3, max_pos_m=0.05, max_rot_rad=0.2,
                     arm_success_keys={"left": "grasped.left", "right": "grasped.right"})
    recs = collect_scene(be, MockOraclePolicy(), FixedCanonicalSelector(CANON), REG, be.arms, 5000, pc,
                         TrialConfig(max_policy_steps=60), PrepConfig(control_dt=0.1), FilterConfig(table_z=0.0),
                         EntrySampleConfig(n_points=64), np.zeros(8, np.float32), str(tmp_path), np.random.default_rng(0))
    assert len(recs) == 4
    labels = {r["outcome"]["label"] for r in recs}
    assert labels <= {"success", "manipulation_timeout", "transit_planning_failure", "transit_execution_failure",
                      "rejected_pre_travel"}
    assert recs[0]["outcome"]["label"] == "success"          # unperturbed canonical entry
    for r in recs:
        assert r["group"] == "scene5000" and "label_version" in r
    arr = build_arrays(recs)
    n = len(arr["success"])
    assert n >= 1 and arr["candidate"].shape == (n, 34) and arr["history"].shape == (n, 4)
    assert arr["points"].shape == (n, 64, 6)


def _trials(be, sel, seeds, **kw):
    return [run_trial(be, MockOraclePolicy(), sel, REG, be.arms, s, TrialConfig(max_policy_steps=60, full_episode_steps=80, **kw),
                      PrepConfig(control_dt=0.1), FilterConfig(table_z=0.0)) for s in seeds]


def test_matched_comparison_and_full_episode_baseline(be):
    seeds = [1, 2]
    a = _trials(be, FixedCanonicalSelector(CANON), seeds)
    b = _trials(be, FullEpisodeSelector(), seeds)
    assert b[0]["arrival"]["skipped"].startswith("full_episode")
    res = compare({"canon": a, "full": b}, "canon", requested=2, time_cap_s=20.0, budgets={"demos": 0})
    assert res["summaries"]["canon"]["success_rate"] == 1.0
    assert res["paired_vs_reference"]["full"]["n"] == 2
    with pytest.raises(ValueError):
        check_matched({"canon": a, "other": _trials(be, FixedCanonicalSelector(CANON), [3, 4])})


def test_paired_diff():
    mk = lambda s, ok: {"seed": s, "outcome": {"success": ok}}
    d = paired_success_diff([mk(1, True), mk(2, True)], [mk(1, False), mk(2, True)])
    assert d["mean_diff"] == 0.5 and d["a_only_success"] == 1
