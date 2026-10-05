import numpy as np
import pytest

from entry_vla.data import alignment as A
from entry_vla.data import splits as S
from entry_vla.sim.mock import MockBackend


def test_target_lag_recovers_shift():
    t = np.linspace(0, 1, 50)
    cmd = np.stack([np.sin(3 * t), t, 0 * t], 1)
    meas = np.concatenate([np.repeat(cmd[:1], 2, 0), cmd[:-2]])       # measurement lags by 2
    assert A.estimate_target_lag(meas, cmd)["best_lag"] == 2


def test_live_one_step_on_mock():
    be = MockBackend({"name": "mock"})
    be.make("mock_bimanual_reach")
    rep = A.live_one_step_diagnostic(be, seed=0)
    assert rep["verdict"] == "obs_after_action" and rep["steps_executed"] == 2


def test_uniform_resample_masks_gaps():
    t = np.array([0.0, 0.1, 0.2, 0.6, 0.7])
    idx, valid = A.uniform_resample_indices(t, 10.0)
    assert len(idx) == 8
    assert valid.tolist() == [True, True, True, False, False, False, True, True]


def test_pad_stack_mask():
    out, m = A.pad_stack([np.ones((2, 3)), np.ones((4, 3))])
    assert out.shape == (2, 4, 3) and m.sum() == 6 and not m[0, 2]


def test_policy_aligned_targets():
    ld = {"a": np.arange(4)}
    a, m = A.policy_aligned_targets(ld, "obs_after_action", "a")
    assert a.tolist() == [1, 2, 3, 3] and m.tolist() == [True, True, True, False]
    with pytest.raises(ValueError):
        A.policy_aligned_targets(ld, "unverified", "a")


def _metas():
    metas = []
    for demo in range(30):
        for child in range(3):        # descendants of one source demo (re-render, probes)
            metas.append({"episode_id": f"d{demo}_c{child}", "task_id": "t" if demo < 25 else "held",
                          "source": {"source_demo_id": f"demo{demo}", "scene_seed": demo}})
    return metas


def test_group_split_no_leakage_and_heldout():
    rep = S.assign_splits(_metas(), heldout={"task_id": ["held"]})
    by_group = {}
    for ep, sp in rep["assignment"].items():
        by_group.setdefault(rep["groups"][ep], set()).add(sp)
    assert all(len(v) == 1 for v in by_group.values())
    assert {rep["assignment"][f"d{d}_c0"] for d in range(25, 30)} == {"heldout"}
    assert set(rep["assignment"].values()) >= {"train", "heldout"}
    assert S.assign_splits(_metas(), heldout={"task_id": ["held"]})["assignment"] == rep["assignment"]


def test_leakage_detector():
    assert S.check_leakage({"a": "train", "b": "test"}, {"a": "g", "b": "g"})


def test_normalizer_train_only():
    x = np.random.default_rng(0).normal(size=(100, 3))
    n = S.Normalizer.fit(x, "train")
    assert np.allclose(n.inverse(n(x)), x)
    with pytest.raises(ValueError):
        S.Normalizer.fit(x, "test")
