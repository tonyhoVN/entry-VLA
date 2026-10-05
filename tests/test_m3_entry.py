import numpy as np
import pytest

from entry_vla.data.entry_dataset import EntrySampleConfig, collate, make_samples
from entry_vla.data.segmentation import AnnotateConfig, propose_paired_start
from entry_vla.data.synthetic import record_mock_episode
from entry_vla.geometry.transforms import apply_T
from entry_vla.models.text import HashingTextEncoder


@pytest.fixture(scope="module")
def annotated_ep():
    ep = record_mock_episode(seed=7)
    ep.annotations = {"skill_starts": [propose_paired_start(ep, AnnotateConfig(visibility_cameras=["front"]))]}
    assert ep.annotations["skill_starts"][0]["status"] == "proposed"
    return ep


def test_samples_are_pre_travel_and_global_only(annotated_ep):
    enc = HashingTextEncoder(16)
    s = make_samples(annotated_ep, EntrySampleConfig(n_points=128, query_stride=2), enc(["lift"])[0])
    assert s, "expected samples"
    start = s[0]["start_frame"]
    assert all(x["query_frame"] <= start - 5 for x in s)
    assert s[0]["input_provenance"]["cameras"] == "non_moving"
    b = collate(s)
    assert b["points"].shape == (len(s), 128, 6) and b["state"].shape[1] == 34
    # target in W equals the demo TCP pose at the annotated start
    assert np.allclose(b["tgt_pos"][0, 0], annotated_ep.lowdim["left.T_WE"][start][:3, 3], atol=1e-5)


def test_moving_camera_requires_opt_in(annotated_ep):
    with pytest.raises(ValueError):
        make_samples(annotated_ep, EntrySampleConfig(cameras=["left_wrist"]), np.zeros(4))


def test_region_frame_needs_geometry_source(annotated_ep):
    with pytest.raises(ValueError):
        make_samples(annotated_ep, EntrySampleConfig(pose_frame="region_point"), np.zeros(4))
    s = make_samples(annotated_ep, EntrySampleConfig(pose_frame="region_point", geometry_source="oracle:handles",
                                                     n_points=64), np.zeros(4))
    x = s[0]
    assert x["input_provenance"]["oracle_condition"] is True
    # round trip: target in region frame mapped back to W matches the demo
    T = x["T_FW"].astype(np.float64)
    p_w = apply_T(T, x["tgt_pos"][0].astype(np.float64))
    assert np.allclose(p_w, annotated_ep.lowdim["left.T_WE"][x["start_frame"]][:3, 3], atol=1e-4)
    assert np.allclose(x["geometry"][:3], [0, 0.15, 0], atol=1e-4)       # left handle in region frame


def test_text_encoder_deterministic():
    e = HashingTextEncoder(32)
    a, b = e(["Lift the bar"]), e(["lift the bar"])
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)


