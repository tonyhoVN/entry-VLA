import numpy as np
import pytest

from entry_vla.perception.distance import (distance_to_points, first_within, fuse_voxel, object_points,
                                           radius_sweep)
from entry_vla.perception.grounding import parse_boxes


def test_object_points_backprojects_masked_plane():
    K = np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]])
    depth = np.full((64, 64), 2.0)
    mask = np.zeros((64, 64), bool)
    mask[30:35, 30:35] = True
    pts = object_points(depth, np.ones_like(mask), mask, K, np.eye(4), erode=1)
    assert len(pts) == 9  # 5x5 eroded by one pixel
    assert np.allclose(pts[:, 2], 2.0)


def test_distance_and_radius_rule():
    cloud = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    q = np.array([[0.0, 0.0, 0.5], [0.1, 0.0, 0.2], [0.1, 0.0, 0.05]])
    d = distance_to_points(q, cloud, chunk=1)
    assert np.allclose(d, [0.5, 0.2, 0.05])
    assert first_within(d, 0.2) == 1
    assert first_within(d, 0.2, before=1) is None
    assert radius_sweep(d, [0.1, 0.6]) == {"0.10": 2, "0.60": 0}
    assert np.isinf(distance_to_points(q, np.zeros((0, 3)))).all()


def test_fuse_voxel_keeps_separate_objects():
    pytest.importorskip("open3d")
    rng = np.random.default_rng(0)
    obj_a = rng.uniform(0, 0.02, (400, 3))                      # big object, dense
    obj_b = np.array([1.0, 0.0, 0.0]) + rng.uniform(0, 0.01, (20, 3))  # small object far away
    fused = fuse_voxel([obj_a, obj_b], voxel_size=0.005)
    assert len(fused) < 420
    assert (np.linalg.norm(fused - [1.0, 0, 0], axis=1) < 0.02).any()  # small object kept
    assert np.allclose(fuse_voxel([obj_a], voxel_size=0), obj_a)


def test_parse_qwen3_relative_boxes():
    text = '```json\n[{"bbox_2d": [500, 250, 750, 1000], "label": "button"}]\n```'
    (det,) = parse_boxes(text, width=256, height=128)
    assert det.label == "button"
    assert np.allclose(det.bbox_xyxy, [128, 32, 192, 127])  # y2 clipped to H-1
    assert parse_boxes("no box here", 10, 10) == []
