import numpy as np

from entry_vla.perception.distance import (distance_to_points, first_within, object_points, radius_sweep,
                                           remove_outliers)
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


def test_remove_outliers_drops_leak():
    pts = np.vstack([np.random.default_rng(0).normal(0, 0.01, (50, 3)), [[1.0, 1.0, 1.0]]])
    assert len(remove_outliers(pts, 0.15)) == 50


def test_parse_qwen3_relative_boxes():
    text = '```json\n[{"bbox_2d": [500, 250, 750, 1000], "label": "button"}]\n```'
    (det,) = parse_boxes(text, width=256, height=128)
    assert det.label == "button"
    assert np.allclose(det.bbox_xyxy, [128, 32, 192, 127])  # y2 clipped to H-1
    assert parse_boxes("no box here", 10, 10) == []
