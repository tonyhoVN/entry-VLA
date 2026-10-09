import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from contact_phase_split_vlabench import interior, static_cameras  # noqa: E402


def test_interior_drops_silhouette_edges():
    seg = np.zeros((6, 6), int)
    seg[2:5, 2:5] = 7
    ok = interior(seg)
    assert ok[3, 3] and not ok[2, 2] and not ok[1, 3] and ok[0, 0]


def test_static_cameras_excludes_moving():
    T = np.tile(np.eye(4), (5, 3, 1, 1))
    T[:, 2, 0, 3] = np.linspace(0, 0.5, 5)   # wrist camera moves
    assert static_cameras(T) == [0, 1]
