"""RLBench manipulation-phase domain for X-VLA (2toinf/X-VLA), registered at runtime so the X-VLA
checkout stays unmodified. Import after putting the X-VLA repo root first on sys.path.

Domain "rlbench_manip": single Franka arm, absolute EE6D in the RLBench world frame:
  left  = [xyz (3), rot6d (6, X-VLA layout R[:, :2] row-major), gripper (1 = closed)], right = zeros.
Chunks: X-VLA samples num_actions + 1 points over ``qdur`` seconds (meta, default 1 s) of the 20 Hz
measured trajectory; point 0 is the proprio (current pose), points 1.. are the targets.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

import numpy as np

DOMAIN = "rlbench_manip"
DEFAULT_DOMAIN_ID = 11   # robomind-franka (single Franka arm, pretraining domain of X-VLA-Pt)


def register(xvla_root: str, domain_id: int = DEFAULT_DOMAIN_ID, weight: float = 1.0):
    xvla_root = os.path.abspath(xvla_root)
    if sys.path[0] != xvla_root:
        sys.path.insert(0, xvla_root)
    import datasets  # noqa: F401  (X-VLA's package, must shadow HF `datasets`)
    if not os.path.abspath(datasets.__file__).startswith(xvla_root):
        raise RuntimeError(f"`datasets` resolved to {datasets.__file__}, not X-VLA's; import order problem")
    from datasets import domain_config
    from datasets.domain_handler import registry
    from datasets.domain_handler.base import BaseHDF5Handler
    from datasets.utils import quat_to_rotate6d

    class RLBenchManipHandler(BaseHDF5Handler):
        dataset_name = DOMAIN

        def build_left_right(self, f):
            freq, qdur = float(self.meta.get("freq", 20.0)), float(self.meta.get("qdur", 1.0))
            pos, quat = f["ee_pos"][()], f["ee_quat_xyzw"][()]
            grip_closed = 1.0 - f["gripper_open"][()][:, None]
            left = np.concatenate([pos, quat_to_rotate6d(quat), grip_closed], -1).astype(np.float32)
            return left, np.zeros_like(left), None, None, freq, qdur

        def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
            return range(0, max(0, T_left - 1))

    registry._REGISTRY[DOMAIN] = RLBenchManipHandler
    domain_config.DATA_DOMAIN_ID[DOMAIN] = domain_id
    domain_config.DATA_WEIGHTS[DOMAIN] = weight
    return RLBenchManipHandler
