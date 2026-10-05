import os

import pytest

from entry_vla.sim.registry import backend_class
from entry_vla.utils.config import load_config

ROOT = os.path.join(os.path.dirname(__file__), "..", "configs")


def test_group_selection_and_dotted_override():
    cfg = load_config(os.path.join(ROOT, "smoke.yaml"), ["backend=peract2", "backend.image_size=[64,64]", "seed=3"])
    assert cfg["backend"]["name"] == "peract2"
    assert cfg["backend"]["image_size"] == [64, 64]
    assert cfg["seed"] == 3
    cfg = load_config(os.path.join(ROOT, "smoke.yaml"))
    assert cfg["backend"]["name"] == "duobench"


def test_unknown_backend_is_an_error_not_a_fallback():
    with pytest.raises(ValueError):
        backend_class("rlbench_upstream")


def test_capabilities_declared():
    for name in ("duobench", "peract2", "mock"):
        caps = backend_class(name).capabilities()
        assert caps.name == name
        assert caps.quaternion_order in ("xyzw", "wxyz")
    from entry_vla.sim.contracts import UnsupportedCapabilityError
    with pytest.raises(UnsupportedCapabilityError):
        backend_class("peract2").capabilities().require("state_snapshot")
