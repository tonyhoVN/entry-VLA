import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def pytest_addoption(parser):
    parser.addoption("--run-sim", action="store_true", default=False,
                     help="run opt-in simulator integration tests (requires installed backend + assets)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-sim"):
        return
    skip = pytest.mark.skip(reason="simulator integration test: pass --run-sim to enable")
    for item in items:
        if "sim" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
