import numpy as np
import pytest

torch = pytest.importorskip("torch")

from entry_vla.data.rlbench_entry import seed_split  # noqa: E402
from entry_vla.models.single_entry import (RZ_PI, SingleEntryConfig, SingleEntryNet, SingleLossConfig,  # noqa: E402
                                           entry_metrics, rot_error, single_entry_loss)


def test_rot_error_gripper_symmetry():
    R = torch.eye(3)[None]
    flipped = (torch.eye(3) @ RZ_PI)[None]
    assert rot_error(R, flipped, symmetric=True).item() < 1e-2
    assert abs(rot_error(R, flipped, symmetric=False).item() - np.pi) < 1e-2


def test_forward_loss_metrics_shapes():
    torch.manual_seed(0)
    net = SingleEntryNet(SingleEntryConfig(num_candidates=3))
    B, N = 4, 64
    pred = net(torch.rand(B, N, 6), torch.ones(B, N, dtype=torch.bool), torch.rand(B, 64), torch.rand(B, 17))
    assert pred["pos"].shape == (B, 3, 3) and pred["rot"].shape == (B, 3, 3, 3)
    assert torch.allclose(pred["rot"] @ pred["rot"].transpose(-1, -2), torch.eye(3).expand(B, 3, 3, 3), atol=1e-4)
    loss, win = single_entry_loss(pred, torch.rand(B, 3), torch.eye(3).expand(B, 3, 3), torch.ones(B), SingleLossConfig())
    loss.backward()
    assert win.shape == (B,) and torch.isfinite(loss)
    m = entry_metrics(pred, torch.rand(B, 3), torch.eye(3).expand(B, 3, 3), torch.ones(B))
    assert all(v.shape == (B,) for v in m.values())
    assert (m["minK_pos_m"] <= m["top1_pos_m"] + 1e-6).all()


def test_seed_split():
    assert seed_split("seed3", [15, 16], [18]) == "train"
    assert seed_split("seed16", [15, 16], [18]) == "val"
    assert seed_split("seed18", [15, 16], [18]) == "test"
