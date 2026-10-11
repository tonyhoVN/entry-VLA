import pytest

torch = pytest.importorskip("torch")

from entry_vla.models.entry_diffusion import EntryDiffusion, EntryDiffusionConfig, cosine_alpha_bar  # noqa: E402
from entry_vla.models.pointnet2 import PointNet2Encoder, ball_query, farthest_point_sample  # noqa: E402


def test_fps_and_ball_query():
    torch.manual_seed(0)
    xyz = torch.rand(2, 100, 3)
    idx = farthest_point_sample(xyz, 10)
    assert idx.shape == (2, 10) and all(len(set(r.tolist())) == 10 for r in idx)
    nb = ball_query(xyz, xyz[:, :5], 0.2, 8)
    d = (xyz[:, None].expand(-1, 5, -1, -1).gather(2, nb[..., None].expand(-1, -1, -1, 3)) - xyz[:, :5, None]).norm(dim=-1)
    assert (d <= 0.2 + 1e-6).all()


def test_pointnet2_shape():
    enc = PointNet2Encoder(in_feat=3, out_dim=32, n1=32, n2=8, k=8)
    assert enc(torch.rand(3, 128, 6)).shape == (3, 32)


def batch(B=2, V=3, D=16):
    return {"obj_points": torch.rand(B, 128, 6) * 0.1, "centroid": torch.rand(B, 3), "text_feat": torch.rand(B, D),
            "crop_feat": torch.rand(B, V, D), "crop_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
            "tgt_pos_C": torch.rand(B, 3) * 0.1, "tgt_rot": torch.eye(3).expand(B, 3, 3)}


def test_diffusion_loss_and_sample():
    torch.manual_seed(0)
    m = EntryDiffusion(EntryDiffusionConfig(vlm_dim=16, cond_dim=32, point_dim=32, hidden=64, n_blocks=2, n_steps=20))
    loss = m.loss(batch())
    loss.backward()
    assert torch.isfinite(loss)
    m.eval()
    p, R = m.sample(batch(), n_samples=4, ddim_steps=5, guidance=2.0)
    assert p.shape == (2, 4, 3) and R.shape == (2, 4, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(2, 4, 3, 3), atol=1e-4)


def test_pose_roundtrip_and_schedule():
    m = EntryDiffusion(EntryDiffusionConfig(vlm_dim=16, cond_dim=32, point_dim=32, hidden=64, n_blocks=1))
    R = torch.linalg.qr(torch.randn(5, 3, 3))[0]
    R = R * torch.sign(torch.det(R))[:, None, None]
    p = torch.randn(5, 3) * 0.1
    p2, R2 = m.decode_pose(m.encode_pose(p, R))
    assert torch.allclose(p, p2, atol=1e-5) and torch.allclose(R, R2, atol=1e-4)
    ab = cosine_alpha_bar(100)
    assert (ab[1:] <= ab[:-1]).all() and ab[0] < 1 and ab[-1] > 0


def test_rot6d_robust_parallel_halves():
    from entry_vla.models.entry_diffusion import rot6d_to_matrix_robust
    R = rot6d_to_matrix_robust(torch.tensor([[1.5, 1.5, 1.5, 1.5, 1.5, 1.5], [1.0, 0, 0, 0, 1.0, 0]]))
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(2, 3, 3), atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(2), atol=1e-5)
    assert torch.allclose(R[1], torch.eye(3))
