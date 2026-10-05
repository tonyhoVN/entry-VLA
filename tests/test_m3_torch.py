"""Torch model tests: tiny randomly initialised CPU forward passes only (no optimizer steps)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_forward_check_tiny_cpu():
    from entry_vla.models.train_entry import forward_check
    r = forward_check()
    assert r["pos_shape"] == [2, 4, 2, 3] and r["rot_shape"] == [2, 4, 2, 3, 3]
    assert r["max_orthonormality_err"] < 1e-5 and r["min_det"] > 0.999
    assert r["optimizer_steps"] == 0 and np.isfinite(r["loss"])


def test_min_over_k_assigns_nearest_paired_candidate():
    from entry_vla.models.entry_proposal import LossConfig, min_over_k_loss
    B, K = 1, 3
    pos = torch.zeros(B, K, 2, 3)
    pos[0, 1] = torch.tensor([[0.5, 0, 0], [0.5, 0, 0]])
    pos[0, 2] = torch.tensor([[0.5, 0, 0], [-0.5, 0, 0]])     # one arm right, other wrong -> paired cost higher
    rot = torch.eye(3).expand(B, K, 2, 3, 3).clone()
    pred = {"pos": pos, "rot": rot, "grip_logit": torch.zeros(B, K, 2), "role_logits": torch.zeros(B, K, 2, 2),
            "cand_logits": torch.zeros(B, K)}
    tgt = {"pos": torch.tensor([[[0.5, 0, 0], [0.5, 0, 0]]]), "rot": torch.eye(3).expand(B, 2, 3, 3).clone(),
           "grip_open": torch.ones(B, 2), "role": torch.zeros(B, 2, dtype=torch.long)}
    L = min_over_k_loss(pred, tgt, LossConfig(wta_epsilon=0.0))
    assert int(L["winner"][0]) == 1 and float(L["min_pos_err_m"]) < 1e-6


def test_geodesic_and_decode_region_frame():
    from entry_vla.models.entry_proposal import decode_candidates, geodesic_t, rot6d_to_matrix_t
    from entry_vla.geometry.rotations import rot_z
    R = torch.tensor(rot_z(0.3), dtype=torch.float32)
    assert abs(float(geodesic_t(torch.eye(3), R)) - 0.3) < 1e-4
    r6 = torch.randn(5, 6)
    M = rot6d_to_matrix_t(r6)
    assert torch.allclose(torch.linalg.det(M), torch.ones(5), atol=1e-5)
    pred = {"pos": torch.zeros(1, 1, 2, 3), "rot": torch.eye(3).expand(1, 1, 2, 3, 3), "grip_logit": torch.zeros(1, 1, 2),
            "role_logits": torch.zeros(1, 1, 2, 2), "cand_logits": torch.zeros(1, 1)}
    T_FW = torch.eye(4)[None].clone(); T_FW[0, :3, 3] = torch.tensor([0.5, 0.0, 0.1])
    c = decode_candidates(pred, T_FW)[0][0]
    assert np.allclose(c["T_WE"]["left"][:3, 3], [0.5, 0, 0.1])


def test_scorer_shapes_and_loss():
    from entry_vla.models.scorer import CAND_DIM, ArrivalVerifier, OutcomeScorer, ScorerConfig, scorer_loss
    cfg = ScorerConfig(point_feat_dim=8, hidden=16, text_dim=4, state_dim=34)
    net = OutcomeScorer(cfg)
    B = 3
    b = {"points": torch.randn(B, 10, 6), "points_mask": torch.ones(B, 10, dtype=torch.bool), "text": torch.zeros(B, 4),
         "state": torch.zeros(B, 34), "history": torch.zeros(B, 4), "candidate": torch.zeros(B, CAND_DIM),
         "success": torch.tensor([1.0, 0.0, 1.0]), "arm_success": torch.tensor([[1.0, 1.0], [float("nan"), 0.0], [1.0, 1.0]]),
         "duration_s": torch.tensor([3.0, 0.0, 4.0]), "mask": torch.tensor([1.0, 1.0, 0.0])}
    with torch.no_grad():
        out = net(b)
        L = scorer_loss(out, b)
    assert out["joint_logit"].shape == (B,) and out["arm_logits"].shape == (B, 2) and torch.isfinite(L["loss"])
    av = ArrivalVerifier(cfg)
    with torch.no_grad():
        v = av({"arrival_points": torch.randn(B, 10, 6), "arrival_points_mask": torch.ones(B, 10, dtype=torch.bool),
                "text": torch.zeros(B, 4), "arrival_state": torch.zeros(B, 34), "history": torch.zeros(B, 4)})
    assert v.shape == (B,)
