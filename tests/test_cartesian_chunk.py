import numpy as np
import pytest

from entry_vla.geometry.rotations import random_rotation, so3_exp
from entry_vla.geometry.transforms import make_T
from entry_vla.policies import cartesian_chunk as CC


def _traj(rng, n, start=None):
    T = start if start is not None else make_T(random_rotation(rng), rng.normal(size=3) * 0.3)
    out = []
    for _ in range(n):
        T = T @ make_T(so3_exp(rng.normal(size=3) * 0.05), rng.normal(size=3) * 0.01)
        out.append(T)
    return np.stack(out)


def test_state_roundtrip(rng):
    Tl, Tr = _traj(rng, 1)[0], _traj(rng, 1)[0]
    s = CC.encode_state(Tl, Tr, 0.04, 0.01)
    assert s.shape == (CC.STATE_DIM,) and s.dtype == np.float32
    Tl2, Tr2, wl, wr = CC.decode_state(s)
    assert np.allclose(Tl2, Tl, atol=1e-5) and np.allclose(Tr2, Tr, atol=1e-5)
    assert np.isclose(wl, 0.04) and np.isclose(wr, 0.01)


def test_chunk_roundtrip_and_first_step_anchoring(rng):
    H = 8
    Tq = make_T(random_rotation(rng), [0.4, 0.1, 0.3])
    targets = _traj(rng, H, start=Tq)
    grip = np.linspace(0, 1, H)
    ch = CC.encode_arm_chunk(Tq, targets, grip)
    assert ch.shape == (H, 7)
    # first step is relative to the measured query pose, in its body frame
    exp0 = Tq[:3, :3].T @ (targets[0, :3, 3] - Tq[:3, 3])
    assert np.allclose(ch[0, :3], exp0)
    dec, g = CC.decode_arm_chunk(Tq, ch)
    assert np.allclose(dec, targets, atol=1e-9)
    assert np.allclose(g, grip)


def test_chunk_is_sequential_not_anchor_relative(rng):
    Tq = np.eye(4)
    t1 = make_T(p=[0.1, 0, 0])
    t2 = make_T(p=[0.2, 0, 0])
    ch = CC.encode_arm_chunk(Tq, np.stack([t1, t2]), np.zeros(2))
    assert np.allclose(ch[1, :3], [0.1, 0, 0])  # increment from previous desired waypoint


def test_chunk_body_frame_rotation():
    Tq = make_T(so3_exp([0, 0, np.pi / 2]))
    t1 = make_T(so3_exp([0, 0, np.pi / 2]), [0.0, 0.1, 0.0])   # world +y = body +x
    ch = CC.encode_arm_chunk(Tq, t1[None], np.zeros(1))
    assert np.allclose(ch[0, :3], [0.1, 0, 0], atol=1e-12)


def test_decoupled_increments_differ_from_se3_log():
    # rotation + translation in one step: decoupled delta_p is NOT the se(3) translational twist
    Tq = np.eye(4)
    t1 = make_T(so3_exp([0, 0, 1.0]), [0.1, 0.0, 0.0])
    ch = CC.encode_arm_chunk(Tq, t1[None], np.zeros(1))
    assert np.allclose(ch[0, :3], [0.1, 0, 0])


def test_bimanual_layout(rng):
    H = 4
    Tl, Tr = np.eye(4), make_T(p=[0, -0.5, 0])
    tl, tr = _traj(rng, H, Tl), _traj(rng, H, Tr)
    ch = CC.encode_bimanual_chunk(Tl, Tr, tl, tr, np.ones(H), np.zeros(H))
    assert ch.shape == (H, CC.ACTION_DIM)
    assert np.allclose(ch[:, 6], 1) and np.allclose(ch[:, 13], 0)
    dl, dr, _, _ = CC.decode_bimanual_chunk(Tl, Tr, ch)
    assert np.allclose(dl, tl, atol=1e-5) and np.allclose(dr, tr, atol=1e-5)


def test_build_training_chunks_padding_and_shift(rng):
    T = 6
    H = 4
    Tl = _traj(rng, T)
    Tr = _traj(rng, T)
    g = np.zeros(T)
    spec = CC.ChunkTargetSpec("measured_future", H, control_assumption="pose[t+1+k] ~ target[t+k]")
    ch, mask = CC.build_training_chunks(Tl, Tr, Tl, Tr, g, g, spec)
    assert ch.shape == (T, H, 14) and mask.shape == (T, H)
    assert mask[0].all()
    assert mask[T - 2].tolist() == [True, False, False, False]
    assert not mask[T - 1].any()
    # padded steps are zero-motion (repeat last target)
    assert np.allclose(ch[T - 2, 1:, :6], 0, atol=1e-6)
    # decoding valid part of chunk t reproduces measured future poses
    dl, _, _, _ = CC.decode_bimanual_chunk(Tl[0], Tr[0], ch[0])
    assert np.allclose(dl, Tl[1:1 + H], atol=1e-5)


def test_target_spec_requires_assumption():
    with pytest.raises(ValueError):
        CC.ChunkTargetSpec("measured_future", 4)
    with pytest.raises(ValueError):
        CC.ChunkTargetSpec("keyframes", 4)
    CC.ChunkTargetSpec("commanded", 4)
