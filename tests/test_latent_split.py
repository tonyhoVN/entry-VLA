import numpy as np

from entry_vla.perception.latent_split import (novelty_curve, novelty_peaks, optimal_segmentation,
                                               segment_cost_table, standardise)


def _piecewise(lengths, dim=8, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    parts = [rng.normal(size=dim) + noise * rng.normal(size=(n, dim)) for n in lengths]
    return np.concatenate(parts)


def test_cost_table_matches_direct():
    X = np.random.default_rng(1).normal(size=(12, 4))
    C = segment_cost_table(X)
    seg = X[3:9]
    assert np.isclose(C[3, 9], ((seg - seg.mean(0)) ** 2).sum())


def test_optimal_segmentation_recovers_boundaries():
    X = standardise(_piecewise([30, 20, 25]))
    assert optimal_segmentation(X, 2) in ([30], [50])
    assert optimal_segmentation(X, 3) == [30, 50]
    assert optimal_segmentation(X[:4], 3) == []  # too short for min_len


def test_novelty_peak_at_change():
    X = _piecewise([40, 40])
    nov = novelty_curve(X, window=5)
    assert novelty_peaks(nov, 1) == [40]
    assert nov[:5].sum() == 0


def test_depth_grid_and_balanced_blocks():
    from entry_vla.perception.latent_split import build_trajectory, depth_grid
    d = np.full((32, 32), 1.0)
    d[:16] = 0.5
    valid = np.ones_like(d, bool)
    valid[:, -1] = False
    g = depth_grid(d, valid, lo=0.5, hi=1.5, grid=2)
    assert g.shape == (4,) and np.allclose(g[:2], [0.0, 1 / 16]) and np.allclose(g[2:], [0.5, 0.5 + 0.5 / 16])
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(50, 384)), rng.normal(size=(50, 16))
    X = build_trajectory([a, b], balance=True)
    assert X.shape == (50, 400)
    assert np.isclose((X[:, :384] ** 2).sum(), (X[:, 384:] ** 2).sum())  # equal total variance per block


def test_cross_frame_attention_tightens_segments():
    from entry_vla.perception.latent_split import cross_frame_attention
    X = _piecewise([30, 30], dim=16, noise=0.4, seed=3)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = cross_frame_attention(X, tau=0.1, lam=10.0, alpha=0.5)
    assert Y.shape == X.shape and np.allclose(np.linalg.norm(Y, axis=1), 1.0)
    spread = lambda Z: sum(((Z[s] - Z[s].mean(0)) ** 2).sum() for s in (slice(0, 30), slice(30, 60)))
    assert spread(Y) < spread(X)
    assert optimal_segmentation(standardise(Y), 2) == [30]
