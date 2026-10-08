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
