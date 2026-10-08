"""Phase splitting from latent frame representations.

Each frame (one or more camera images) is embedded with a frozen image encoder
(DINOv2 by default: CLS token, L2-normalised; cameras concatenated). The episode
then becomes a trajectory in latent space, and phase boundaries are change points
of that trajectory:

* ``optimal_segmentation`` - exact least-squares segmentation into K contiguous
  segments (dynamic programming over cumulative sums, O(K T^2)). With K=2 this is
  the single best transition -> manipulation boundary.
* ``novelty_curve`` - cosine distance between the mean embeddings of the windows
  before and after each frame (a Foote-style novelty score); peaks are candidate
  boundaries and the curve is saved for inspection.

Everything except ``FrameEncoder`` is numpy-only.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class FrameEncoder:
    def __init__(self, model_id: str = "facebook/dinov2-small", device: str = "cuda:0", batch_size: int = 32,
                 pooling: str = "cls"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.model_id, self.device, self.batch_size, self.pooling = model_id, device, batch_size, pooling
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self._torch = torch

    def encode(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """(N, H, W, 3) uint8 images -> (N, D) L2-normalised embeddings."""
        torch = self._torch
        out = []
        for i in range(0, len(images), self.batch_size):
            inputs = self.processor(images=list(images[i:i + self.batch_size]), return_tensors="pt").to(self.device)
            with torch.inference_mode():
                h = self.model(**inputs).last_hidden_state
            z = h[:, 0] if self.pooling == "cls" else h[:, 1:].mean(1)
            out.append(torch.nn.functional.normalize(z.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def segment_cost_table(X: np.ndarray) -> np.ndarray:
    """cost[i, j] = sum of squared deviations from the mean for frames i..j-1 (j > i)."""
    X = np.asarray(X, dtype=np.float64)
    T = len(X)
    c1 = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
    c2 = np.concatenate([[0.0], np.cumsum((X * X).sum(1))])
    cost = np.full((T + 1, T + 1), np.inf)
    for i in range(T):
        n = np.arange(1, T - i + 1)
        s1 = c1[i + 1:] - c1[i]
        cost[i, i + 1:] = (c2[i + 1:] - c2[i]) - (s1 * s1).sum(1) / n
    return cost


def optimal_segmentation(X: np.ndarray, k: int, min_len: int = 3) -> List[int]:
    """Exact K-segment least-squares segmentation; returns the K-1 boundary frames
    (first frame of each new segment)."""
    T = len(X)
    if k < 2 or T < k * min_len:
        return []
    cost = segment_cost_table(X)
    D = np.full((k + 1, T + 1), np.inf)
    arg = np.zeros((k + 1, T + 1), dtype=int)
    D[0, 0] = 0.0
    for s in range(1, k + 1):
        for j in range(s * min_len, T + 1):
            i = np.arange((s - 1) * min_len, j - min_len + 1)
            v = D[s - 1, i] + cost[i, j]
            b = int(np.argmin(v))
            D[s, j], arg[s, j] = v[b], i[b]
    bounds, j = [], T
    for s in range(k, 0, -1):
        j = arg[s, j]
        bounds.append(int(j))
    return sorted(bounds)[1:]  # drop the leading 0


def novelty_curve(X: np.ndarray, window: int = 5) -> np.ndarray:
    """Cosine distance between mean embeddings of [t-w, t) and [t, t+w); 0 at the edges."""
    X = np.asarray(X, dtype=np.float64)
    T = len(X)
    nov = np.zeros(T)
    for t in range(window, T - window + 1):
        a, b = X[t - window:t].mean(0), X[t:t + window].mean(0)
        nov[t] = 1.0 - float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return nov


def novelty_peaks(nov: np.ndarray, k: int, min_gap: int = 5) -> List[int]:
    """Top-k local maxima of the novelty curve at least ``min_gap`` frames apart."""
    order = np.argsort(nov)[::-1]
    picked: List[int] = []
    for t in order:
        if nov[t] <= 0:
            break
        if all(abs(int(t) - p) >= min_gap for p in picked):
            picked.append(int(t))
        if len(picked) == k:
            break
    return sorted(picked)


def standardise(X: np.ndarray) -> np.ndarray:
    """Per-dimension z-score within the episode (unsupervised, no labels)."""
    X = np.asarray(X, dtype=np.float64)
    return (X - X.mean(0)) / np.maximum(X.std(0), 1e-8)


def first_boundary(bounds: Sequence[int]) -> Optional[int]:
    return int(bounds[0]) if len(bounds) else None


def depth_grid(depth: np.ndarray, valid: np.ndarray, lo: float, hi: float, grid: int = 16) -> np.ndarray:
    """Metric depth (H, W) -> normalised (grid*grid,) feature: (d - lo) / (hi - lo) clipped to [0, 1],
    invalid / far-plane pixels set to 1, then area-averaged to a grid x grid map."""
    d = np.clip((np.asarray(depth, dtype=np.float64) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    d[~np.asarray(valid, bool)] = 1.0
    H, W = d.shape
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    return np.array([[d[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean() for j in range(grid)] for i in range(grid)]).ravel()


def build_trajectory(blocks: Sequence[np.ndarray], balance: bool = False) -> np.ndarray:
    """Concatenate per-frame feature blocks (each (T, d_b)) after per-episode standardisation.
    With ``balance`` every block is divided by sqrt(d_b) so each block (camera x modality)
    contributes the same total variance regardless of its dimensionality."""
    out = []
    for B in blocks:
        Z = standardise(B)
        out.append(Z / np.sqrt(Z.shape[1]) if balance else Z)
    return np.concatenate(out, axis=1)


def cross_frame_attention(Z: np.ndarray, tau: float = 0.1, lam: float = 10.0, alpha: float = 0.5,
                          iters: int = 1) -> np.ndarray:
    """Training-free temporal self-attention over one camera's frame embeddings (T, D):
    A_ts = softmax_s(z_t.z_s / tau - |t - s| / lam), z'_t = (1 - alpha) z_t + alpha sum_s A_ts z_s,
    re-normalised. Frames are pulled toward similar, nearby frames (sharper phase clusters)."""
    Z = np.asarray(Z, dtype=np.float64)
    T = len(Z)
    t = np.arange(T)
    bias = -np.abs(t[:, None] - t[None, :]) / max(lam, 1e-6) if np.isfinite(lam) else 0.0
    for _ in range(iters):
        Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        logits = Zn @ Zn.T / tau + bias
        logits -= logits.max(1, keepdims=True)
        A = np.exp(logits)
        A /= A.sum(1, keepdims=True)
        Z = (1 - alpha) * Z + alpha * A @ Z
        Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    return Z
