"""PointNet++ single-scale-grouping encoder in pure torch (no CUDA extensions).

Set abstraction (SA): farthest point sampling -> ball query (radius, k neighbours, padded with the
first neighbour) -> shared MLP on [relative xyz, features] -> max pool. Two local SA layers + a global
SA layer give one feature vector per cloud. Coordinates are metres in the canonical object frame.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


def farthest_point_sample(xyz: torch.Tensor, m: int) -> torch.Tensor:
    """(B, N, 3) -> (B, m) indices; deterministic start at index 0."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, m, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device)
    far = torch.zeros(B, dtype=torch.long, device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(m):
        idx[:, i] = far
        d = ((xyz - xyz[ar, far][:, None]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(1)
    return idx


def ball_query(xyz: torch.Tensor, centres: torch.Tensor, radius: float, k: int) -> torch.Tensor:
    """(B, N, 3), (B, M, 3) -> (B, M, k) neighbour indices within radius (padded with the nearest)."""
    d = torch.cdist(centres, xyz)                                    # (B, M, N)
    d_sorted, idx = d.topk(k, dim=-1, largest=False)
    first = idx[..., :1].expand_as(idx)
    return torch.where(d_sorted <= radius, idx, first)


def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (B, N, C), idx (B, ...) -> (B, ..., C)."""
    B = x.shape[0]
    flat = idx.reshape(B, -1)
    out = x.gather(1, flat[..., None].expand(-1, -1, x.shape[-1]))
    return out.view(*idx.shape, x.shape[-1])


def mlp(dims: Sequence[int]) -> nn.Sequential:
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU()]
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    def __init__(self, n_centres: Optional[int], radius: float, k: int, in_dim: int, dims: Sequence[int]):
        super().__init__()
        self.n_centres, self.radius, self.k = n_centres, radius, k
        self.mlp = mlp([in_dim + 3, *dims])

    def forward(self, xyz: torch.Tensor, feat: Optional[torch.Tensor]):
        if self.n_centres is None:                                   # global: one group of all points
            g = xyz[:, None]
            f = feat[:, None] if feat is not None else None
            centres = xyz.new_zeros(xyz.shape[0], 1, 3)
            rel = g
        else:
            centres = gather(xyz, farthest_point_sample(xyz, self.n_centres))
            nb = ball_query(xyz, centres, self.radius, self.k)
            rel = gather(xyz, nb) - centres[:, :, None]
            f = gather(feat, nb) if feat is not None else None
        h = torch.cat([rel, f], -1) if f is not None else rel
        return centres, self.mlp(h).max(2).values


class PointNet2Encoder(nn.Module):
    """(B, N, 3+C) -> (B, out_dim). Default radii suit table-top objects (metres)."""

    def __init__(self, in_feat: int = 3, out_dim: int = 256, n1: int = 256, n2: int = 64,
                 r1: float = 0.03, r2: float = 0.08, k: int = 32):
        super().__init__()
        self.in_feat = in_feat
        self.sa1 = SetAbstraction(n1, r1, k, in_feat, [64, 64, 128])
        self.sa2 = SetAbstraction(n2, r2, k, 128, [128, 128, 256])
        self.sa3 = SetAbstraction(None, 0.0, 0, 256, [256, 512, out_dim])

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        xyz, feat = pts[..., :3].float(), (pts[..., 3:].float() if self.in_feat else None)
        xyz, feat = self.sa1(xyz, feat)
        xyz, feat = self.sa2(xyz, feat)
        _, feat = self.sa3(xyz, feat)
        return feat.squeeze(1)
