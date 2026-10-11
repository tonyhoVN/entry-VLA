"""Conditional diffusion model for the entry (key-frame) gripper pose.

Pose x = [p_C / pos_scale (3), rot6d (6)] with p_C = p_W - c (object-centroid canonical frame, axes
parallel to world) and R_C = R_W. Condition
    c = Fuse(E_PointNet++(P_C), E_vision(I_crop), E_text(l))
where E_vision / E_text are frozen VLM features (precomputed, scripts/embed_vlm_features.py), projected
and, for the per-camera crop tokens, attention-pooled with a validity mask. The noise predictor
eps_theta(x_n, n, c) is an MLP with FiLM residual blocks. Training: DDPM epsilon objective with a cosine
schedule and condition dropout (classifier-free guidance at sampling). Sampling: deterministic DDIM.
Optional extra conditions: object centroid c in world (robot reachability) via ``use_centroid``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2 import PointNet2Encoder

POSE_DIM = 9


@dataclass
class EntryDiffusionConfig:
    vlm_dim: int = 2560          # Qwen3-VL-4B hidden size
    cond_dim: int = 256
    point_dim: int = 256
    hidden: int = 512
    n_blocks: int = 4
    n_steps: int = 100           # diffusion steps
    pos_scale: float = 0.2       # metres -> unit scale for p_C
    use_centroid: bool = True
    use_vision: bool = True
    use_text: bool = True
    cond_dropout: float = 0.1    # classifier-free guidance training
    dropout: float = 0.0


def rot6d_to_matrix_robust(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Gram-Schmidt (columns = 6D halves) that stays a rotation when the halves are (near) parallel:
    the second axis then falls back to b1 x e, with e the world axis least aligned with b1."""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    u = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    e = torch.eye(3, dtype=x.dtype, device=x.device)[b1.abs().argmin(-1)]
    alt = torch.cross(b1, e, dim=-1)
    u = torch.where(u.norm(dim=-1, keepdim=True) > eps, u, alt)
    b2 = F.normalize(u, dim=-1, eps=eps)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)


def cosine_alpha_bar(n_steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, n_steps, n_steps + 1) / n_steps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-5, 1.0)
    return ab[1:]                                           # alpha_bar for n = 1..N (index n-1)


def timestep_embedding(n: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=n.device) / half)
    a = n.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


class FiLMBlock(nn.Module):
    def __init__(self, hidden: int, cond: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1, self.fc2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.film = nn.Linear(cond, 2 * hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, c):
        g, b = self.film(c).chunk(2, -1)
        x = self.norm(h) * (1 + g) + b
        return h + self.fc2(self.drop(F.silu(self.fc1(F.silu(x)))))


class ConditionEncoder(nn.Module):
    def __init__(self, cfg: EntryDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.points = PointNet2Encoder(in_feat=3, out_dim=cfg.point_dim)
        d = cfg.cond_dim
        self.text = nn.Sequential(nn.LayerNorm(cfg.vlm_dim), nn.Linear(cfg.vlm_dim, d))
        self.vision = nn.Sequential(nn.LayerNorm(cfg.vlm_dim), nn.Linear(cfg.vlm_dim, d))
        self.cam_query = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.centroid = nn.Linear(3, d)
        n_in = cfg.point_dim + 3 * d
        self.fuse = nn.Sequential(nn.Linear(n_in, d), nn.SiLU(), nn.Linear(d, d))
        self.null = nn.Parameter(torch.zeros(d))         # unconditional embedding (CFG)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        B = batch["obj_points"].shape[0]
        z = [self.points(batch["obj_points"])]
        d = cfg.cond_dim
        t = self.text(batch["text_feat"].float()) if cfg.use_text else batch["obj_points"].new_zeros(B, d)
        if cfg.use_vision:
            v = self.vision(batch["crop_feat"].float())                      # (B, V, d)
            m = batch["crop_mask"].bool()
            att = (v @ self.cam_query) / math.sqrt(d)
            att = att.masked_fill(~m, float("-inf")).softmax(-1).nan_to_num(0.0)
            v = (att[..., None] * v).sum(1)
        else:
            v = t.new_zeros(B, d)
        cen = self.centroid(batch["centroid"].float()) if cfg.use_centroid else t.new_zeros(B, d)
        return self.fuse(torch.cat(z + [t, v, cen], -1))


class EntryDiffusion(nn.Module):
    def __init__(self, cfg: EntryDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.cond = ConditionEncoder(cfg)
        h = cfg.hidden
        self.inp = nn.Linear(POSE_DIM, h)
        self.temb = nn.Sequential(nn.Linear(128, cfg.cond_dim), nn.SiLU(), nn.Linear(cfg.cond_dim, cfg.cond_dim))
        self.blocks = nn.ModuleList(FiLMBlock(h, cfg.cond_dim, cfg.dropout) for _ in range(cfg.n_blocks))
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, POSE_DIM))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)
        self.register_buffer("alpha_bar", cosine_alpha_bar(cfg.n_steps))

    def eps(self, x: torch.Tensor, n: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Noise prediction eps_theta(x_n, n, c); n in 1..N."""
        k = c + self.temb(timestep_embedding(n, 128))
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, k)
        return self.out(h)

    # pose <-> x
    def encode_pose(self, pos_C: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        r6 = torch.cat([R[..., :, 0], R[..., :, 1]], -1)
        return torch.cat([pos_C / self.cfg.pos_scale, r6], -1)

    def decode_pose(self, x: torch.Tensor):
        return x[..., :3] * self.cfg.pos_scale, rot6d_to_matrix_robust(x[..., 3:9])

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x0 = self.encode_pose(batch["tgt_pos_C"].float(), batch["tgt_rot"].float())
        B = x0.shape[0]
        c = self.cond(batch)
        if self.training and self.cfg.cond_dropout > 0:
            drop = torch.rand(B, device=c.device) < self.cfg.cond_dropout
            c = torch.where(drop[:, None], self.cond.null.expand_as(c), c)
        n = torch.randint(1, self.cfg.n_steps + 1, (B,), device=x0.device)
        ab = self.alpha_bar[n - 1][:, None]
        noise = torch.randn_like(x0)
        xn = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        return F.mse_loss(self.eps(xn, n, c), noise)

    @torch.no_grad()
    def sample(self, batch: Dict[str, torch.Tensor], n_samples: int = 16, ddim_steps: int = 25,
               guidance: float = 1.0, generator: Optional[torch.Generator] = None):
        """Returns positions (B, S, 3) in the canonical frame and rotations (B, S, 3, 3)."""
        c = self.cond(batch)
        B = c.shape[0]
        c = c.repeat_interleave(n_samples, 0)
        null = self.cond.null.expand_as(c)
        x = torch.randn(B * n_samples, POSE_DIM, device=c.device, generator=generator)
        ns = torch.linspace(self.cfg.n_steps, 1, ddim_steps, device=c.device).round().long()
        for i, n in enumerate(ns):
            nn_ = n.expand(x.shape[0])
            e = self.eps(x, nn_, c)
            if guidance != 1.0:
                e = self.eps(x, nn_, null) + guidance * (e - self.eps(x, nn_, null))
            ab = self.alpha_bar[n - 1]
            x0 = (x - (1 - ab).sqrt() * e) / ab.sqrt()
            x0 = torch.cat([x0[:, :3].clamp(-5, 5), x0[:, 3:].clamp(-1.5, 1.5)], -1)
            ab_prev = self.alpha_bar[ns[i + 1] - 1] if i + 1 < len(ns) else x.new_tensor(1.0)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * e
        pos, R = self.decode_pose(x)
        return pos.view(B, n_samples, 3), R.view(B, n_samples, 3, 3)
