"""Supervised K-candidate paired entry proposal network (torch).

Inputs (all pre-travel; no future wrist images, no future actions):
  * ``points``   (B, N, 6)  xyz (in the declared pose frame) + rgb in [0,1], from GLOBAL RGB-D views
  * ``points_mask`` (B, N) valid points
  * ``text``     (B, D_text) frozen instruction embedding
  * ``state``    (B, S)  robot state (20D Cartesian [+ q_L, q_R]) in the pose frame
  * ``geometry`` (B, G)  optional object/region geometry (labelled source; zeros if absent)
Outputs per candidate k and arm a: position (3, metres, pose frame), rot6d (6),
gripper-open logit (1), role logits (R); plus candidate logits (K).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS = ("left", "right")


@dataclass
class EntryModelConfig:
    num_candidates: int = 8
    point_feat_dim: int = 128
    hidden: int = 256
    text_dim: int = 64
    state_dim: int = 34          # 20D Cartesian + 2x7 joints
    geometry_dim: int = 6        # e.g. two interaction points (labelled source)
    num_roles: int = 2           # approach | holding
    pose_frame: str = "region_point"   # "W" or "region_point" (positions relative to region centre)
    dropout: float = 0.0

    def out_per_arm(self) -> int:
        return 3 + 6 + 1 + self.num_roles


class PointNetLite(nn.Module):
    """Shared per-point MLP + masked max/mean pooling."""

    def __init__(self, in_dim: int = 6, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
                                 nn.Linear(128, out_dim))

    def forward(self, pts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        f = self.mlp(pts)                                     # (B, N, C)
        m = mask.unsqueeze(-1).to(f.dtype)
        neg = torch.finfo(f.dtype).min
        fmax = torch.where(m > 0, f, torch.full_like(f, neg)).max(1).values
        fmax = torch.where(torch.isfinite(fmax) & (fmax > neg / 2), fmax, torch.zeros_like(fmax))
        fmean = (f * m).sum(1) / m.sum(1).clamp_min(1.0)
        return torch.cat([fmax, fmean], -1)


def rot6d_to_matrix_t(x: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3), Gram-Schmidt (columns = first two 6D halves)."""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class EntryProposalNet(nn.Module):
    def __init__(self, cfg: EntryModelConfig):
        super().__init__()
        self.cfg = cfg
        self.points = PointNetLite(6, cfg.point_feat_dim)
        fuse_in = 2 * cfg.point_feat_dim + cfg.text_dim + cfg.state_dim + cfg.geometry_dim
        self.fuse = nn.Sequential(nn.Linear(fuse_in, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU())
        K, P = cfg.num_candidates, cfg.out_per_arm()
        self.head = nn.Linear(cfg.hidden, K * 2 * P)
        self.logits = nn.Linear(cfg.hidden, K)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c = self.cfg
        B = batch["points"].shape[0]
        g = batch.get("geometry")
        if g is None:
            g = batch["points"].new_zeros(B, c.geometry_dim)
        h = self.fuse(torch.cat([self.points(batch["points"], batch["points_mask"]), batch["text"], batch["state"], g], -1))
        out = self.head(h).view(B, c.num_candidates, 2, c.out_per_arm())
        pos = out[..., 0:3]
        r6 = out[..., 3:9] + torch.tensor([1, 0, 0, 0, 1, 0], dtype=out.dtype, device=out.device)  # identity bias
        return {"pos": pos, "rot": rot6d_to_matrix_t(r6), "rot6d": r6, "grip_logit": out[..., 9],
                "role_logits": out[..., 10:], "cand_logits": self.logits(h)}


@dataclass
class LossConfig:
    w_pos: float = 1.0           # per metre (L1 on translation)
    w_rot: float = 0.5           # per radian (geodesic)
    w_grip: float = 0.1
    w_role: float = 0.1
    w_cls: float = 0.2           # candidate classification toward the assigned candidate
    wta_epsilon: float = 0.05    # relaxed WTA: non-winners get epsilon/(K-1) weight (fights collapse/dead heads)


def geodesic_t(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    tr = (Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))


def min_over_k_loss(pred: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor], cfg: LossConfig) -> Dict[str, torch.Tensor]:
    """Paired (both-arm) min-over-K with relaxed winner-take-all.

    ``tgt``: pos (B,2,3) m, rot (B,2,3,3), grip_open (B,2) in {0,1}, role (B,2) long.
    The assignment uses the PAIRED pose cost so the two arms of one candidate
    are never matched to different demonstrations' modes."""
    pos_err = (pred["pos"] - tgt["pos"][:, None]).abs().sum(-1)                       # (B,K,2) m
    rot_err = geodesic_t(pred["rot"], tgt["rot"][:, None].expand_as(pred["rot"]))     # (B,K,2) rad
    pose_cost = (cfg.w_pos * pos_err + cfg.w_rot * rot_err).sum(-1)                   # (B,K)
    K = pose_cost.shape[1]
    win = pose_cost.argmin(1)
    w = torch.full_like(pose_cost, cfg.wta_epsilon / max(K - 1, 1))
    w.scatter_(1, win[:, None], 1.0 - cfg.wta_epsilon if K > 1 else 1.0)
    l_pose = (w * pose_cost).sum(1).mean()
    grip = F.binary_cross_entropy_with_logits(pred["grip_logit"], tgt["grip_open"][:, None].expand_as(pred["grip_logit"]).float(),
                                              reduction="none").sum(-1)
    R = pred["role_logits"].shape[-1]
    role = F.cross_entropy(pred["role_logits"].reshape(-1, R), tgt["role"][:, None].expand(-1, K, -1).reshape(-1),
                           reduction="none").view(-1, K, 2).sum(-1)
    l_aux = ((w * (cfg.w_grip * grip + cfg.w_role * role)).sum(1)).mean()
    l_cls = F.cross_entropy(pred["cand_logits"], win)
    total = l_pose + l_aux + cfg.w_cls * l_cls
    with torch.no_grad():
        best_pos = pos_err.gather(1, win[:, None, None].expand(-1, 1, 2)).squeeze(1)    # (B,2)
        best_rot = rot_err.gather(1, win[:, None, None].expand(-1, 1, 2)).squeeze(1)
    return {"loss": total, "pose": l_pose.detach(), "aux": l_aux.detach(), "cls": l_cls.detach(),
            "min_pos_err_m": best_pos.mean(), "min_rot_err_rad": best_rot.mean(), "winner": win}


def collapse_stats(pred: Dict[str, torch.Tensor], winners: torch.Tensor) -> Dict[str, float]:
    """Candidate-collapse monitor: mean pairwise candidate spread (m) and winner entropy (nats)."""
    with torch.no_grad():
        p = pred["pos"].flatten(2)                                   # (B,K,6)
        d = torch.cdist(p, p).mean().item()
        K = pred["pos"].shape[1]
        hist = torch.bincount(winners, minlength=K).float()
        q = hist / hist.sum().clamp_min(1)
        ent = float(-(q[q > 0] * q[q > 0].log()).sum())
        return {"candidate_spread_m": d, "winner_entropy": ent, "max_entropy": float(torch.log(torch.tensor(float(K)))),
                "used_candidates": int((hist > 0).sum())}


def decode_candidates(pred: Dict[str, torch.Tensor], T_frame_W: torch.Tensor | None = None) -> List[Dict[str, object]]:
    """Predictions -> per-sample list of K candidates with 4x4 poses in W.

    If the pose frame is ``region_point``, pass ``T_frame_W`` (B,4,4) = the
    region frame in W used to build the inputs; positions/rotations are mapped back."""
    B, K = pred["pos"].shape[:2]
    out = []
    for b in range(B):
        cands = []
        for k in range(K):
            T = {}
            for i, a in enumerate(ARMS):
                M = torch.eye(4, dtype=pred["pos"].dtype)
                M[:3, :3] = pred["rot"][b, k, i].detach().cpu()
                M[:3, 3] = pred["pos"][b, k, i].detach().cpu()
                if T_frame_W is not None:
                    M = T_frame_W[b].detach().cpu().to(M.dtype) @ M
                T[a] = M.numpy()
            cands.append({"T_WE": T, "gripper_open_p": torch.sigmoid(pred["grip_logit"][b, k]).detach().cpu().numpy(),
                          "role": pred["role_logits"][b, k].argmax(-1).detach().cpu().numpy(),
                          "logit": float(pred["cand_logits"][b, k])})
        out.append(cands)
    return out
