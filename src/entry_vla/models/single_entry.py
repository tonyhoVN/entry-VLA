"""Single-arm K-candidate entry-pose network for RLBench (torch).

Inputs (pre-travel, frame t < entry): fused exo point cloud (B, N, 6) world xyz + rgb, frozen instruction
embedding (B, D), robot state (B, 17). Positions are centred on SCENE_CENTRE (world frame, metres).
Outputs per candidate: TCP position (3, m, world), rot6d -> SO(3), gripper-open logit, candidate logit.

Loss: relaxed winner-take-all over K (min-over-K with epsilon weight on the other candidates) of
w_pos * L1 position (per metre) + w_rot * geodesic angle (per radian). The parallel gripper is
symmetric under a pi rotation about its approach axis (TCP z), so the rotation error is the minimum
over {R, R Rz(pi)}; the raw (asymmetric) error is reported as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .entry_proposal import PointNetLite, geodesic_t, rot6d_to_matrix_t

SCENE_CENTRE = (0.3, 0.0, 1.0)
RZ_PI = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))


@dataclass
class SingleEntryConfig:
    num_candidates: int = 4
    point_feat_dim: int = 128
    hidden: int = 256
    text_dim: int = 64
    state_dim: int = 17
    dropout: float = 0.1
    symmetric_gripper: bool = True


class SingleEntryNet(nn.Module):
    def __init__(self, cfg: SingleEntryConfig):
        super().__init__()
        self.cfg = cfg
        self.points = PointNetLite(6, cfg.point_feat_dim)
        fuse_in = 2 * cfg.point_feat_dim + cfg.text_dim + cfg.state_dim
        self.fuse = nn.Sequential(nn.Linear(fuse_in, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU())
        self.head = nn.Linear(cfg.hidden, cfg.num_candidates * 10)
        self.logits = nn.Linear(cfg.hidden, cfg.num_candidates)
        self.register_buffer("centre", torch.tensor(SCENE_CENTRE))

    def forward(self, points, points_mask, text, state) -> Dict[str, torch.Tensor]:
        c = self.cfg
        pts = torch.cat([points[..., :3] - self.centre, points[..., 3:]], -1)
        st = torch.cat([state[..., :3] - self.centre, state[..., 3:]], -1)
        h = self.fuse(torch.cat([self.points(pts, points_mask), text, st], -1))
        out = self.head(h).view(-1, c.num_candidates, 10).float()
        r6 = out[..., 3:9] + out.new_tensor([1, 0, 0, 0, 1, 0])
        return {"pos": out[..., :3] + self.centre, "rot": rot6d_to_matrix_t(r6), "grip_logit": out[..., 9],
                "cand_logits": self.logits(h).float()}


def rot_error(R_pred: torch.Tensor, R_tgt: torch.Tensor, symmetric: bool) -> torch.Tensor:
    e = geodesic_t(R_pred, R_tgt)
    if symmetric:
        e = torch.minimum(e, geodesic_t(R_pred, R_tgt @ RZ_PI.to(R_tgt)))
    return e


@dataclass
class SingleLossConfig:
    w_pos: float = 1.0       # per metre
    w_rot: float = 0.2       # per radian
    w_grip: float = 0.05
    w_cls: float = 0.1
    wta_epsilon: float = 0.05


def single_entry_loss(pred, tgt_pos, tgt_rot, tgt_grip, cfg: SingleLossConfig, symmetric: bool = True):
    K = pred["pos"].shape[1]
    pos_err = (pred["pos"] - tgt_pos[:, None]).abs().sum(-1)                          # (B,K) m (L1)
    rot_err = rot_error(pred["rot"], tgt_rot[:, None].expand_as(pred["rot"]), symmetric)  # (B,K) rad
    cost = cfg.w_pos * pos_err + cfg.w_rot * rot_err
    win = cost.argmin(1)
    w = torch.full_like(cost, cfg.wta_epsilon / max(K - 1, 1))
    w.scatter_(1, win[:, None], 1.0 - cfg.wta_epsilon if K > 1 else 1.0)
    grip = F.binary_cross_entropy_with_logits(pred["grip_logit"], tgt_grip[:, None].expand(-1, K), reduction="none")
    loss = (w * (cost + cfg.w_grip * grip)).sum(1).mean() + cfg.w_cls * F.cross_entropy(pred["cand_logits"], win)
    return loss, win


@torch.no_grad()
def entry_metrics(pred, tgt_pos, tgt_rot, tgt_grip, symmetric: bool = True) -> Dict[str, torch.Tensor]:
    """Per-sample errors of the top-1 candidate (highest logit) and the best of K (oracle choice)."""
    top = pred["cand_logits"].argmax(1)
    pos_d = (pred["pos"] - tgt_pos[:, None]).norm(dim=-1)                             # (B,K) Euclidean m
    rot_s = rot_error(pred["rot"], tgt_rot[:, None].expand_as(pred["rot"]), symmetric)
    rot_raw = geodesic_t(pred["rot"], tgt_rot[:, None].expand_as(pred["rot"]))
    g = lambda x: x.gather(1, top[:, None]).squeeze(1)  # noqa: E731
    grip_ok = ((pred["grip_logit"] > 0).float() == tgt_grip[:, None]).float()
    return {"top1_pos_m": g(pos_d), "top1_rot_rad": g(rot_s), "top1_rot_raw_rad": g(rot_raw),
            "top1_grip_acc": g(grip_ok), "minK_pos_m": pos_d.min(1).values, "minK_rot_rad": rot_s.min(1).values}
