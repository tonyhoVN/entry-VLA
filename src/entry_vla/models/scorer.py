"""Candidate outcome scorer (pre-travel) and arrival verifier (endpoint) -- separate models.

Labels come from autonomous rollouts of ONE frozen policy checkpoint under one
controller/reset/timeout protocol (``label_version``); a scorer trained on one
label version must not be applied to another without revalidation. Outputs
are learned estimates, not calibrated safety certificates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .entry_proposal import PointNetLite

CAND_DIM = 2 * (3 + 6 + 1) + 14      # per-arm pose (pos, rot6d, grip) + IK joints q_L, q_R


@dataclass
class ScorerConfig:
    point_feat_dim: int = 128
    hidden: int = 256
    text_dim: int = 64
    state_dim: int = 34
    cand_dim: int = CAND_DIM
    history_dim: int = 4          # e.g. gripper widths, held flags (contact/history features)


class OutcomeScorer(nn.Module):
    """Pre-travel scorer: context (observations available BEFORE travel) x candidate."""

    def __init__(self, cfg: ScorerConfig):
        super().__init__()
        self.cfg = cfg
        self.points = PointNetLite(6, cfg.point_feat_dim)
        ctx_in = 2 * cfg.point_feat_dim + cfg.text_dim + cfg.state_dim + cfg.history_dim
        self.ctx = nn.Sequential(nn.Linear(ctx_in, cfg.hidden), nn.ReLU())
        self.cand = nn.Sequential(nn.Linear(cfg.cand_dim, cfg.hidden), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(2 * cfg.hidden, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU())
        self.joint = nn.Linear(cfg.hidden, 1)          # p(success of the pair)
        self.per_arm = nn.Linear(cfg.hidden, 2)        # factorized p_left, p_right
        self.duration = nn.Linear(cfg.hidden, 1)       # log successful skill duration (s)

    def forward(self, b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c = self.ctx(torch.cat([self.points(b["points"], b["points_mask"]), b["text"], b["state"], b["history"]], -1))
        h = self.trunk(torch.cat([c, self.cand(b["candidate"])], -1))
        return {"joint_logit": self.joint(h).squeeze(-1), "arm_logits": self.per_arm(h),
                "log_duration": self.duration(h).squeeze(-1)}


class ArrivalVerifier(nn.Module):
    """Endpoint verifier: ACTUAL arrival observations (incl. wrist views captured after arrival)."""

    def __init__(self, cfg: ScorerConfig):
        super().__init__()
        self.points = PointNetLite(6, cfg.point_feat_dim)
        self.mlp = nn.Sequential(nn.Linear(2 * cfg.point_feat_dim + cfg.text_dim + cfg.state_dim + cfg.history_dim, cfg.hidden),
                                 nn.ReLU(), nn.Linear(cfg.hidden, 1))

    def forward(self, b: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.mlp(torch.cat([self.points(b["arrival_points"], b["arrival_points_mask"]), b["text"],
                                   b["arrival_state"], b["history"]], -1)).squeeze(-1)


def scorer_loss(out: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], w_arm: float = 0.5, w_dur: float = 0.1
                ) -> Dict[str, torch.Tensor]:
    """``success`` (B,) in {0,1}; ``arm_success`` (B,2) (NaN = unknown); ``duration_s`` (B,) for successes.
    Transit failures are excluded upstream (``mask``): the scorer predicts manipulation outcome given arrival."""
    m = b["mask"].float()
    n = m.sum().clamp_min(1)
    l_joint = (F.binary_cross_entropy_with_logits(out["joint_logit"], b["success"].float(), reduction="none") * m).sum() / n
    arm = b["arm_success"]
    known = (~torch.isnan(arm)).float() * m[:, None]
    l_arm = (F.binary_cross_entropy_with_logits(out["arm_logits"], torch.nan_to_num(arm), reduction="none") * known).sum() / known.sum().clamp_min(1)
    s = (b["success"] > 0).float() * m
    l_dur = (F.smooth_l1_loss(out["log_duration"], torch.log(b["duration_s"].clamp_min(1e-3)), reduction="none") * s).sum() / s.sum().clamp_min(1)
    return {"loss": l_joint + w_arm * l_arm + w_dur * l_dur, "joint": l_joint.detach(), "arm": l_arm.detach(), "dur": l_dur.detach()}
