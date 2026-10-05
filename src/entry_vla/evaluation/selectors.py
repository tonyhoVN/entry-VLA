"""Entry selectors (baselines + learned) operating on a pre-travel context.

Selectors see only pre-travel information: current observation (no future
wrist images), robot state and a *labelled* interaction-region estimate.
Oracle regions are allowed only as an explicitly labelled condition; the
``RegionEstimator`` records which one was used.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..geometry.franka import PandaArm
from ..geometry.rotations import geodesic_angle, rot_z
from ..geometry.shells import FilterConfig, ShellSpec, filter_pair, sample_shell_tcp_poses
from ..geometry.transforms import invert_T, make_T
from ..geometry.visibility import target_visibility
from ..sim.contracts import ARMS, Observation, UnsupportedCapabilityError


@dataclass
class RegionEstimate:
    T_WO: np.ndarray                         # region/object reference frame (position + yaw)
    points: Dict[str, np.ndarray]            # per-arm interaction points in W
    source: str                              # "oracle" | "oracle_noisy(sigma=..)" | "learned:<model>"


class RegionEstimator:
    """Interaction-region estimate for selectors.

    ``mode``: ``oracle`` (labelled oracle condition), ``oracle_noisy`` (oracle +
    injected localisation error), or ``learned`` (callable on the observation).
    ``oracle_points`` maps arm -> oracle key in ``obs.oracle`` (backend-specific)."""

    def __init__(self, mode: str, oracle_points: Optional[Dict[str, str]] = None, sigma_m: float = 0.0,
                 learned_fn: Optional[Callable[[Observation], Dict[str, np.ndarray]]] = None, seed: int = 0):
        self.mode, self.oracle_points, self.sigma = mode, oracle_points or {}, sigma_m
        self.learned_fn = learned_fn
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: Observation) -> RegionEstimate:
        if self.mode in ("oracle", "oracle_noisy"):
            pts = {}
            for a in ARMS:
                key = self.oracle_points.get(a)
                node: Any = obs.oracle
                for p in (key or "").split("."):
                    node = node.get(p) if isinstance(node, dict) else None
                if node is None:
                    raise UnsupportedCapabilityError(f"oracle region key {key!r} not in observation oracle")
                pts[a] = np.asarray(node, dtype=np.float64).copy()
            src = "oracle"
            if self.mode == "oracle_noisy":
                for a in ARMS:
                    pts[a] = pts[a] + self.rng.normal(scale=self.sigma, size=3)
                src = f"oracle_noisy(sigma={self.sigma})"
        elif self.mode == "learned":
            if self.learned_fn is None:
                raise ValueError("learned region mode needs learned_fn")
            pts, src = self.learned_fn(obs), "learned"
        else:
            raise ValueError(f"unknown region mode {self.mode!r}")
        c = 0.5 * (pts["left"] + pts["right"])
        d = pts["left"] - pts["right"]
        yaw = float(np.arctan2(d[1], d[0])) - np.pi / 2
        return RegionEstimate(T_WO=make_T(rot_z(yaw), c), points=pts, source=src)


@dataclass
class EntryCandidate:
    T_WE: Dict[str, np.ndarray]
    gripper: Dict[str, float] = field(default_factory=lambda: {a: 1.0 for a in ARMS})
    roles: Dict[str, str] = field(default_factory=lambda: {a: "approach" for a in ARMS})
    source: str = ""
    q: Optional[Dict[str, np.ndarray]] = None
    valid: Optional[bool] = None
    reject_reasons: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    def as_record(self) -> Dict[str, Any]:
        return {"T_WE": {a: self.T_WE[a].tolist() for a in ARMS}, "gripper": self.gripper, "roles": self.roles,
                "source": self.source, "valid": self.valid, "reject_reasons": self.reject_reasons, "scores": self.scores}


@dataclass
class PreTravelContext:
    obs: Observation
    region: RegionEstimate
    arms: Dict[str, PandaArm]
    filter_cfg: FilterConfig
    wrist_T_EC: Optional[Dict[str, np.ndarray]] = None   # hand-eye, for geometric (not image) visibility


def validate_candidates(ctx: PreTravelContext, cands: List[EntryCandidate]) -> List[EntryCandidate]:
    for c in cands:
        r = filter_pair(c.T_WE["left"], c.T_WE["right"], ctx.arms["left"], ctx.arms["right"],
                        ctx.obs.arms["left"].q, ctx.obs.arms["right"].q, ctx.filter_cfg)
        c.valid, c.reject_reasons = r["accepted"], r["reasons"]
        c.q = {"left": r["q_left"], "right": r["q_right"]}
        if r["pair_clearance"] is not None:
            c.scores["pair_clearance_m"] = float(r["pair_clearance"])
    return cands


def joint_travel(ctx: PreTravelContext, c: EntryCandidate) -> float:
    if c.q is None or any(v is None for v in c.q.values()):
        return np.inf
    return float(sum(np.abs(c.q[a] - ctx.obs.arms[a].q).sum() for a in ARMS))


class Selector(abc.ABC):
    name = "abstract"
    uses_scorer = False

    @abc.abstractmethod
    def propose(self, ctx: PreTravelContext) -> List[EntryCandidate]: ...

    def select(self, ctx: PreTravelContext) -> Dict[str, Any]:
        cands = validate_candidates(ctx, self.propose(ctx))
        valid = [c for c in cands if c.valid]
        ranked = self.rank(ctx, valid) if valid else []
        chosen = ranked[0] if ranked else None
        return {"selector": self.name, "region_source": ctx.region.source, "n_candidates": len(cands),
                "n_valid": len(valid), "chosen": chosen,
                "rejections": [c.reject_reasons for c in cands if not c.valid]}

    def rank(self, ctx: PreTravelContext, valid: List[EntryCandidate]) -> List[EntryCandidate]:
        return sorted(valid, key=lambda c: joint_travel(ctx, c))


def _in_region(ctx: PreTravelContext, T_OE: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {a: ctx.region.T_WO @ np.asarray(T_OE[a]) for a in ARMS}


class FixedCanonicalSelector(Selector):
    """One canonical paired entry expressed in the region frame O."""

    name = "fixed_canonical"

    def __init__(self, T_OE: Dict[str, np.ndarray]):
        self.T_OE = {a: np.asarray(v, dtype=np.float64) for a, v in T_OE.items()}

    def propose(self, ctx):
        return [EntryCandidate(_in_region(ctx, self.T_OE), source=self.name)]


class DistanceShellSelector(Selector):
    """Per-arm partial shells around each interaction point; paired; validity-filtered; min travel."""

    name = "distance_shell"

    def __init__(self, n: int = 32, radius: float = 0.20, elevation=(np.deg2rad(45), np.deg2rad(85)),
                 seed: int = 0, radius_jitter: float = 0.02):
        self.n, self.radius, self.elev, self.rj = n, radius, elevation, radius_jitter
        self.rng = np.random.default_rng(seed)

    def propose(self, ctx):
        poses = {a: sample_shell_tcp_poses(ShellSpec(center=ctx.region.points[a], radius=self.radius,
                                                     radius_jitter=self.rj, elevation_range=self.elev), self.n, self.rng)
                 for a in ARMS}
        return [EntryCandidate({a: poses[a][i] for a in ARMS}, source=self.name) for i in range(self.n)]


class VisibilityGeometricSelector(DistanceShellSelector):
    """Shell + predicted (geometric) wrist-camera visibility of the interaction point.
    Uses camera geometry only: no endpoint wrist image is requested before arrival."""

    name = "visibility_geometric"

    def __init__(self, K: np.ndarray, image_wh: Sequence[int], min_visible: float = 1.0, **kw):
        super().__init__(**kw)
        self.K, self.wh, self.min_visible = np.asarray(K), tuple(image_wh), min_visible

    def rank(self, ctx, valid):
        if ctx.wrist_T_EC is None:
            raise UnsupportedCapabilityError("visibility selector needs hand-eye transforms (wrist_T_EC)")
        keep = []
        for c in valid:
            vis = []
            for a in ARMS:
                T_WC = c.T_WE[a] @ ctx.wrist_T_EC[a]
                v = target_visibility(ctx.region.points[a][None], self.K, T_WC, self.wh[0], self.wh[1])
                vis.append(v.visible_fraction)
                c.scores[f"{a}.visibility_margin_px"] = v.mean_pixel_margin
            c.scores["visible_fraction"] = float(min(vis))
            if c.scores["visible_fraction"] >= self.min_visible:
                keep.append(c)
        return sorted(keep, key=lambda c: (-sum(c.scores.get(f"{a}.visibility_margin_px", 0) for a in ARMS),
                                            joint_travel(ctx, c)))


class NearestDemoSelector(Selector):
    """Demonstration-supported entries: annotated demo starts expressed in the
    demo's region frame, transferred to the current region estimate.
    Library rows: {"T_OE": {arm: 4x4}, "region_points": {arm: xyz}, "episode_id": ...}."""

    name = "nearest_demo"

    def __init__(self, library: List[Dict[str, Any]], k: int = 5):
        if not library:
            raise ValueError("empty demo-entry library")
        self.library, self.k = library, k

    @staticmethod
    def _sig(points: Dict[str, np.ndarray]) -> np.ndarray:
        d = np.asarray(points["left"]) - np.asarray(points["right"])
        return np.array([np.linalg.norm(d), d[2]])

    def propose(self, ctx):
        sig = self._sig(ctx.region.points)
        order = sorted(self.library, key=lambda r: float(np.linalg.norm(self._sig(r["region_points"]) - sig)))
        return [EntryCandidate(_in_region(ctx, r["T_OE"]), source=f"{self.name}:{r['episode_id']}") for r in order[:self.k]]

    def rank(self, ctx, valid):
        return valid    # keep nearest-first order


class ModelSelector(Selector):
    """Supervised entry proposals (models.entry_proposal) ranked by candidate logits."""

    name = "supervised"

    def __init__(self, propose_fn: Callable[[PreTravelContext], List[EntryCandidate]]):
        self.propose_fn = propose_fn

    def propose(self, ctx):
        return self.propose_fn(ctx)

    def rank(self, ctx, valid):
        return sorted(valid, key=lambda c: -c.scores.get("proposal_logit", 0.0))


class ScoredSelector(Selector):
    """Wraps a candidate generator with an outcome scorer.

    ``mode='joint'`` scores the pair; ``mode='factorized'`` multiplies per-arm
    success estimates. Scores are not calibrated safety certificates."""

    uses_scorer = True

    def __init__(self, generator: Selector, score_fn: Callable[[PreTravelContext, List[EntryCandidate]], List[Dict[str, float]]],
                 mode: str = "joint", min_success: float = 0.0, duration_weight: float = 0.0):
        if mode not in ("joint", "factorized"):
            raise ValueError("mode must be joint|factorized")
        self.gen, self.score_fn, self.mode = generator, score_fn, mode
        self.min_success, self.dw = min_success, duration_weight
        self.name = f"{mode}_scored[{generator.name}]"

    def propose(self, ctx):
        return self.gen.propose(ctx)

    def rank(self, ctx, valid):
        if not valid:
            return []
        scores = self.score_fn(ctx, valid)
        for c, s in zip(valid, scores):
            if self.mode == "factorized":
                c.scores["p_success"] = float(s["p_left"] * s["p_right"])
            else:
                c.scores["p_success"] = float(s["p_success"])
            c.scores["pred_duration_s"] = float(s.get("duration_s", 0.0))
        keep = [c for c in valid if c.scores["p_success"] >= self.min_success]
        return sorted(keep, key=lambda c: -(c.scores["p_success"] - self.dw * c.scores["pred_duration_s"]))

    def select(self, ctx):
        out = super().select(ctx)
        if out["chosen"] is None and out["n_valid"] > 0:
            out["rejected_by_scorer"] = True
        return out


def demo_library_from_annotations(episodes, annotations: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build NearestDemo rows from exported episodes + accepted start annotations."""
    rows = []
    for ep in episodes:
        ann = annotations.get(ep.episode_id)
        if not ann or ann.get("status") not in ("proposed", "manual"):
            continue
        t = int(ann["proposed_start"])
        pts = {a: np.asarray(ann["region_point"][a]) for a in ARMS}
        c = 0.5 * (pts["left"] + pts["right"])
        d = pts["left"] - pts["right"]
        T_WO = make_T(rot_z(float(np.arctan2(d[1], d[0])) - np.pi / 2), c)
        rows.append({"episode_id": ep.episode_id, "region_points": pts,
                     "T_OE": {a: invert_T(T_WO) @ ep.lowdim[f"{a}.T_WE"][t] for a in ARMS}})
    return rows


class FullEpisodeSelector(Selector):
    """VLA-only full-episode baseline: no preparation; the policy starts from the
    reset pose and gets the full-episode step budget (``trial.full_episode_steps``)."""

    name = "full_episode"
    skip_preparation = True

    def propose(self, ctx):
        return [EntryCandidate({a: ctx.obs.arms[a].T_WE.copy() for a in ARMS}, source=self.name)]

    def select(self, ctx):
        c = self.propose(ctx)[0]
        c.valid = True
        return {"selector": self.name, "region_source": ctx.region.source, "n_candidates": 1, "n_valid": 1,
                "chosen": c, "rejections": []}
