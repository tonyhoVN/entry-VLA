"""Skill-start interval proposals for bimanual demonstrations.

Per arm, frames before interaction onset are candidate starts when:
  * TCP is within ``radius`` (0.20 m baseline, NOT ground truth) of the arm's
    interaction point,
  * TCP orientation is within ``max_angle`` of the onset orientation,
  * the gripper is in the mode required by the arm's role (open for approach,
    closed/holding for a holding role),
  * the interaction point has been visible for ``stable_frames`` consecutive
    frames in at least one configured camera (weak readiness signal only),
  * at least ``min_lead_s`` remain before onset (local approach is retained).
A *paired* start requires both arms to satisfy their role conditions at the same
frame, before either arm's onset (no independent cropping into incompatible
phases). Stage progress is used for diagnostics only, never as a label.

Interaction point sources: ``tcp_at_onset`` (non-privileged, default) or an
``oracle`` key (annotation bootstrap; recorded in provenance).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..geometry.camera import project_world
from ..geometry.rotations import geodesic_angle
from .episode import Episode

ARMS = ("left", "right")
METHOD = "entry_vla.skill_start_heuristic/v1"


@dataclass
class AnnotateConfig:
    label: str = "bimanual_local_skill"
    radius: float = 0.20
    max_angle_rad: float = float(np.deg2rad(45))
    open_width_m: float = 0.03            # width above which the gripper counts as open
    close_drop_m: float = 0.01            # width drop that marks closing onset
    min_lead_s: float = 0.3
    stable_frames: int = 3
    visibility_cameras: List[str] = field(default_factory=list)  # empty = skip visibility criterion
    region_source: str = "tcp_at_onset"   # or "oracle:handles.{arm}" / "oracle:region_center"
    onset_source: str = "gripper_close"   # or "oracle:grasped.{arm}"
    roles: Dict[str, str] = field(default_factory=lambda: {"left": "approach", "right": "approach"})


def _onset(ep: Episode, arm: str, cfg: AnnotateConfig) -> Optional[int]:
    if cfg.onset_source.startswith("oracle:"):
        key = cfg.onset_source.split(":", 1)[1].format(arm=arm)
        flags = np.asarray(ep.oracle[key]).astype(bool)
        idx = np.where(flags)[0]
        return int(idx[0]) if len(idx) else None
    w = ep.arm(arm, "gripper_width")
    w_max = np.maximum.accumulate(w)
    idx = np.where((w_max - w) > cfg.close_drop_m)[0]
    return int(idx[0]) if len(idx) else None


def _region(ep: Episode, arm: str, onset: int, cfg: AnnotateConfig) -> np.ndarray:
    if cfg.region_source == "tcp_at_onset":
        return ep.arm(arm, "T_WE")[onset, :3, 3]
    if cfg.region_source.startswith("oracle:"):
        key = cfg.region_source.split(":", 1)[1].format(arm=arm)
        v = np.asarray(ep.oracle[key])
        return v[onset] if v.ndim == 2 else v
    raise ValueError(f"unknown region_source {cfg.region_source!r}")


def _visible(ep: Episode, point: np.ndarray, cams: List[str]) -> np.ndarray:
    T = ep.num_frames
    if not cams:
        return np.ones(T, bool)
    vis = np.zeros(T, bool)
    for c in cams:
        cam, cm = ep.cameras[c], ep.meta["cameras"][c]
        for t in range(T):
            T_WC = cam["T_WC"][t] if cam["T_WC"].ndim == 3 else cam["T_WC"]
            uv, z = project_world(point, cam["K"], T_WC)
            u, v = uv[0]
            if z[0] > 0 and 0 <= u < cm["width"] and 0 <= v < cm["height"]:
                if "depth" in cam:     # depth test against the recorded depth (occlusion)
                    d = cam["depth"][t][int(round(min(max(v, 0), cm["height"] - 1))), int(round(min(max(u, 0), cm["width"] - 1)))]
                    if d > 0 and d < z[0] - 0.03:
                        continue
                vis[t] = True
    return vis


def _stable(mask: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(mask)
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        out[i] = run >= k
    return out


def arm_criteria(ep: Episode, arm: str, cfg: AnnotateConfig) -> Dict[str, Any]:
    onset = _onset(ep, arm, cfg)
    T = ep.num_frames
    if onset is None:
        return {"onset": None, "reason": "no interaction onset found"}
    p = _region(ep, arm, onset, cfg)
    TWE = ep.arm(arm, "T_WE")
    dist = np.linalg.norm(TWE[:, :3, 3] - p, axis=1)
    ang = geodesic_angle(TWE[:, :3, :3], np.repeat(TWE[onset, :3, :3][None], T, 0))
    w = ep.arm(arm, "gripper_width")
    role = cfg.roles.get(arm, "approach")
    grip_ok = w >= cfg.open_width_m if role == "approach" else w < cfg.open_width_m
    vis = _stable(_visible(ep, p, cfg.visibility_cameras), cfg.stable_frames)
    t = ep.lowdim["time"]
    lead_ok = (t[onset] - t) >= cfg.min_lead_s
    before = np.arange(T) < onset if role == "approach" else np.ones(T, bool)
    ok = (dist <= cfg.radius) & (ang <= cfg.max_angle_rad) & grip_ok & vis & lead_ok & before & ep.lowdim["valid"]
    return {"onset": onset, "region_point": p, "dist": dist, "angle": ang, "grip_ok": grip_ok, "visible": vis,
            "lead_ok": lead_ok, "ok": ok, "role": role}


def propose_paired_start(ep: Episode, cfg: AnnotateConfig) -> Dict[str, Any]:
    crit = {a: arm_criteria(ep, a, cfg) for a in ARMS}
    prov = {"method": METHOD, "params": asdict(cfg), "uses_oracle": cfg.region_source.startswith("oracle")
            or cfg.onset_source.startswith("oracle"), "source": "automatic"}
    if any(crit[a]["onset"] is None for a in ARMS):
        return {"label": cfg.label, "status": "rejected", "reason": {a: crit[a].get("reason") for a in ARMS},
                "provenance": prov}
    joint = crit["left"]["ok"] & crit["right"]["ok"]
    idx = np.where(joint)[0]
    if len(idx) == 0:
        return {"label": cfg.label, "status": "rejected", "reason": "no frame satisfies both arms' criteria",
                "onset": {a: crit[a]["onset"] for a in ARMS}, "provenance": prov}
    # last contiguous run of candidates before the earliest onset: retains local approach
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    run = runs[-1]
    start = int(run[0])
    soft = []
    for a in ARMS:
        c = crit[a]
        soft.append(1.0 - c["dist"][start] / cfg.radius)
        soft.append(1.0 - c["angle"][start] / cfg.max_angle_rad)
    confidence = float(np.clip(np.mean(soft), 0, 1)) * min(1.0, len(run) / max(cfg.stable_frames, 1))
    return {
        "label": cfg.label, "status": "proposed",
        "interval": [int(run[0]), int(run[-1])], "proposed_start": start,
        "onset": {a: crit[a]["onset"] for a in ARMS},
        "roles": {a: crit[a]["role"] for a in ARMS},
        "contact_prerequisites": {a: ("holding" if crit[a]["role"] != "approach" else "none") for a in ARMS},
        "criteria_at_start": {a: {"dist_m": float(crit[a]["dist"][start]), "angle_rad": float(crit[a]["angle"][start]),
                                  "visible": bool(crit[a]["visible"][start])} for a in ARMS},
        "region_point": {a: crit[a]["region_point"].tolist() for a in ARMS},
        "confidence": confidence, "provenance": prov,
        "stage_at_start_diagnostic": int(ep.oracle["stage"][start]) if "stage" in ep.oracle else None,
    }


def apply_overrides(ep_id: str, proposal: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Manual overrides keyed by episode id: ``{reject: bool, proposed_start, interval, label, roles, note}``."""
    ov = overrides.get(ep_id)
    if not ov:
        return proposal
    out = dict(proposal)
    if ov.get("reject"):
        out["status"] = "rejected"
        out["reason"] = f"manual: {ov.get('note', '')}"
    for k in ("proposed_start", "interval", "label", "roles", "contact_prerequisites"):
        if k in ov:
            out[k] = ov[k]
    if not ov.get("reject") and ("proposed_start" in ov or "interval" in ov):
        out["status"] = "manual"
    out["provenance"] = {**proposal.get("provenance", {}), "source": "manual_override", "note": ov.get("note", ""),
                         "automatic": {k: proposal.get(k) for k in ("proposed_start", "interval", "status")}}
    out["confidence"] = 1.0 if out["status"] == "manual" else proposal.get("confidence")
    return out
