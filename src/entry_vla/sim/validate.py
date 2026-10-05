"""Observation-contract checks shared by smoke tests and exporters."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..geometry.camera import project_world
from ..geometry.transforms import is_valid_T
from .contracts import ARMS, Observation


def check_observation(obs: Observation, require_depth: bool = False) -> List[str]:
    """Return a list of contract violations (empty = OK)."""
    problems = []
    for a in ARMS:
        if a not in obs.arms:
            problems.append(f"missing arm {a}")
            continue
        st = obs.arms[a]
        if not is_valid_T(st.T_WE, atol=1e-4):
            problems.append(f"{a}: T_WE is not a valid rigid transform")
        if st.q.shape != (7,):
            problems.append(f"{a}: expected 7 joints, got {st.q.shape}")
        if st.gripper_width is not None and not (-1e-3 <= st.gripper_width <= 0.2):
            problems.append(f"{a}: implausible gripper width {st.gripper_width} m")
    for name, cam in obs.cameras.items():
        if not is_valid_T(cam.T_WC, atol=1e-4):
            problems.append(f"{name}: T_WC invalid")
        if cam.K.shape != (3, 3) or cam.K[2, 2] != 1.0:
            problems.append(f"{name}: K malformed")
        if cam.rgb is not None:
            if cam.rgb.dtype != np.uint8 or cam.rgb.shape != (cam.height, cam.width, 3):
                problems.append(f"{name}: rgb must be uint8 (H,W,3), got {cam.rgb.dtype} {cam.rgb.shape}")
        if cam.depth is None:
            if require_depth:
                problems.append(f"{name}: depth required but unavailable")
        else:
            if cam.depth.dtype != np.float32 or cam.depth.shape != (cam.height, cam.width):
                problems.append(f"{name}: depth must be float32 (H,W), got {cam.depth.dtype} {cam.depth.shape}")
            v = cam.depth[cam.depth_valid] if cam.depth_valid is not None else cam.depth[np.isfinite(cam.depth)]
            if v.size and (v.min() < 0 or v.max() > 50):
                problems.append(f"{name}: depth range [{v.min():.3f},{v.max():.3f}] m implausible (units?)")
            if cam.depth_source_encoding == "unavailable":
                problems.append(f"{name}: depth present but source encoding undeclared")
    return problems


def tcp_projection_report(obs: Observation) -> Dict[str, Any]:
    """Project each TCP into each camera: a cheap calibration sanity check.
    A TCP 'behind' its own wrist camera, or far outside the exo views,
    indicates a frame/extrinsics convention error."""
    rep: Dict[str, Any] = {}
    for name, cam in obs.cameras.items():
        rep[name] = {}
        for a in ARMS:
            uv, z = project_world(obs.arms[a].T_WE[:3, 3], cam.K, cam.T_WC)
            u, v = float(uv[0, 0]), float(uv[0, 1])
            rep[name][a] = {"u": u, "v": v, "z_m": float(z[0]),
                            "in_image": bool(z[0] > 0 and 0 <= u < cam.width and 0 <= v < cam.height)}
    return rep


def depth_summary(obs: Observation) -> Dict[str, Any]:
    out = {}
    for name, cam in obs.cameras.items():
        if cam.depth is None:
            out[name] = {"available": False}
            continue
        m = cam.depth_valid if cam.depth_valid is not None else np.isfinite(cam.depth)
        v = cam.depth[m]
        out[name] = {"available": True, "encoding": cam.depth_source_encoding, "valid_fraction": float(m.mean()),
                     "min_m": float(v.min()) if v.size else None, "max_m": float(v.max()) if v.size else None}
    return out
