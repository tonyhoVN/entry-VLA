"""Diagnostic overlays for skill-start annotation (PNG frames, GIF strips, timeline plots)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..geometry.camera import project_world
from .episode import Episode

ARMS = ("left", "right")
COLORS = {"left": (255, 80, 80), "right": (80, 160, 255), "region": (255, 220, 0)}


def _dot(img: np.ndarray, u: float, v: float, color, r: int = 2) -> None:
    H, W = img.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    img[max(0, vi - r):min(H, vi + r + 1), max(0, ui - r):min(W, ui + r + 1)] = color


def overlay_frame(ep: Episode, cam: str, t: int, annotation: Optional[Dict[str, Any]] = None) -> np.ndarray:
    c = ep.cameras[cam]
    if "rgb" not in c:
        raise KeyError(f"camera {cam} has no rgb in this episode")
    img = c["rgb"][t].copy()
    T_WC = c["T_WC"][t] if c["T_WC"].ndim == 3 else c["T_WC"]
    for a in ARMS:
        uv, z = project_world(ep.lowdim[f"{a}.T_WE"][t, :3, 3], c["K"], T_WC)
        if z[0] > 0:
            _dot(img, uv[0, 0], uv[0, 1], COLORS[a])
        if annotation and "region_point" in annotation:
            uv, z = project_world(np.asarray(annotation["region_point"][a]), c["K"], T_WC)
            if z[0] > 0:
                _dot(img, uv[0, 0], uv[0, 1], COLORS["region"], r=1)
    if annotation and annotation.get("interval"):
        a0, a1 = annotation["interval"]
        if a0 <= t <= a1:
            img[:3] = (0, 255, 0)            # green bar: frame inside the proposed start interval
    return img


def save_overlays(ep: Episode, annotation: Dict[str, Any], out_dir: str, cams: List[str], stride: int = 2,
                  gif: bool = True) -> List[str]:
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for cam in cams:
        frames = [overlay_frame(ep, cam, t, annotation) for t in range(0, ep.num_frames, stride)]
        if annotation.get("proposed_start") is not None:
            p = os.path.join(out_dir, f"{ep.episode_id}_{cam}_start.png")
            Image.fromarray(overlay_frame(ep, cam, annotation["proposed_start"], annotation)).save(p)
            written.append(p)
        if gif and frames:
            p = os.path.join(out_dir, f"{ep.episode_id}_{cam}.gif")
            Image.fromarray(frames[0]).save(p, save_all=True, append_images=[Image.fromarray(f) for f in frames[1:]],
                                            duration=100, loop=0)
            written.append(p)
    return written


def save_timeline(ep: Episode, annotation: Dict[str, Any], crit: Dict[str, Dict[str, Any]], out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = ep.lowdim["time"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for a in ARMS:
        if crit[a].get("onset") is None:
            continue
        axes[0].plot(t, crit[a]["dist"], label=f"{a} dist to region (m)")
        axes[1].plot(t, np.rad2deg(crit[a]["angle"]), label=f"{a} angle to onset orient. (deg)")
        axes[2].plot(t, ep.lowdim[f"{a}.gripper_width"], label=f"{a} gripper width (m)")
        for ax in axes:
            ax.axvline(t[crit[a]["onset"]], ls="--", lw=0.8)
    if annotation.get("interval"):
        for ax in axes:
            ax.axvspan(t[annotation["interval"][0]], t[annotation["interval"][1]], alpha=0.2, color="green")
    axes[0].axhline(annotation["provenance"]["params"]["radius"], color="k", lw=0.5)
    for ax in axes:
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{ep.episode_id}: {annotation.get('status')} (conf {annotation.get('confidence')})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path
