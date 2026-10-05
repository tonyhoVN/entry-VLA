"""Versioned shared episode format (``entry_vla.episode/1``).

On disk, one directory per episode::

    meta.json           ids, provenance, timing, conventions, availability map, action specs
    lowdim.npz          time/step/valid, per-arm measured state, raw actions, commanded targets
    cam_<name>.npz      rgb uint8 (T,H,W,3), depth float32 m (T,H,W), depth_valid, K, T_WC
    annotations.json    skill-start intervals, labels, roles/contacts (+confidence, provenance)
    oracle.npz          privileged namespace (object poses, masks, predicates) -- never a model input

Every required field is either present or listed in ``meta["unavailable"]``
with a reason. Conventions: metres, radians, seconds; ``T_AB`` maps B->A;
camera frames are OpenCV; quaternions do not appear in this format (poses
are 4x4); native raw actions keep their declared convention in ``action_specs``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

FORMAT_VERSION = "entry_vla.episode/1"
ARMS = ("left", "right")

REQUIRED_META = ("episode_id", "backend", "task_id", "variation_id", "instruction", "split", "seed",
                 "sim_timestep_s", "recording_rate_hz", "provenance", "world_frame", "alignment")
# lowdim keys that must exist or be declared unavailable
REQUIRED_LOWDIM = tuple(f"{a}.{k}" for a in ARMS for k in ("T_WE", "q", "dq", "gripper_width")) + ("time", "valid")
REQUIRED_CAMERA = ("rgb", "depth", "depth_valid", "K", "T_WC")


@dataclass
class ActionSpec:
    """Declares the convention of one stored action/target stream."""

    key: str                      # lowdim key prefix, e.g. "raw_action" or "cmd_target"
    kind: str                     # "raw_simulator_action" | "commanded_target" | "derived_target"
    controller: str               # e.g. "rcs CARTESIAN_TQuat relative_to=?" / "rlbench joint position"
    frame: str                    # frame of poses ("W", "arm_base_left", "joint_space", ...)
    quaternion_order: Optional[str] = None   # if the raw stream stored quaternions
    scale: str = "SI"             # units/scale note
    alignment: str = ""           # e.g. "row k = action applied after obs k-1, before obs k"
    notes: str = ""


@dataclass
class Episode:
    meta: Dict[str, Any]
    lowdim: Dict[str, np.ndarray]
    cameras: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)
    oracle: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(len(self.lowdim["time"]))

    @property
    def episode_id(self) -> str:
        return str(self.meta["episode_id"])

    def unavailable(self) -> Dict[str, str]:
        return self.meta.setdefault("unavailable", {})

    def mark_unavailable(self, key: str, reason: str) -> None:
        self.unavailable()[key] = reason

    def is_available(self, key: str) -> bool:
        return key not in self.unavailable()

    def arm(self, arm: str, key: str) -> np.ndarray:
        k = f"{arm}.{key}"
        if k not in self.lowdim:
            raise KeyError(f"{k} unavailable: {self.unavailable().get(k, 'not exported')}")
        return self.lowdim[k]


def new_meta(**kw) -> Dict[str, Any]:
    meta = {"format_version": FORMAT_VERSION, "unavailable": {}, "action_specs": [], "cameras": {}}
    meta.update(kw)
    return meta


def validate_episode(ep: Episode) -> List[str]:
    """Contract violations (empty list = valid)."""
    p: List[str] = []
    m = ep.meta
    if m.get("format_version") != FORMAT_VERSION:
        p.append(f"format_version {m.get('format_version')!r} != {FORMAT_VERSION}")
    for k in REQUIRED_META:
        if k not in m and k not in ep.unavailable():
            p.append(f"meta.{k} missing and not declared unavailable")
    T = len(ep.lowdim.get("time", []))
    if T == 0:
        p.append("no frames")
        return p
    for k in REQUIRED_LOWDIM:
        if k not in ep.lowdim and k not in ep.unavailable():
            p.append(f"lowdim.{k} missing and not declared unavailable")
    for k, v in ep.lowdim.items():
        if len(v) != T:
            p.append(f"lowdim.{k} has {len(v)} rows, expected {T}")
    t = ep.lowdim["time"]
    if np.any(np.diff(t) <= 0):
        p.append("time not strictly increasing")
    for a in ARMS:
        k = f"{a}.T_WE"
        if k in ep.lowdim:
            R = ep.lowdim[k][:, :3, :3]
            err = np.abs(np.einsum("tij,tkj->tik", R, R) - np.eye(3)).max()
            if err > 1e-4:
                p.append(f"{k} contains non-orthonormal rotations (err {err:.2e})")
    for spec in m.get("action_specs", []):
        for f_ in ("key", "kind", "controller", "frame"):
            if f_ not in spec:
                p.append(f"action_spec missing {f_}: {spec}")
        if spec.get("kind") not in ("raw_simulator_action", "commanded_target", "derived_target"):
            p.append(f"action_spec {spec.get('key')}: unknown kind {spec.get('kind')!r}")
    for name, cam in ep.cameras.items():
        cm = m.get("cameras", {}).get(name, {})
        for k in REQUIRED_CAMERA:
            if k not in cam and f"cam.{name}.{k}" not in ep.unavailable():
                p.append(f"camera {name}: {k} missing and not declared unavailable")
        for k in ("width", "height", "depth_source_encoding", "moving"):
            if k not in cm:
                p.append(f"meta.cameras.{name}.{k} missing")
        if "rgb" in cam:
            if cam["rgb"].dtype != np.uint8 or cam["rgb"].shape[0] != T or cam["rgb"].shape[-1] != 3:
                p.append(f"camera {name}: rgb must be uint8 (T,H,W,3)")
        if "depth" in cam:
            if cam["depth"].dtype != np.float32 or cam["depth"].shape[0] != T:
                p.append(f"camera {name}: depth must be float32 metres (T,H,W)")
        if "T_WC" in cam and cam["T_WC"].shape not in ((T, 4, 4), (4, 4)):
            p.append(f"camera {name}: T_WC must be (4,4) static or (T,4,4)")
        if cm.get("moving") and "T_WC" in cam and cam["T_WC"].shape != (T, 4, 4):
            p.append(f"camera {name}: moving camera needs per-frame T_WC")
    leaked = [k for k in ep.lowdim if k.startswith("oracle.")]
    if leaked:
        p.append(f"oracle keys stored in lowdim namespace: {leaked}")
    return p


# ------------------------------------------------------------------ I/O

def save_episode(ep: Episode, root: str, overwrite: bool = False) -> str:
    problems = validate_episode(ep)
    if problems:
        raise ValueError(f"refusing to save invalid episode {ep.episode_id}: {problems}")
    d = os.path.join(root, ep.episode_id)
    if os.path.exists(d) and not overwrite:
        raise FileExistsError(f"{d} exists (raw/exported episodes are immutable; pass overwrite explicitly)")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(ep.meta, f, indent=2, default=_json_default)
    np.savez_compressed(os.path.join(d, "lowdim.npz"), **ep.lowdim)
    for name, cam in ep.cameras.items():
        np.savez_compressed(os.path.join(d, f"cam_{name}.npz"), **cam)
    with open(os.path.join(d, "annotations.json"), "w") as f:
        json.dump(ep.annotations, f, indent=2, default=_json_default)
    if ep.oracle:
        np.savez_compressed(os.path.join(d, "oracle.npz"), **ep.oracle)
    return d


def load_episode(path: str, cameras: Optional[List[str]] = None, load_images: bool = True) -> Episode:
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported format {meta.get('format_version')!r}")
    with np.load(os.path.join(path, "lowdim.npz")) as z:
        lowdim = {k: z[k] for k in z.files}
    cams = {}
    names = cameras if cameras is not None else list(meta.get("cameras", {}))
    for name in names:
        fp = os.path.join(path, f"cam_{name}.npz")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"camera {name} not exported in {path}")
        with np.load(fp) as z:
            keys = z.files if load_images else [k for k in z.files if k in ("K", "T_WC")]
            cams[name] = {k: z[k] for k in keys}
    ann = {}
    ap = os.path.join(path, "annotations.json")
    if os.path.exists(ap):
        with open(ap) as f:
            ann = json.load(f)
    oracle = {}
    op = os.path.join(path, "oracle.npz")
    if os.path.exists(op):
        with np.load(op, allow_pickle=False) as z:
            oracle = {k: z[k] for k in z.files}
    return Episode(meta=meta, lowdim=lowdim, cameras=cams, annotations=ann, oracle=oracle)


def list_episodes(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.exists(os.path.join(root, d, "meta.json")))


def save_annotations(path: str, annotations: Dict[str, Any]) -> None:
    """Annotations are the only mutable part of an exported episode."""
    with open(os.path.join(path, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=2, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")
