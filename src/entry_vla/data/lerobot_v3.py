"""Reader for LeRobot v3.0 datasets (e.g. the reduced DuoBench export).

Path templates verified in lerobot v0.6.1 ``datasets/utils.py``:
``meta/info.json``, ``meta/episodes/chunk-XXX/file-YYY.parquet``,
``data/chunk-XXX/file-YYY.parquet``, ``videos/<key>/chunk-XXX/file-YYY.mp4``.
Episode rows carry ``data/chunk_index``, ``data/file_index``,
``dataset_from_index``/``dataset_to_index`` and per-video ``from_timestamp``.

The DuoBench LeRobot export is a *separate input format*: 16D joint-space
state/action (``left_joint_0..6, left_gripper, right_joint_0..6, right_gripper``),
224x224 video, no depth, no calibration, no simulator state, and actions
that are IK-converted absolute joint targets. Cartesian TCP poses are
therefore marked unavailable rather than reconstructed silently.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .episode import ActionSpec, Episode, new_meta

CHUNK_FILE = "chunk-{chunk_index:03d}/file-{file_index:03d}"


def _pq():
    try:
        import pyarrow.parquet as pq
        return pq
    except ImportError as e:  # pragma: no cover
        raise ImportError("pyarrow is required to read LeRobot datasets") from e


def read_info(root: str) -> Dict[str, Any]:
    with open(os.path.join(root, "meta", "info.json")) as f:
        return json.load(f)


def read_episodes_meta(root: str) -> List[Dict[str, Any]]:
    pq = _pq()
    d = os.path.join(root, "meta", "episodes")
    rows: List[Dict[str, Any]] = []
    for chunk in sorted(os.listdir(d)):
        for fn in sorted(os.listdir(os.path.join(d, chunk))):
            if fn.endswith(".parquet"):
                rows += pq.read_table(os.path.join(d, chunk, fn)).to_pylist()
    return sorted(rows, key=lambda r: r["episode_index"])


def inspect(root: str) -> Dict[str, Any]:
    info = read_info(root)
    feats = {k: {"dtype": v.get("dtype"), "shape": v.get("shape"), "names": v.get("names")}
             for k, v in info.get("features", {}).items()}
    rep = {"root": root, "codebase_version": info.get("codebase_version"), "fps": info.get("fps"),
           "robot_type": info.get("robot_type"), "total_episodes": info.get("total_episodes"),
           "total_frames": info.get("total_frames"), "features": feats}
    if info.get("codebase_version") != "v3.0":
        rep["warning"] = "reader written for codebase_version v3.0"
    return rep


def read_episode_frames(root: str, episode_index: int, max_frames: Optional[int] = None) -> Dict[str, np.ndarray]:
    pq = _pq()
    eps = {e["episode_index"]: e for e in read_episodes_meta(root)}
    if episode_index not in eps:
        raise IndexError(f"episode {episode_index} not in {root}")
    e = eps[episode_index]
    path = os.path.join(root, "data", CHUNK_FILE.format(chunk_index=e["data/chunk_index"],
                                                        file_index=e["data/file_index"]) + ".parquet")
    t = pq.read_table(path)
    idx = np.asarray(t.column("episode_index").to_numpy())
    sel = np.where(idx == episode_index)[0]
    if max_frames is not None:
        sel = sel[:max_frames]
    out = {}
    for name in t.column_names:
        col = t.column(name).take(sel).to_pylist()
        try:
            out[name] = np.asarray(col)
        except ValueError:
            out[name] = np.asarray(col, dtype=object)
    out["_episode_meta"] = e
    return out


def decode_video_frames(root: str, episode_meta: Dict[str, Any], video_key: str, n_frames: int,
                        fps: float) -> np.ndarray:
    """Decode the first ``n_frames`` of one episode from a concatenated v3 video file (needs PyAV)."""
    try:
        import av
    except ImportError as e:  # pragma: no cover
        raise ImportError("PyAV (pip install av) is required to decode LeRobot videos") from e
    vp = os.path.join(root, "videos", video_key, CHUNK_FILE.format(
        chunk_index=episode_meta[f"videos/{video_key}/chunk_index"],
        file_index=episode_meta[f"videos/{video_key}/file_index"]) + ".mp4")
    t0 = float(episode_meta[f"videos/{video_key}/from_timestamp"])
    frames = []
    with av.open(vp) as c:
        stream = c.streams.video[0]
        c.seek(int(t0 / stream.time_base), stream=stream, any_frame=False, backward=True)
        for fr in c.decode(stream):
            ts = float(fr.pts * stream.time_base)
            if ts + 0.5 / fps < t0:
                continue
            frames.append(fr.to_ndarray(format="rgb24"))
            if len(frames) >= n_frames:
                break
    return np.stack(frames).astype(np.uint8)


def export_episode(root: str, episode_index: int, task_id: str, backend: str = "duobench",
                   max_frames: Optional[int] = None, decode_videos: bool = False) -> Episode:
    info = read_info(root)
    fr = read_episode_frames(root, episode_index, max_frames)
    T = len(fr["timestamp"])
    state = np.asarray(fr["observation.state"].tolist(), dtype=np.float64)
    action = np.asarray(fr["action"].tolist(), dtype=np.float64)
    names = info["features"]["observation.state"].get("names") or []
    lowdim: Dict[str, np.ndarray] = {"time": np.asarray(fr["timestamp"], dtype=np.float64),
                                     "step": np.asarray(fr["frame_index"]), "valid": np.ones(T, bool),
                                     "source_index": np.asarray(fr["index"])}
    unavailable: Dict[str, str] = {}
    for a in ("left", "right"):
        ji = [names.index(f"{a}_joint_{i}") for i in range(7)] if names else list(range(7) if a == "left" else range(8, 15))
        gi = names.index(f"{a}_gripper") if names else (7 if a == "left" else 15)
        lowdim[f"{a}.q"] = state[:, ji]
        lowdim[f"{a}.gripper_native"] = state[:, gi]
        lowdim[f"{a}.cmd_joint_position"] = action[:, ji]
        lowdim[f"{a}.cmd_gripper"] = action[:, gi]
        unavailable[f"{a}.T_WE"] = "LeRobot export is joint-space only; FK needs the stock FR3+Robotiq model"
        unavailable[f"{a}.dq"] = "not in LeRobot export"
        unavailable[f"{a}.gripper_width"] = "only normalised/binarised gripper in LeRobot export"
    cams: Dict[str, Dict[str, Any]] = {}
    cam_meta: Dict[str, Any] = {}
    for key, feat in info["features"].items():
        if not key.startswith("observation.images."):
            continue
        cam = key.split(".", 2)[2]
        if decode_videos and feat.get("dtype") == "video":
            rgb = decode_video_frames(root, fr["_episode_meta"], key, T, float(info["fps"]))
            cams[cam] = {"rgb": rgb}
            cam_meta[cam] = {"width": rgb.shape[2], "height": rgb.shape[1], "depth_source_encoding": "unavailable",
                             "moving": "wrist" in cam, "modality": "rgb_video_lossy"}
            for k in ("depth", "depth_valid", "K", "T_WC"):
                unavailable[f"cam.{cam}.{k}"] = "LeRobot export carries no depth/calibration"
        else:
            unavailable[f"cam.{cam}"] = "video not decoded (pass decode_videos)"
    for k in ("seed", "variation_id", "sim_timestep_s"):
        unavailable[k] = "not in LeRobot export"
    meta = new_meta(
        episode_id=f"{backend}_lerobot_{task_id}_ep{episode_index:05d}", backend=backend, task_id=task_id,
        instruction=str(fr.get("task", [""])[0]) if "task" in fr else "", split="unassigned",
        recording_rate_hz=float(info["fps"]), provenance=f"LeRobot {info.get('codebase_version')} export ({root})",
        world_frame="joint_space_only",
        alignment={"obs_vs_action": "unverified", "detail": "converter drops consecutive duplicate actions; "
                   "check against raw recordings", "verified_by": None},
        source={"lerobot_root": root, "episode_index": episode_index,
                "source_demo_id": f"{task_id}/lerobot_ep{episode_index}"},
        cameras=cam_meta, gripper={"semantics": "lerobot gripper channel (possibly binarised >0.9 => 1=open)"},
    )
    meta["unavailable"].update(unavailable)
    meta["action_specs"] = [ActionSpec(key="cmd_joint_position", kind="commanded_target",
                                       controller="rcs lerobot_joint_converter IK of absolute_action",
                                       frame="joint_space", alignment="UNVERIFIED").__dict__]
    return Episode(meta=meta, lowdim=lowdim, cameras=cams, annotations={}, oracle={})
