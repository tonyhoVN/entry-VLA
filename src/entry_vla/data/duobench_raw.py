"""DuoBench raw recordings (RCS ``StorageWrapper`` parquet) -> shared episodes.

Schema (parsed from the footer of ``carry_pot/sim/2026-05-15_part-0_no_rotation.parquet``):
``obs.{arm}.{tquat,joints,xyzrpy,gripper}``,
``obs.frames.{cam}.rgb.{data(JPEG),intrinsics(3x4),extrinsics(4x4 world->cam)}``,
``info.{arm}.{sim_state,sim_state_schema,absolute_action,is_grasped,gripper_width,collision,...}``,
``action.*`` (one-step-delayed copy), ``env_action.*`` (applied at this step),
``uuid``, ``step``, ``instruction``, ``timestamp``, ``success``, task-specific ``info.*``.

Facts checked on 3 rows of that file: timestamps 1/30 s apart; ``action[k] == env_action[k-1]``;
``info.gripper_width == obs.gripper`` (both normalised, NOT metres); ``env_action.tquat`` is
not in the frame of ``obs.tquat``. The file has no depth columns.

Frames: ``obs.*.tquat`` is in the RCS *shared base frame*; camera extrinsics are in the
MuJoCo world. The exporter needs ``T_world_shared`` (from the scene config:
``root_frame_to_world @ shared_base_frame_to_root_frame``). Use
``entry-vla inspect --handeye`` to check it: with the right transform, the wrist
camera's hand-eye transform is constant over frames.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from ..geometry.camera import resize_images
from ..geometry.rotations import so3_log
from ..geometry.transforms import invert_T, pose_to_T
from .episode import ActionSpec, Episode, new_meta

ARMS = ("left", "right")
CAMERAS = ("head", "left_wrist", "right_wrist")
QUAT = "xyzw"
ROBOTIQ_2F85_STROKE_M = 0.085


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
        return pq
    except ImportError as e:  # pragma: no cover
        raise ImportError("pyarrow is required to read DuoBench raw parquet: pip install pyarrow") from e


def get(row: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = row
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def decode_jpeg(data: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


def parquet_summary(path: str) -> Dict[str, Any]:
    pq = _require_pyarrow()
    f = pq.ParquetFile(path)
    return {"path": path, "num_rows": f.metadata.num_rows, "num_row_groups": f.metadata.num_row_groups,
            "columns": [f.schema.column(i).path for i in range(len(f.schema))]}


def iter_rows(path: str, columns: Optional[Sequence[str]] = None, max_rows: Optional[int] = None) -> Iterator[Dict]:
    """Stream rows row-group by row-group (never loads the whole file)."""
    pq = _require_pyarrow()
    f = pq.ParquetFile(path)
    n = 0
    for g in range(f.metadata.num_row_groups):
        for row in f.read_row_group(g, columns=list(columns) if columns else None).to_pylist():
            yield row
            n += 1
            if max_rows is not None and n >= max_rows:
                return


def read_episode_rows(path: str, episode: int = 0, max_frames: Optional[int] = None,
                      columns: Optional[Sequence[str]] = None) -> List[Dict]:
    """Rows of the ``episode``-th uuid in file order (episodes are contiguous runs of ``uuid``)."""
    cols = list(columns) if columns else None
    if cols is not None and "uuid" not in cols:
        cols.append("uuid")
    out: List[Dict] = []
    current, idx = None, -1
    for row in iter_rows(path, cols):
        if row["uuid"] != current:
            if out:
                break
            current, idx = row["uuid"], idx + 1
        if idx == episode:
            out.append(row)
            if max_frames is not None and len(out) >= max_frames:
                break
    if not out:
        raise IndexError(f"episode {episode} not found in {path}")
    return out


def _mat(v: Any, shape) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(shape)


def handeye_consistency(rows: List[Dict], T_world_shared: np.ndarray, arm: str = "left") -> Dict[str, float]:
    """Spread of ``T_EC = inv(T_WE) T_WC`` for the arm's wrist camera across rows.
    Near-zero spread supports the assumed ``T_world_shared``; large spread means
    the arm and camera frames are inconsistent."""
    Ts = []
    for r in rows:
        ext = get(r, f"obs.frames.{arm}_wrist.rgb.extrinsics")
        if ext is None:
            raise KeyError(f"rows lack obs.frames.{arm}_wrist.rgb.extrinsics")
        tq = np.asarray(get(r, f"obs.{arm}.tquat"))
        T_WE = T_world_shared @ pose_to_T(tq[:3], tq[3:7], QUAT)
        Ts.append(invert_T(T_WE) @ invert_T(_mat(ext, (4, 4))))
    Ts = np.stack(Ts)
    rot = max(float(np.linalg.norm(so3_log(Ts[0, :3, :3].T @ T[:3, :3]))) for T in Ts)
    return {"frames": len(Ts), "translation_spread_m": float(np.ptp(Ts[:, :3, 3], axis=0).max()),
            "rotation_spread_rad": rot, "T_EC_first": Ts[0].tolist()}


def alignment_report(rows: List[Dict]) -> Dict[str, Any]:
    """Check the documented delayed-action convention on actual rows."""
    rep: Dict[str, Any] = {}
    ts = np.array([r["timestamp"] for r in rows])
    if len(ts) > 1:
        dt = np.diff(ts)
        rep["dt_mean_s"], rep["dt_min_s"], rep["dt_max_s"] = float(dt.mean()), float(dt.min()), float(dt.max())
    match_prev, match_same = [], []
    for k in range(1, len(rows)):
        a = np.asarray(get(rows[k], "action.left.tquat", []))
        e_prev = np.asarray(get(rows[k - 1], "env_action.left.tquat", []))
        e_same = np.asarray(get(rows[k], "env_action.left.tquat", []))
        if a.size and e_prev.size:
            match_prev.append(bool(np.allclose(a, e_prev)))
            match_same.append(bool(np.allclose(a, e_same)))
    rep["action_equals_prev_env_action"] = match_prev
    rep["action_equals_same_env_action"] = match_same
    g_obs = [float(np.asarray(get(r, "obs.left.gripper", [np.nan]))[0]) for r in rows]
    g_inf = [get(r, "info.left.gripper_width") for r in rows]
    rep["info_gripper_width_equals_obs_gripper"] = [bool(i is not None and np.isclose(i, o)) for i, o in zip(g_inf, g_obs)]
    return rep


def export_rows(rows: List[Dict], task_id: str, source_file: str, episode_index: int, cfg: Dict[str, Any]) -> Episode:
    """Convert one episode's rows to the shared format.

    ``cfg`` keys: ``T_world_shared`` (4x4, required unless
    ``allow_unverified_identity_frame``), ``T_shared_armbase.{left,right}`` (optional, to
    express ``absolute_action`` in W), ``image_hw`` ([H,W] or null), ``cameras``,
    ``gripper_stroke_m``.
    """
    T = len(rows)
    T_ws = cfg.get("T_world_shared")
    unverified_frame = T_ws is None
    if unverified_frame and not cfg.get("allow_unverified_identity_frame", False):
        raise ValueError("T_world_shared is required (compose root_frame_to_world @ shared_base_frame_to_root_frame "
                         "from the task config, or set allow_unverified_identity_frame=true and validate with --handeye)")
    T_ws = np.eye(4) if unverified_frame else _mat(T_ws, (4, 4))
    stroke = float(cfg.get("gripper_stroke_m", ROBOTIQ_2F85_STROKE_M))
    ts = np.array([float(r["timestamp"]) for r in rows])
    lowdim: Dict[str, np.ndarray] = {
        "time": ts - ts[0], "step": np.array([int(r["step"]) for r in rows]), "valid": np.ones(T, bool),
        "source_index": np.arange(T),
    }
    derived: Dict[str, str] = {}
    for a in ARMS:
        tq = np.stack([np.asarray(get(r, f"obs.{a}.tquat")) for r in rows])
        lowdim[f"{a}.T_WE"] = np.stack([T_ws @ pose_to_T(x[:3], x[3:7], QUAT) for x in tq])
        q = np.stack([np.asarray(get(r, f"obs.{a}.joints"), dtype=np.float64) for r in rows])
        lowdim[f"{a}.q"] = q
        lowdim[f"{a}.dq"] = np.gradient(q, ts, axis=0) if T > 1 else np.zeros_like(q)
        derived[f"{a}.dq"] = "np.gradient of measured joints over recorded timestamps (not measured velocity)"
        g = np.array([float(np.asarray(get(r, f"obs.{a}.gripper"))[0]) for r in rows])
        lowdim[f"{a}.gripper_native"] = g
        lowdim[f"{a}.gripper_width"] = g * stroke
        derived[f"{a}.gripper_width"] = f"normalised obs gripper x nominal stroke {stroke} m (linearity UNVERIFIED)"
        ea = np.stack([np.asarray(get(r, f"env_action.{a}.tquat")) for r in rows])
        lowdim[f"{a}.raw_action_tquat"] = ea
        lowdim[f"{a}.raw_action_gripper"] = np.array([float(np.asarray(get(r, f"env_action.{a}.gripper"))[0]) for r in rows])
        aa = [get(r, f"info.{a}.absolute_action") for r in rows]
        if all(x is not None for x in aa):
            aa = np.stack([np.asarray(x) for x in aa])
            lowdim[f"{a}.cmd_tquat_native"] = aa
            T_sb = (cfg.get("T_shared_armbase") or {}).get(a)
            if T_sb is not None:
                lowdim[f"{a}.cmd_T_WE"] = np.stack([T_ws @ _mat(T_sb, (4, 4)) @ pose_to_T(x[:3], x[3:7], QUAT) for x in aa])
            lowdim[f"{a}.cmd_gripper"] = lowdim[f"{a}.raw_action_gripper"]
        st = [get(r, f"info.{a}.sim_state") for r in rows]
        if all(s is not None for s in st) and a == "left":
            lowdim["sim_state"] = np.stack([np.asarray(s, dtype=np.float64) for s in st])
    ep_id = f"duobench_{task_id}_{rows[0]['uuid'][:12]}"
    image_hw = cfg.get("image_hw")
    cams: Dict[str, Dict[str, Any]] = {}
    cam_meta: Dict[str, Any] = {}
    unavailable: Dict[str, str] = {}
    for c in cfg.get("cameras", CAMERAS):
        if get(rows[0], f"obs.frames.{c}.rgb.data") is None:
            unavailable[f"cam.{c}"] = "camera not present in rows"
            continue
        K0 = _mat(get(rows[0], f"obs.frames.{c}.rgb.intrinsics"), (3, 4))[:3, :3]
        T_WC = np.stack([invert_T(_mat(get(r, f"obs.frames.{c}.rgb.extrinsics"), (4, 4))) for r in rows])
        imgs = []
        K = K0
        for r in rows:
            im = decode_jpeg(get(r, f"obs.frames.{c}.rgb.data"))
            if image_hw:
                im, _, _, K = resize_images(im, None, None, K0, tuple(image_hw))
            imgs.append(im)
        moving = not np.allclose(T_WC, T_WC[0], atol=1e-6)
        cams[c] = {"rgb": np.stack(imgs), "K": K, "T_WC": T_WC if moving else T_WC[0]}
        unavailable[f"cam.{c}.depth"] = "duobench_raw has no depth columns; use replay-render with include_depth"
        unavailable[f"cam.{c}.depth_valid"] = "no depth"
        cam_meta[c] = {"width": imgs[0].shape[1], "height": imgs[0].shape[0], "near": None, "far": None,
                       "depth_source_encoding": "unavailable", "moving": bool(moving), "modality": "rgb_recorded",
                       "jpeg_compressed_source": True, "source_size_hw": list(decode_jpeg(get(rows[0], f"obs.frames.{c}.rgb.data")).shape[:2])}
    unavailable["seed"] = "scene seed not recorded in duobench_raw"
    unavailable["variation_id"] = "no variation ids in DuoBench"
    unavailable["sim_timestep_s"] = "physics timestep not recorded; control period from timestamps"
    meta = new_meta(
        episode_id=ep_id, backend="duobench", task_id=task_id, instruction=str(rows[0].get("instruction", "")),
        split="unassigned", recording_rate_hz=float(1.0 / np.median(np.diff(ts))) if T > 1 else None,
        provenance="DuoBench human teleoperation recording (duobench_raw)",
        world_frame="mujoco_world" if not unverified_frame else "duobench_shared_base (IDENTITY ASSUMED, UNVERIFIED)",
        alignment={"obs_vs_action": "obs_after_action",
                   "detail": "row k holds obs after env_action[k] was applied; action[k] == env_action[k-1]",
                   "verified_by": "alignment_report on exported rows"},
        source={"file": source_file, "episode_index_in_file": episode_index, "uuid": rows[0]["uuid"],
                "first_step": int(rows[0]["step"]), "source_demo_id": rows[0]["uuid"]},
        cameras=cam_meta,
        gripper={"semantics": "normalized_width_0closed_1open", "model": "Robotiq 2F-85", "stroke_m": stroke},
        derived=derived,
        reset={"procedure": "unverified: DuoBench reset(seed) cannot reproduce teleop scene (seed not recorded)",
               "sim_state": "info.left.sim_state per row (joint qpos/qvel per schema); completeness UNVERIFIED",
               "sim_state_schema": get(rows[0], "info.left.sim_state_schema")},
    )
    meta["unavailable"].update(unavailable)
    meta["action_specs"] = [
        ActionSpec(key="raw_action", kind="raw_simulator_action", controller="RCS env_action tquat (+gripper); "
                   "recording-time control_mode/relative_to UNVERIFIED", frame="UNVERIFIED (not obs.tquat frame)",
                   quaternion_order=QUAT, alignment="raw_action[k] applied before obs[k]").__dict__,
        ActionSpec(key="cmd", kind="commanded_target", controller="RCS info.absolute_action (absolute TCP target)",
                   frame="arm base frame (native); cmd_T_WE in W only if T_shared_armbase given",
                   quaternion_order=QUAT, alignment="cmd[k] = target of the action applied before obs[k]").__dict__,
    ]
    oracle: Dict[str, np.ndarray] = {}
    for a in ARMS:
        for k in ("is_grasped", "collision", "ik_success", "is_sim_converged"):
            vals = [get(r, f"info.{a}.{k}") for r in rows]
            if all(v is not None for v in vals):
                oracle[f"{k}.{a}"] = np.array(vals, dtype=bool)
    if all(r.get("success") is not None for r in rows):
        oracle["success"] = np.array([bool(r["success"]) for r in rows])
    for k, v in (rows[0].get("info") or {}).items():
        if k not in ARMS and isinstance(v, (bool, float, int)) and k not in ("frame_timestamp",):
            oracle[f"info.{k}"] = np.array([get(r, f"info.{k}") for r in rows], dtype=np.float64)
    return Episode(meta=meta, lowdim=lowdim, cameras=cams, annotations={}, oracle=oracle)
