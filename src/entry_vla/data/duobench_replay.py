"""User-run preprocessing: depth-enabled re-rendering of DuoBench raw recordings.

For each recorded row, the recorded ``info.<arm>.sim_state`` is restored with
RCS's replay mechanism and the scene is rendered with ``include_depth=True``.
Rendering runs in the simulator environment, separate from model training.

Replay is validated before use: the rendered TCP pose must match the recorded
``obs.<arm>.tquat``, and recorded object joints must move as recorded. A
restored joint state is not a complete resumable physics snapshot (contacts,
controller targets, task bookkeeping); these episodes are for perception and
geometry, NOT for resuming physics mid-episode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..geometry.rotations import geodesic_angle
from ..geometry.transforms import invert_T, pose_to_T
from ..sim.duobench import DuoBenchBackend
from .duobench_raw import ARMS, QUAT, export_rows, get, read_episode_rows


def validate_replay(recorded_rows: List[Dict], rendered_T_WE: Dict[str, np.ndarray], T_world_shared: np.ndarray,
                    pos_tol: float = 0.005, rot_tol: float = 0.02) -> Dict[str, Any]:
    errs = {}
    for a in ARMS:
        rec = np.stack([T_world_shared @ pose_to_T(np.asarray(get(r, f"obs.{a}.tquat"))[:3],
                                                   np.asarray(get(r, f"obs.{a}.tquat"))[3:7], QUAT)
                        for r in recorded_rows])
        ren = rendered_T_WE[a]
        errs[a] = {"max_pos_err_m": float(np.max(np.linalg.norm(rec[:, :3, 3] - ren[:, :3, 3], axis=1))),
                   "max_rot_err_rad": float(np.max(geodesic_angle(rec[:, :3, :3], ren[:, :3, :3])))}
    ok = all(e["max_pos_err_m"] <= pos_tol and e["max_rot_err_rad"] <= rot_tol for e in errs.values())
    st = np.stack([np.asarray(get(r, "info.left.sim_state")) for r in recorded_rows])
    return {"tcp_errors": errs, "within_tolerance": ok,
            "sim_state_changes": int(np.sum(np.any(np.abs(np.diff(st, axis=0)) > 1e-9, axis=1))),
            "note": "joint-level restore; contacts/controller/task bookkeeping not restored"}


def replay_render_episode(parquet: str, episode_index: int, task_id: str, backend_cfg: Dict[str, Any],
                          export_cfg: Dict[str, Any], max_frames: Optional[int]) -> Dict[str, Any]:
    rows = read_episode_rows(parquet, episode_index, max_frames=max_frames)
    be = DuoBenchBackend({**backend_cfg, "include_depth": True})
    be.make(task_id)
    try:
        be.reset(seed=0)
        rendered = {a: [] for a in ARMS}
        cams: Dict[str, Dict[str, list]] = {}
        for r in rows:
            hold = {a: {"tquat": list(get(r, f"obs.{a}.tquat")), "gripper": list(get(r, f"env_action.{a}.gripper"))}
                    for a in ARMS}
            snap = {"sim_state": np.asarray(get(r, "info.left.sim_state")),
                    "sim_state_schema": get(r, "info.left.sim_state_schema")}
            obs = be.restore_state_snapshot(snap, hold_action=hold)
            for a in ARMS:
                rendered[a].append(obs.arms[a].T_WE)
            for name, c in obs.cameras.items():
                d = cams.setdefault(name, {"rgb": [], "depth": [], "depth_valid": [], "T_WC": [], "K": c.K})
                d["rgb"].append(c.rgb); d["depth"].append(c.depth); d["depth_valid"].append(c.depth_valid)
                d["T_WC"].append(c.T_WC)
        rendered = {a: np.stack(v) for a, v in rendered.items()}
        T_ws = be.T_W_shared
        report = validate_replay(rows, rendered, T_ws)
    finally:
        be.close()
    ep = export_rows(rows, task_id, parquet, episode_index, {**export_cfg, "T_world_shared": T_ws.tolist(),
                                                              "cameras": []})
    for name, d in cams.items():
        T_WC = np.stack(d["T_WC"])
        moving = not np.allclose(T_WC, T_WC[0], atol=1e-6)
        ep.cameras[name] = {"rgb": np.stack(d["rgb"]), "depth": np.stack(d["depth"]).astype(np.float32),
                            "depth_valid": np.stack(d["depth_valid"]), "K": d["K"], "T_WC": T_WC if moving else T_WC[0]}
        ep.meta["cameras"][name] = {"width": d["rgb"][0].shape[1], "height": d["rgb"][0].shape[0], "near": None,
                                    "far": None, "depth_source_encoding": "uint16_mm", "moving": bool(moving),
                                    "modality": "rgbd_sim_rerendered"}
        ep.meta["unavailable"].pop(f"cam.{name}", None)
    ep.meta["episode_id"] = ep.meta["episode_id"] + "_rgbd"
    ep.meta["provenance"] += " | depth re-rendered from recorded sim_state via RCS set_replay_state (RGB-D sim)"
    ep.meta["replay_validation"] = report
    ep.meta["source"]["source_demo_id"] = rows[0]["uuid"]   # groups with the RGB-only export of the same demo
    return {"episode": ep, "report": report}
