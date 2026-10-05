"""PerAct2 bimanual RLBench episode directories -> shared episodes.

Layout written by ``tools/dataset_generator_bimanual.py`` (markusgrotz/RLBench@8af748c)::

    <task>/all_variations/episodes/episode<N>/
        low_dim_obs.pkl  variation_number.pkl  variation_descriptions.pkl
        <cam>_rgb/rgb_0000.png  <cam>_depth/depth_0000.png  <cam>_mask/mask_0000.png

``low_dim_obs.pkl`` is a pickled ``rlbench.demo.Demo`` of ``BimanualObservation``s with
``perception_data`` cleared. It is loaded with a restricted unpickler that maps rlbench
classes to inert stubs, so inspection does not need RLBench installed and never runs
pickled callables from those modules. Depth PNGs are 24-bit RGB-packed linear buffers;
metric = near + d * (far - near), with per-frame near/far from ``obs.misc``.

Saved observations do not contain the original commanded Cartesian actions; when present,
``misc['{arm}_executed_demo_joint_position_action']`` is exported as a commanded
joint-position target.
"""

from __future__ import annotations

import glob
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..geometry.camera import decode_depth, normalize_negative_focal, resize_images, valid_depth_mask
from ..geometry.transforms import pose_to_T
from .episode import ActionSpec, Episode, new_meta

ARMS = ("left", "right")
CAMERAS = ("front", "over_shoulder_left", "over_shoulder_right", "overhead", "wrist_left", "wrist_right")
QUAT = "xyzw"
_SAFE_PREFIXES = ("rlbench", "pyrep", "yarr")


class _Stub:
    """Inert stand-in for rlbench/pyrep classes found in pickles."""

    def __init__(self, *args, **kwargs):
        self._args = args

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state

    def __repr__(self):
        return f"<stub {type(self).__name__}>"


class _StubUnpickler(pickle.Unpickler):
    _cache: Dict[str, type] = {}

    def find_class(self, module: str, name: str):
        if module.split(".")[0] in _SAFE_PREFIXES:
            key = f"{module}.{name}"
            if key not in self._cache:
                self._cache[key] = type(name, (_Stub,), {"__module__": module})
            return self._cache[key]
        if module.split(".")[0] in ("numpy", "builtins", "collections", "copyreg", "_codecs"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a dataset pickle")


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return _StubUnpickler(f).load()


def demo_observations(demo: Any) -> List[Any]:
    """``Demo`` stores the observation list as ``_observations``."""
    for attr in ("_observations", "observations"):
        if hasattr(demo, attr):
            return list(getattr(demo, attr))
    if isinstance(demo, list):
        return demo
    raise ValueError("unrecognised Demo pickle structure")


def list_episode_dirs(task_root: str) -> List[str]:
    d = os.path.join(task_root, "all_variations", "episodes")
    eps = glob.glob(os.path.join(d, "episode*"))
    return sorted(eps, key=lambda p: int(os.path.basename(p).replace("episode", "")))


def _png(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path))


def inspect_episode_dir(ep_dir: str, max_frames: int = 10) -> Dict[str, Any]:
    obs = demo_observations(load_pickle(os.path.join(ep_dir, "low_dim_obs.pkl")))
    o0 = obs[0]
    rep: Dict[str, Any] = {"episode_dir": ep_dir, "num_frames": len(obs),
                           "obs_attributes": sorted(vars(o0).keys()),
                           "arm_attributes": sorted(vars(o0.left).keys()) if hasattr(o0, "left") else None,
                           "misc_keys": sorted(o0.misc.keys()) if isinstance(getattr(o0, "misc", None), dict) else None}
    cams = {}
    for c in CAMERAS:
        rgb_dir = os.path.join(ep_dir, f"{c}_rgb")
        if os.path.isdir(rgb_dir):
            files = sorted(os.listdir(rgb_dir))
            im = _png(os.path.join(rgb_dir, files[0]))
            dfiles = sorted(os.listdir(os.path.join(ep_dir, f"{c}_depth"))) if os.path.isdir(os.path.join(ep_dir, f"{c}_depth")) else []
            cams[c] = {"num_rgb": len(files), "rgb_shape": list(im.shape), "rgb_dtype": str(im.dtype),
                       "num_depth": len(dfiles), "first_files": files[:2]}
            if dfiles:
                dim = _png(os.path.join(ep_dir, f"{c}_depth", dfiles[0]))
                cams[c]["depth_png_shape"], cams[c]["depth_png_dtype"] = list(dim.shape), str(dim.dtype)
    rep["cameras"] = cams
    for p in ("variation_number.pkl", "variation_descriptions.pkl"):
        fp = os.path.join(ep_dir, p)
        rep[p] = load_pickle(fp) if os.path.exists(fp) else None
    rep["first_frames"] = [{"left_gripper_pose": np.asarray(o.left.gripper_pose).tolist(),
                            "right_gripper_pose": np.asarray(o.right.gripper_pose).tolist(),
                            "left_gripper_open": getattr(o.left, "gripper_open", None)}
                           for o in obs[:max_frames]]
    return rep


def export_episode_dir(ep_dir: str, task_id: str, cfg: Dict[str, Any], max_frames: Optional[int] = None) -> Episode:
    obs = demo_observations(load_pickle(os.path.join(ep_dir, "low_dim_obs.pkl")))
    if max_frames is not None:
        obs = obs[:max_frames]
    T = len(obs)
    dt = float(cfg.get("assumed_timestep_s", 0.05))
    lowdim: Dict[str, np.ndarray] = {"time": np.arange(T) * dt, "step": np.arange(T), "valid": np.ones(T, bool),
                                     "source_index": np.arange(T)}
    for a in ARMS:
        arm = [getattr(o, a) for o in obs]
        gp = np.stack([np.asarray(x.gripper_pose, dtype=np.float64) for x in arm])
        lowdim[f"{a}.T_WE"] = np.stack([pose_to_T(g[:3], g[3:7], QUAT) for g in gp])
        lowdim[f"{a}.q"] = np.stack([np.asarray(x.joint_positions, dtype=np.float64) for x in arm])
        lowdim[f"{a}.dq"] = np.stack([np.asarray(x.joint_velocities, dtype=np.float64) for x in arm])
        gjp = [getattr(x, "gripper_joint_positions", None) for x in arm]
        if all(g is not None for g in gjp):
            lowdim[f"{a}.gripper_width"] = np.array([float(np.sum(g)) for g in gjp])
        lowdim[f"{a}.gripper_native"] = np.array([float(x.gripper_open) for x in arm])
        key = f"{a}_executed_demo_joint_position_action"
        if all(isinstance(o.misc, dict) and o.misc.get(key) is not None for o in obs):
            lowdim[f"{a}.cmd_joint_position"] = np.stack([np.asarray(o.misc[key], dtype=np.float64) for o in obs])
    image_hw = cfg.get("image_hw")
    cams: Dict[str, Dict[str, Any]] = {}
    cam_meta: Dict[str, Any] = {}
    unavailable: Dict[str, str] = {}
    for c in cfg.get("cameras", ("front", "over_shoulder_left", "over_shoulder_right", "wrist_left", "wrist_right")):
        rgb_dir, depth_dir = os.path.join(ep_dir, f"{c}_rgb"), os.path.join(ep_dir, f"{c}_depth")
        if not os.path.isdir(rgb_dir):
            unavailable[f"cam.{c}"] = "camera folder absent"
            continue
        rgbs, depths, valids, Ks, Ts = [], [], [], [], []
        near_far = []
        for i, o in enumerate(obs):
            K_raw = np.asarray(o.misc[f"{c}_camera_intrinsics"], dtype=np.float64)
            T_raw = np.asarray(o.misc[f"{c}_camera_extrinsics"], dtype=np.float64)
            near, far = float(o.misc[f"{c}_camera_near"]), float(o.misc[f"{c}_camera_far"])
            K, T_WC = normalize_negative_focal(K_raw, T_raw)
            rgb = _png(os.path.join(rgb_dir, f"rgb_{i:04d}.png"))[..., :3].astype(np.uint8)
            dpng = _png(os.path.join(depth_dir, f"depth_{i:04d}.png"))[..., :3]
            d = decode_depth(dpng, "rgb24_linear_buffer", near=near, far=far)
            m = valid_depth_mask(d, near=near, far=far)
            if image_hw:
                rgb, d, m, K = resize_images(rgb, d, m, K, tuple(image_hw))
            rgbs.append(rgb); depths.append(d); valids.append(m); Ks.append(K); Ts.append(T_WC)
            near_far.append((near, far))
        Ts = np.stack(Ts)
        moving = not np.allclose(Ts, Ts[0], atol=1e-6)
        cams[c] = {"rgb": np.stack(rgbs), "depth": np.stack(depths).astype(np.float32), "depth_valid": np.stack(valids),
                   "K": Ks[0], "T_WC": Ts if moving else Ts[0]}
        cam_meta[c] = {"width": rgbs[0].shape[1], "height": rgbs[0].shape[0], "near": near_far[0][0],
                       "far": near_far[0][1], "depth_source_encoding": "rgb24_linear_buffer",
                       "moving": bool(moving), "modality": "rgbd_sim",
                       "intrinsics_note": "PyRep negative-focal K rewritten as positive focal + Rz(pi) camera frame"}
    var = load_pickle(os.path.join(ep_dir, "variation_number.pkl")) if os.path.exists(os.path.join(ep_dir, "variation_number.pkl")) else None
    desc = load_pickle(os.path.join(ep_dir, "variation_descriptions.pkl")) if os.path.exists(os.path.join(ep_dir, "variation_descriptions.pkl")) else []
    unavailable["seed"] = "scene seed not stored in PerAct2 episodes"
    unavailable["recording_rate_hz"] = "not stored; time axis assumes assumed_timestep_s (UNVERIFIED)"
    for a in ARMS:
        if f"{a}.gripper_width" not in lowdim:
            unavailable[f"{a}.gripper_width"] = "gripper_joint_positions not recorded"
    ep_id = f"peract2_{task_id}_{os.path.basename(ep_dir)}"
    meta = new_meta(
        episode_id=ep_id, backend="peract2", task_id=task_id, variation_id=None if var is None else int(var),
        instruction=desc[0] if desc else "", instruction_variants=list(desc), split=cfg.get("split", "unassigned"),
        sim_timestep_s=dt, provenance="PerAct2 simulator-generated waypoint + motion-planning demonstration "
                                      "(not human teleoperation)",
        world_frame="coppeliasim_world",
        alignment={"obs_vs_action": "unverified", "detail": "saved observations only; no original commanded "
                   "Cartesian actions", "verified_by": None},
        source={"episode_dir": ep_dir, "source_demo_id": f"{task_id}/{os.path.basename(ep_dir)}"},
        cameras=cam_meta,
        gripper={"semantics": "width_m=sum(gripper_joint_positions); native gripper_open binary", "model": "Panda hand"},
        derived={"time": f"frame index x {dt} s (assumed CoppeliaSim step)"},
        reset={"procedure": "regenerate scene with RLBench task reset + variation; demos are not physical snapshots"},
    )
    meta["unavailable"].update(unavailable)
    if any(k.endswith("cmd_joint_position") for k in lowdim):
        meta["action_specs"] = [ActionSpec(key="cmd_joint_position", kind="commanded_target",
                                           controller="RLBench executed demo joint-position action", frame="joint_space",
                                           alignment="UNVERIFIED").__dict__]
    oracle = {}
    lows = [getattr(o, "task_low_dim_state", None) for o in obs]
    lows = [l[0] if isinstance(l, tuple) and len(l) == 1 else l for l in lows]
    if all(l is not None for l in lows) and len({np.asarray(l).shape for l in lows}) == 1:
        oracle["task_low_dim_state"] = np.stack([np.asarray(l, dtype=np.float64) for l in lows])
    return Episode(meta=meta, lowdim=lowdim, cameras=cams, annotations={}, oracle=oracle)


def write_fake_episode_dir(root: str, T: int = 3, size: int = 8, cameras: Sequence[str] = ("front", "wrist_left")) -> str:
    """Write a tiny FAKE episode directory with the PerAct2 layout (tests only).

    Pickles reference rlbench class paths so the stub unpickler is exercised. The
    fake modules exist in ``sys.modules`` only while pickling, then the original
    entries are restored (a real rlbench install is never shadowed afterwards)."""
    import sys
    import types

    from PIL import Image

    from ..geometry.transforms import look_at

    names = {"rlbench": [], "rlbench.backend": [], "rlbench.demo": ["Demo"],
             "rlbench.backend.observation": ["BimanualObservation", "UnimanualObservationData"]}
    saved = {m: sys.modules.get(m) for m in names}
    classes = {}
    for m, cls_names in names.items():
        mod = types.ModuleType(m)
        for cn in cls_names:
            cls = type(cn, (), {"__module__": m})
            setattr(mod, cn, cls)
            classes[cn] = cls
        sys.modules[m] = mod
    try:
        ep = os.path.join(root, "all_variations", "episodes", "episode0")
        os.makedirs(ep, exist_ok=True)
        f = -size / (2 * np.tan(np.deg2rad(30)))
        K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])   # PyRep-style negative focal
        obs_list = []
        for t in range(T):
            o = classes["BimanualObservation"]()
            o.misc = {}
            for c in cameras:
                o.misc[f"{c}_camera_intrinsics"] = K
                o.misc[f"{c}_camera_extrinsics"] = look_at([1.0, 0.1 * t if c.startswith("wrist") else 0.0, 1.5],
                                                           [0, 0, 0.75])
                o.misc[f"{c}_camera_near"], o.misc[f"{c}_camera_far"] = 0.01, 4.5
            o.task_low_dim_state = (np.zeros(3),)
            for a, y in (("left", 0.3), ("right", -0.3)):
                arm = classes["UnimanualObservationData"]()
                arm.gripper_pose = np.array([0.3, y, 1.0 + 0.01 * t, 1.0, 0.0, 0.0, 0.0])  # xyzw: pi about x
                arm.joint_positions = np.zeros(7)
                arm.joint_velocities = np.zeros(7)
                arm.gripper_open = 1.0
                arm.gripper_joint_positions = np.array([0.04, 0.04])
                setattr(o, a, arm)
                o.misc[f"{a}_executed_demo_joint_position_action"] = np.zeros(8)
            obs_list.append(o)
            for c in cameras:
                for sub in ("rgb", "depth"):
                    os.makedirs(os.path.join(ep, f"{c}_{sub}"), exist_ok=True)
                Image.fromarray(np.full((size, size, 3), 100, np.uint8)).save(os.path.join(ep, f"{c}_rgb", f"rgb_{t:04d}.png"))
                ints = np.full((size, size), int(0.25 * (2**24 - 1)), np.uint32)
                rgb = np.stack([(ints >> 16) & 255, (ints >> 8) & 255, ints & 255], -1).astype(np.uint8)
                Image.fromarray(rgb).save(os.path.join(ep, f"{c}_depth", f"depth_{t:04d}.png"))
        demo = classes["Demo"]()
        demo._observations = obs_list
        with open(os.path.join(ep, "low_dim_obs.pkl"), "wb") as fh:
            pickle.dump(demo, fh)
    finally:
        for m, old in saved.items():
            if old is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = old
    with open(os.path.join(ep, "variation_number.pkl"), "wb") as fh:
        pickle.dump(0, fh)
    with open(os.path.join(ep, "variation_descriptions.pkl"), "wb") as fh:
        pickle.dump(["hand over the item"], fh)
    return ep
