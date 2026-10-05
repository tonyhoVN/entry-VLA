"""SmolVLA (lerobot 0.6.1) integration for the bimanual Cartesian contract.

API facts used (read from lerobot v0.6.1 source, not executed here):
``from lerobot.policies.smolvla import SmolVLAPolicy, SmolVLAConfig``;
``SmolVLAPolicy.from_pretrained(path)``; ``policy.forward(batch) -> (loss, dict)``;
``policy.predict_action_chunk(batch) -> (B, chunk, action_dim)``;
``lerobot.policies.make_policy(cfg, ds_meta=..., rename_map=...)`` and
``make_pre_post_processors(policy_cfg=..., pretrained_path=..., ...)``;
``LeRobotDataset.create(repo_id, fps, features, root, robot_type, use_videos)``,
``add_frame(dict with 'task')``, ``save_episode()``, ``finalize()``;
config fields ``chunk_size, n_action_steps, freeze_vision_encoder, train_expert_only,
train_state_proj, use_amp, device, normalization_mapping``.

Why a custom loop: LeRobot builds an action chunk by stacking per-frame actions.
Our chunk anchors step 0 to the *measured* query pose and later steps to the
preceding *desired* waypoint, so stacking stored per-frame deltas would mix
unrelated measured-to-commanded increments. The dataset therefore stores the
absolute desired targets (``target.abs``, 20D) and the loop re-encodes chunks from
the raw measured state on the fly. Plain ``lerobot-train`` on this dataset is wrong
and is refused by the dataset's ``entry_vla`` metadata note. lerobot 0.6.1 also
lacks gradient accumulation, which the loop adds.

The checkpoint's state/action projections were trained on a 6D layout; the
20D/14D contract occupies the zero-padded 32D slots with new semantics. No
zero-shot compatibility is claimed.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from ..data.alignment import policy_aligned_targets
from ..data.episode import Episode
from ..geometry.rotations import matrix_to_rot6d, rot6d_to_matrix
from .base import LocalPolicy
from .cartesian_chunk import (ACTION_DIM, ACTION_NAMES, STATE_DIM, STATE_NAMES, encode_arm_chunk, encode_state)

LEROBOT_VERSION = "0.6.1"
TARGET_NAMES = [n.replace("gripper_width", "gripper_cmd") for n in STATE_NAMES]


def lerobot_features(cameras: Dict[str, str], image_hw: Sequence[int], use_videos: bool = True) -> Dict[str, Any]:
    """``cameras`` maps our camera name -> policy image key suffix (e.g. head -> camera1)."""
    H, W = int(image_hw[0]), int(image_hw[1])
    feats: Dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ACTION_NAMES},
        "target.abs": {"dtype": "float32", "shape": (STATE_DIM,), "names": TARGET_NAMES},
        "target.frame": {"dtype": "float32", "shape": (1,), "names": ["frame"]},
    }
    for _, key in cameras.items():
        feats[f"observation.images.{key}"] = {"dtype": "video" if use_videos else "image", "shape": (H, W, 3),
                                              "names": ["height", "width", "channels"]}
    return feats


def _target_abs(T_l: np.ndarray, T_r: np.ndarray, g_l: float, g_r: float) -> np.ndarray:
    return encode_state(T_l, T_r, g_l, g_r)   # same 20D layout; slot 10/20 = gripper command


def episode_frames(ep: Episode, cameras: Dict[str, str], image_hw: Sequence[int], start: int = 0,
                   end: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Frames for one episode (optionally a skill segment [start, end)).

    Targets: commanded ``{arm}.cmd_T_WE`` when exported, aligned so the target
    at row t is the command that FOLLOWS obs[t] (per declared alignment).
    Without commanded targets, refuse (the measured-future variant must be
    selected explicitly via ``target_source=measured_future``)."""
    from ..geometry.camera import resize_images
    al = ep.meta["alignment"]["obs_vs_action"]
    tgt, m = {}, None
    for a in ("left", "right"):
        if f"{a}.cmd_T_WE" not in ep.lowdim:
            raise KeyError(f"{ep.episode_id}: no commanded targets in W ({a}.cmd_T_WE); export T_shared_armbase "
                           "or build with target_source=measured_future (documented variant)")
        tgt[a], m_a = policy_aligned_targets(ep.lowdim, al, f"{a}.cmd_T_WE")
        tgt[f"{a}.g"], _ = policy_aligned_targets(ep.lowdim, al, f"{a}.cmd_gripper")
        m = m_a if m is None else (m & m_a)
    end = ep.num_frames if end is None else end
    for t in range(start, end):
        if not m[t] or not ep.lowdim["valid"][t]:
            continue
        Tl, Tr = ep.lowdim["left.T_WE"][t], ep.lowdim["right.T_WE"][t]
        state = encode_state(Tl, Tr, ep.lowdim["left.gripper_width"][t], ep.lowdim["right.gripper_width"][t])
        first = np.concatenate([encode_arm_chunk(Tl, tgt["left"][t][None], [tgt["left.g"][t]])[0],
                                encode_arm_chunk(Tr, tgt["right"][t][None], [tgt["right.g"][t]])[0]])
        fr = {"observation.state": state.astype(np.float32), "action": first.astype(np.float32),
              "target.abs": _target_abs(tgt["left"][t], tgt["right"][t], tgt["left.g"][t], tgt["right.g"][t]),
              "target.frame": np.array([t], np.float32), "task": ep.meta.get("instruction", "")}
        for cam, key in cameras.items():
            c = ep.cameras[cam]
            img = c["rgb"][t]
            if tuple(img.shape[:2]) != tuple(image_hw):
                img, _, _, _ = resize_images(img, None, None, c["K"], tuple(image_hw))
            fr[f"observation.images.{key}"] = img
        yield fr


def encode_chunks_from_targets(state: np.ndarray, target_abs: np.ndarray, target_frame: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
    """(B,20) raw measured state + (B,H,20) absolute targets + (B,H) frame idx
    -> (B,H,14) sequential chunks anchored at the measured pose, (B,H) pad mask.
    Padding = clamped (non-increasing) frame indices from LeRobot's boundary handling."""
    B, H, _ = target_abs.shape
    out = np.zeros((B, H, ACTION_DIM), np.float32)
    pad = np.zeros((B, H), bool)
    for b in range(B):
        f = target_frame[b].reshape(H)
        pad[b, 1:] = np.diff(f) <= 0
        pad[b] = np.logical_or.accumulate(pad[b])
        for i, sl in enumerate((slice(0, 10), slice(10, 20))):
            s = state[b, sl]
            Tq = np.eye(4); Tq[:3, :3] = rot6d_to_matrix(s[3:9]); Tq[:3, 3] = s[:3]
            tg = target_abs[b, :, sl]
            Ts = np.repeat(np.eye(4)[None], H, 0)
            Ts[:, :3, :3] = rot6d_to_matrix(tg[:, 3:9]); Ts[:, :3, 3] = tg[:, :3]
            out[b, :, 7 * i:7 * (i + 1)] = encode_arm_chunk(Tq, Ts, tg[:, 9])
    return out, pad


def compute_action_stats(chunks: np.ndarray, pad: np.ndarray, split: str) -> Dict[str, List[float]]:
    from ..data.splits import Normalizer
    valid = chunks[~pad]
    n = Normalizer.fit(valid, split)
    return n.to_dict()


def write_lerobot_dataset(episodes: Sequence[Episode], segments: Sequence[Optional[tuple]], repo_id: str, root: str,
                          fps: float, cameras: Dict[str, str], image_hw: Sequence[int], robot_type: str,
                          use_videos: bool = True) -> Dict[str, Any]:
    from lerobot.datasets import LeRobotDataset  # lerobot 0.6.1
    ds = LeRobotDataset.create(repo_id=repo_id, fps=int(round(fps)), features=lerobot_features(cameras, image_hw, use_videos),
                               root=root, robot_type=robot_type, use_videos=use_videos)
    n_frames = 0
    for ep, seg in zip(episodes, segments):
        s, e = seg if seg else (0, None)
        for fr in episode_frames(ep, cameras, image_hw, s, e):
            ds.add_frame(fr)
            n_frames += 1
        ds.save_episode()
    ds.finalize()
    note = {"entry_vla": {"warning": "train with `entry-vla train-smolvla` only: chunks must be re-encoded from "
                                     "target.abs; plain lerobot-train would stack unrelated per-frame deltas",
                          "source_episodes": [ep.episode_id for ep in episodes], "segments": list(segments),
                          "cameras": cameras, "lerobot_version": LEROBOT_VERSION}}
    with open(os.path.join(root, "meta", "entry_vla.json"), "w") as f:
        json.dump(note, f, indent=2, default=str)
    return {"root": root, "frames": n_frames, "episodes": len(episodes)}


class SmolVLAPolicyAdapter(LocalPolicy):
    """Frozen fine-tuned SmolVLA checkpoint as a LocalPolicy (user-run inference)."""

    name = "smolvla"
    is_vla = True

    def __init__(self, checkpoint: str, cameras: Dict[str, str], image_hw: Sequence[int], action_stats: Dict[str, Any],
                 device: str = "cuda:0", n_action_steps: Optional[int] = None):
        import torch
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla import SmolVLAPolicy
        self.torch = torch
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        self.pre, self.post = make_pre_post_processors(policy_cfg=self.policy.config, pretrained_path=checkpoint)
        self.cameras, self.image_hw, self.device = cameras, tuple(image_hw), device
        self.mean = np.asarray(action_stats["mean"]); self.std = np.asarray(action_stats["std"])
        self.checkpoint_id = checkpoint
        self.n_action_steps = n_action_steps

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs) -> np.ndarray:
        from ..geometry.camera import resize_images
        from .base import obs_state
        torch = self.torch
        batch = {"observation.state": torch.from_numpy(obs_state(obs)).float()[None].to(self.device),
                 "task": [obs.instruction]}
        for cam, key in self.cameras.items():
            c = obs.cameras[cam]
            img = c.rgb
            if img.shape[:2] != self.image_hw:
                img, _, _, _ = resize_images(img, None, None, c.K, self.image_hw)
            batch[f"observation.images.{key}"] = torch.from_numpy(img).permute(2, 0, 1).float()[None].div(255).to(self.device)
        batch = self.pre(batch)
        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(batch)
        chunk = chunk[0, :, :ACTION_DIM].float().cpu().numpy()
        return chunk * self.std + self.mean     # actions normalised by entry_vla train-only stats (IDENTITY in lerobot)
