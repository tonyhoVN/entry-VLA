"""Leakage-safe splits and train-only normalisation.

Splits are made over *groups* (scene/layout seed or source demonstration),
never over frames, so adjacent frames and descendants of the same source
demonstration (perturbation probes, re-renders) always share a split.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

SPLITS = ("train", "val", "test")


def group_key(meta: Dict[str, Any], by: str = "source_demo_id") -> str:
    """Grouping key for an episode/probe record. ``by`` options:
    ``source_demo_id`` (default; groups descendants), ``scene_seed``, ``episode_id``."""
    src = meta.get("source", {}) or {}
    if by == "source_demo_id":
        return str(src.get("source_demo_id") or meta["episode_id"])
    if by == "scene_seed":
        seed = src.get("scene_seed", meta.get("seed"))
        if seed is None:
            raise ValueError(f"{meta['episode_id']}: no scene seed recorded; split by source_demo_id instead")
        return f"{meta.get('task_id')}/seed{seed}"
    if by == "episode_id":
        return str(meta["episode_id"])
    raise ValueError(f"unknown grouping {by!r}")


def _u01(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)


def assign_splits(metas: Sequence[Dict[str, Any]], fractions=(0.8, 0.1, 0.1), by: str = "source_demo_id",
                  salt: str = "entry_vla", heldout: Optional[Dict[str, Iterable]] = None) -> Dict[str, Any]:
    """Deterministic group-hash split. ``heldout`` maps a meta field (e.g.
    ``task_id`` or ``source.placement_bin``) to values forced into a separate
    ``heldout`` split (reported separately, never tuned on)."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    edges = np.cumsum(fractions)
    heldout = {k: set(v) for k, v in (heldout or {}).items()}
    assignment: Dict[str, str] = {}
    groups: Dict[str, str] = {}
    for m in metas:
        g = group_key(m, by)
        groups[m["episode_id"]] = g
        split = None
        for field, values in heldout.items():
            node: Any = m
            for p in field.split("."):
                node = node.get(p) if isinstance(node, dict) else None
            if node in values:
                split = "heldout"
        if split is None:
            u = _u01(g, salt)
            split = SPLITS[int(np.searchsorted(edges, u, side="right"))] if u < edges[-1] else SPLITS[-1]
        assignment[m["episode_id"]] = split
    rep = {"by": by, "salt": salt, "fractions": list(fractions), "heldout": {k: sorted(map(str, v)) for k, v in heldout.items()},
           "assignment": assignment, "groups": groups}
    problems = check_leakage(assignment, groups)
    if problems:
        raise AssertionError(problems)
    return rep


def check_leakage(assignment: Dict[str, str], groups: Dict[str, str]) -> List[str]:
    seen: Dict[str, str] = {}
    problems = []
    for ep, split in assignment.items():
        g = groups[ep]
        if g in seen and seen[g] != split:
            problems.append(f"group {g} appears in both {seen[g]} and {split}")
        seen.setdefault(g, split)
    return problems


def save_splits(rep: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)


def load_splits(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


class Normalizer:
    """Mean/std normaliser fitted on TRAIN data only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, fitted_on: str, n: int):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.fitted_on = fitted_on
        self.n = n

    @classmethod
    def fit(cls, x: np.ndarray, split: str, eps: float = 1e-6) -> "Normalizer":
        if split != "train":
            raise ValueError(f"normalisation must be fitted on the train split only (got {split!r})")
        x = np.asarray(x, dtype=np.float64).reshape(-1, np.asarray(x).shape[-1])
        return cls(x.mean(0), np.maximum(x.std(0), eps), split, len(x))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x) - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z) * self.std + self.mean

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "fitted_on": self.fitted_on, "n": self.n}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Normalizer":
        return cls(np.array(d["mean"]), np.array(d["std"]), d["fitted_on"], d["n"])
