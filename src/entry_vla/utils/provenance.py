"""Run provenance: seeds, file checksums, source commits, run directories."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
import subprocess
from typing import Any, Dict, Optional

import numpy as np


def seed_everything(seed: int) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    info = {"seed": seed, "torch": False}
    try:
        import torch  # optional
        torch.manual_seed(seed)
        info["torch"] = True
    except ImportError:
        pass
    return info


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir or os.getcwd(), capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def git_dirty(repo_dir: Optional[str] = None) -> Optional[bool]:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir or os.getcwd(), capture_output=True,
                             text=True, timeout=5)
        return bool(out.stdout.strip())
    except Exception:
        return None


def make_run_dir(root: str, name: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = os.path.join(root, f"{stamp}_{name}")
    os.makedirs(d, exist_ok=False)
    return d


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def run_manifest(cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "created": _dt.datetime.now().isoformat(),
        "entry_vla_commit": git_commit(os.path.dirname(__file__)),
        "entry_vla_dirty": git_dirty(os.path.dirname(__file__)),
        "config": cfg,
        **(extra or {}),
    }
