"""Minimal YAML config composition.

* A config may declare ``defaults: {group: option}``; each entry loads
  ``configs/<group>/<option>.yaml`` into key ``group`` (then the file's own
  keys are merged on top).
* CLI overrides: ``group=option`` re-selects a group when
  ``configs/<group>/<option>.yaml`` exists, otherwise ``a.b.c=value`` sets a
  dotted key (value parsed as YAML).
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, Optional

import yaml

CONFIG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs"))


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _group_file(root: str, group: str, option: str) -> str:
    return os.path.join(root, group, f"{option}.yaml")


def set_dotted(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def get_dotted(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def load_config(path: Optional[str], overrides: Iterable[str] = (), root: str = CONFIG_ROOT) -> Dict[str, Any]:
    raw = _load_yaml(path) if path else {}
    defaults = dict(raw.pop("defaults", {}) or {})
    overrides = list(overrides)
    dotted = []
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override {ov!r} must look like key=value")
        k, v = ov.split("=", 1)
        if "." not in k and os.path.exists(_group_file(root, k, v)):
            defaults[k] = v
        else:
            dotted.append((k, yaml.safe_load(v)))
    cfg: Dict[str, Any] = {}
    for group, option in defaults.items():
        f = _group_file(root, group, option)
        if not os.path.exists(f):
            raise FileNotFoundError(f"config group {group}={option}: {f} not found")
        cfg[group] = _load_yaml(f)
    cfg = deep_merge(cfg, raw)
    for k, v in dotted:
        set_dotted(cfg, k, v)
    cfg.setdefault("_meta", {})["selected_groups"] = defaults
    cfg["_meta"]["source"] = os.path.abspath(path) if path else None
    cfg["_meta"]["overrides"] = overrides
    return cfg


def dump_config(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
