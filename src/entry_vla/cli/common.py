"""Shared CLI helpers: config loading, bounded integer options, dry-run output."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from typing import Any, Dict, Iterable

from ..utils.config import load_config


def add_common(p: argparse.ArgumentParser, default_config: str) -> None:
    p.add_argument("--config", default=default_config, help=f"YAML config (default: {default_config})")
    p.add_argument("--seed", type=int, default=None, help="override cfg.seed")
    p.add_argument("--dry-run", action="store_true",
                   help="validate config/schema only: no weights, dataset iteration, simulator or optimisation")
    p.add_argument("overrides", nargs="*", help="key=value overrides, e.g. backend=peract2 train.batch_size=4")


def bounded_int(lo: int, hi: int):
    def parse(s: str) -> int:
        v = int(s)
        if not lo <= v <= hi:
            raise argparse.ArgumentTypeError(f"must be in [{lo}, {hi}], got {v}")
        return v
    return parse


def load(args) -> Dict[str, Any]:
    cfg = load_config(args.config, args.overrides)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("seed", 0)
    return cfg


def module_available(names: Iterable[str]) -> Dict[str, bool]:
    """find_spec on top-level packages only (does not import/initialise them)."""
    out = {}
    for n in names:
        try:
            out[n] = importlib.util.find_spec(n) is not None
        except (ImportError, ValueError):
            out[n] = False
    return out


def emit(obj: Any) -> None:
    json.dump(obj, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def fail(msg: str, code: int = 2) -> None:
    sys.stderr.write(f"error: {msg}\n")
    raise SystemExit(code)
