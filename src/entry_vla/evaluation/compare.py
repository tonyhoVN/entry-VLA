"""Matched-policy selector comparisons on identical seeds (one benchmark at a time)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from .metrics import summarize_trials


def check_matched(records_by_selector: Dict[str, List[Dict[str, Any]]]) -> None:
    """Identical frozen policy/controller/reset/timeout and identical seeds across selectors."""
    keys, seeds = set(), set()
    for recs in records_by_selector.values():
        for r in recs:
            lv = dict(r["label_version"])
            lv.pop("timeout_steps", None)   # full-episode baseline uses its own documented step budget
            keys.add(json.dumps(lv, sort_keys=True))
        seeds.add(tuple(sorted(r["seed"] for r in recs)))
    if len(keys) != 1:
        raise ValueError("selectors were not run with the same frozen policy/controller/protocol")
    if len(seeds) != 1:
        raise ValueError("selectors were not evaluated on identical seeds")


def paired_success_diff(a: List[Dict[str, Any]], b: List[Dict[str, Any]], n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    sa = {r["seed"]: float(r["outcome"]["success"]) for r in a}
    sb = {r["seed"]: float(r["outcome"]["success"]) for r in b}
    common = sorted(set(sa) & set(sb))
    d = np.array([sa[s] - sb[s] for s in common])
    if len(d) == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1)
    return {"n": len(d), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "a_only_success": int((d > 0).sum()), "b_only_success": int((d < 0).sum())}


def compare(records_by_selector: Dict[str, List[Dict[str, Any]]], reference: str, requested: int,
            time_cap_s: float | None, budgets: Dict[str, Any]) -> Dict[str, Any]:
    check_matched(records_by_selector)
    out = {"reference": reference, "budgets": budgets, "summaries": {}, "paired_vs_reference": {}}
    for name, recs in records_by_selector.items():
        out["summaries"][name] = summarize_trials(recs, requested=requested, time_cap_s=time_cap_s)
        if name != reference:
            out["paired_vs_reference"][name] = paired_success_diff(recs, records_by_selector[reference])
    return out
