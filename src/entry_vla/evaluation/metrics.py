"""Evaluation metrics over ALL requested trials (failures and rejections stay in the denominator)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from ..geometry.rotations import geodesic_angle
from .failures import MANIPULATION, REJECTION, TRANSIT


def wilson_interval(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> List[float]:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize_trials(records: List[Dict[str, Any]], requested: Optional[int] = None, time_cap_s: Optional[float] = None
                     ) -> Dict[str, Any]:
    """Aggregate one condition (one benchmark, one selector). Never pool backends."""
    backends = {r["label_version"]["backend"] for r in records}
    if len(backends) > 1:
        raise ValueError(f"refusing to pool success rates across benchmarks: {backends}")
    n = requested if requested is not None else len(records)
    if len(records) > n:
        raise ValueError("more records than requested trials")
    missing = n - len(records)   # crashed/unrecorded trials count as failures
    labels = Counter(r["outcome"]["label"] for r in records)
    labels["missing_record"] += missing
    k = labels.get("success", 0)
    cap = time_cap_s
    times, succ_times, prep_t, queries, lat = [], [], [], [], []
    for r in records:
        o = r["outcome"]
        tt = float(o.get("prep_sim_time_s", 0.0)) + float(o.get("skill_sim_time_s", 0.0))
        if cap is not None and not o["success"]:
            tt = cap
        times.append(min(tt, cap) if cap is not None else tt)
        if o["success"]:
            succ_times.append(tt)
        prep_t.append(float(o.get("prep_sim_time_s", 0.0)))
        if "policy_queries" in o:
            queries.append(o["policy_queries"]); lat.append(o["mean_query_latency_s"])
    times += [cap] * missing if cap is not None else []
    accepted = sum(1 for r in records if r["outcome"]["label"] not in REJECTION)
    clear = [r["arrival"]["min_pair_clearance"] for r in records
             if r.get("arrival") and r["arrival"].get("min_pair_clearance") is not None]
    arr_err = [max(r["arrival"]["arrival_pos_err"].values()) for r in records
               if r.get("arrival") and r["arrival"].get("arrival_pos_err")]
    return {
        "backend": next(iter(backends)) if backends else None,
        "requested_trials": n, "recorded_trials": len(records),
        "success_rate": k / n if n else 0.0, "success_ci95_wilson": wilson_interval(k, n),
        "label_counts": dict(labels),
        "transit_failures": sum(labels[l] for l in TRANSIT), "manipulation_failures": sum(labels[l] for l in MANIPULATION),
        "acceptance_coverage": accepted / n if n else 0.0,
        "capped_episode_time_s_mean": float(np.mean(times)) if times else None,
        "capped_episode_time_ci95": bootstrap_mean_ci(np.array(times)) if times else None,
        "time_cap_s": cap,
        "successful_completion_time_s_mean": float(np.mean(succ_times)) if succ_times else None,
        "prep_time_s_mean": float(np.mean(prep_t)) if prep_t else None,
        "policy_queries_mean": float(np.mean(queries)) if queries else None,
        "policy_latency_s_mean": float(np.mean(lat)) if lat else None,
        "min_pair_clearance_m": float(np.min(clear)) if clear else None,
        "arrival_pos_err_m_mean": float(np.mean(arr_err)) if arr_err else None,
        "seeds": sorted(r["seed"] for r in records),
        "note": "rollout success estimates are not formal safety guarantees",
    }


def entry_pose_errors(pred_T: np.ndarray, true_T: np.ndarray) -> Dict[str, float]:
    """(N,4,4) predicted vs reference entry poses -> translation (m) / rotation (rad) errors."""
    dp = np.linalg.norm(pred_T[:, :3, 3] - true_T[:, :3, 3], axis=1)
    dr = geodesic_angle(pred_T[:, :3, :3], true_T[:, :3, :3])
    return {"pos_err_m_mean": float(dp.mean()), "pos_err_m_median": float(np.median(dp)),
            "rot_err_rad_mean": float(dr.mean()), "rot_err_rad_median": float(np.median(dr)), "n": int(len(dp))}
