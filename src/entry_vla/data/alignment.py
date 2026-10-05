"""Temporal alignment: obs/action ordering, command lag, variable spacing, padding.

Never assume alignment: each exporter declares ``meta.alignment.obs_vs_action``
("obs_before_action" | "obs_after_action" | "unverified") and these tools check it.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..sim.contracts import ARMS, BimanualTcpCommand


def estimate_target_lag(p_meas: np.ndarray, p_cmd: np.ndarray, lags: Iterable[int] = range(-3, 6)) -> Dict[str, Any]:
    """Mean distance between measured position ``t+L`` and commanded target ``t``.

    The minimising lag indicates after how many frames a command is reflected
    in the measurement (0 => obs row already reflects its own command, i.e.
    obs-after-action with a fast tracker)."""
    p_meas, p_cmd = np.asarray(p_meas), np.asarray(p_cmd)
    T = len(p_meas)
    errs = {}
    for L in lags:
        lo, hi = max(0, -L), min(T, T - L)
        if hi - lo < 2:
            continue
        errs[int(L)] = float(np.mean(np.linalg.norm(p_meas[lo + L:hi + L] - p_cmd[lo:hi], axis=1)))
    best = min(errs, key=errs.get) if errs else None
    return {"errors_m": errs, "best_lag": best}


def live_one_step_diagnostic(backend, seed: int, delta_m: float = 0.02, arm: str = "left") -> Dict[str, Any]:
    """Bounded (1 reset + 2 steps) check of whether ``step(a_t)`` returns an
    observation taken before or after ``a_t`` takes effect.

    Commands a small +z TCP offset for one arm, holds the other, and measures
    displacement in the returned observation and in the next one.
    """
    obs0 = backend.reset(seed=seed)
    T0 = {a: obs0.arms[a].T_WE.copy() for a in ARMS}
    tgt = {a: T0[a].copy() for a in ARMS}
    tgt[arm][2, 3] += delta_m
    cmd = BimanualTcpCommand(tgt, {a: 1.0 for a in ARMS})
    r1 = backend.step(cmd, obs0)
    r2 = backend.step(cmd, r1.obs)
    d1 = float(r1.obs.arms[arm].T_WE[2, 3] - T0[arm][2, 3])
    d2 = float(r2.obs.arms[arm].T_WE[2, 3] - T0[arm][2, 3])
    if abs(d1) >= 0.3 * delta_m:
        verdict = "obs_after_action"
    elif abs(d2) >= 0.3 * delta_m:
        verdict = "obs_lags_action_by_one_step"
    else:
        verdict = "inconclusive (tracking slower than 2 steps or command rejected)"
    return {"delta_m": delta_m, "arm": arm, "dz_after_step1_m": d1, "dz_after_step2_m": d2, "verdict": verdict,
            "steps_executed": 2}


def uniform_resample_indices(times: np.ndarray, rate_hz: float, max_gap_s: Optional[float] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-frame indices on a uniform grid + validity mask.

    Grid points whose nearest frame is further than ``max_gap_s`` (default
    one grid period) are masked invalid rather than silently duplicated."""
    t = np.asarray(times, dtype=np.float64)
    period = 1.0 / rate_hz
    grid = np.arange(t[0], t[-1] + 1e-9, period)
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    prev = np.clip(idx - 1, 0, len(t) - 1)
    choose_prev = np.abs(t[prev] - grid) < np.abs(t[idx] - grid)
    idx = np.where(choose_prev, prev, idx)
    tol = period if max_gap_s is None else max_gap_s
    valid = np.abs(t[idx] - grid) <= tol * 0.5 + 1e-9
    return idx, valid


def pad_stack(seqs: Sequence[np.ndarray], length: Optional[int] = None, pad_value: float = 0.0
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Stack variable-length sequences into (B, L, ...) with a (B, L) bool mask."""
    L = length or max(len(s) for s in seqs)
    tail = np.asarray(seqs[0]).shape[1:]
    out = np.full((len(seqs), L) + tail, pad_value, dtype=np.asarray(seqs[0]).dtype)
    mask = np.zeros((len(seqs), L), dtype=bool)
    for i, s in enumerate(seqs):
        n = min(len(s), L)
        out[i, :n] = np.asarray(s)[:n]
        mask[i, :n] = True
    return out, mask


def policy_aligned_targets(ep_lowdim: Dict[str, np.ndarray], alignment: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-frame "action that follows obs[t]" for a stored action stream
    plus a validity mask, given the episode's declared alignment.

    * obs_before_action: action[t] follows obs[t] -> identity.
    * obs_after_action: row t's action was applied *before* obs[t], so the
      action following obs[t] is row t+1; the last frame has none (masked).
    """
    a = np.asarray(ep_lowdim[key])
    T = len(a)
    if alignment == "obs_before_action":
        return a, np.ones(T, bool)
    if alignment == "obs_after_action":
        out = np.concatenate([a[1:], a[-1:]], axis=0)
        mask = np.ones(T, bool)
        mask[-1] = False
        return out, mask
    raise ValueError(f"alignment {alignment!r} is not verified; run the alignment diagnostic before training")
