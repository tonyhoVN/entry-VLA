"""Contact-event phase splitting (end-effector vs environment).

Per frame, a contact score is computed from point clouds (see scripts/contact_phase_split.py):
``n_close(t)`` = number of environment points within ``eps`` of any end-effector point, and
``d_min(t)`` = their minimum distance. A frame is in contact when ``n_close >= k``. Contact frames
are grouped into intervals (gaps <= ``merge_gap`` filled, intervals shorter than ``min_len`` dropped),
and phase boundaries are the interval onsets/offsets. Numpy only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def close_counts(hand: np.ndarray, env: np.ndarray, eps: float) -> Tuple[int, float]:
    """(#env points within eps of the hand cloud, min hand-env distance); (0, inf) if either is empty."""
    if len(hand) == 0 or len(env) == 0:
        return 0, float("inf")
    from scipy.spatial import cKDTree
    d, _ = cKDTree(hand).query(env, k=1, distance_upper_bound=max(eps, 0.5))
    return int((d <= eps).sum()), float(d.min())


def contact_intervals(flags: Sequence[bool], min_len: int = 2, merge_gap: int = 2) -> List[Tuple[int, int]]:
    """Half-open [start, end) runs of True, gaps <= merge_gap merged, runs < min_len dropped."""
    f = np.asarray(flags, bool)
    runs, t, T = [], 0, len(f)
    while t < T:
        if f[t]:
            s = t
            while t < T and f[t]:
                t += 1
            runs.append([s, t])
        else:
            t += 1
    merged: List[List[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(s, e) for s, e in merged if e - s >= min_len]


def phases_from_contacts(intervals: Sequence[Tuple[int, int]], T: int, mode: str = "first",
                         pre: int = 0) -> List[Dict]:
    """Phase list. ``first``: [free motion | from first contact onset]; ``all``: alternating
    free / contact segments at every onset and offset. ``pre`` moves every onset ``pre`` frames earlier
    (clamped to 0 and to the previous interval's end), so a contact phase starts before the contact."""
    if pre:
        shifted, prev = [], 0
        for s, e in intervals:
            shifted.append((max(prev, s - pre), e))
            prev = e
        intervals = shifted
    if not intervals:
        return [{"phase": "free", "start": 0, "end": T - 1}]
    if mode == "first":
        s = intervals[0][0]
        out = [{"phase": "free", "start": 0, "end": s - 1}] if s > 0 else []
        return out + [{"phase": "contact_onward", "start": s, "end": T - 1}]
    out, t = [], 0
    for s, e in intervals:
        if s > t:
            out.append({"phase": "free", "start": t, "end": s - 1})
        out.append({"phase": "contact", "start": s, "end": e - 1})
        t = e
    if t < T:
        out.append({"phase": "free", "start": t, "end": T - 1})
    return out


def first_onset(intervals: Sequence[Tuple[int, int]]) -> Optional[int]:
    return int(intervals[0][0]) if intervals else None
