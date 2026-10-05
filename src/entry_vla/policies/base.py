"""Local-policy interface and receding-horizon chunk execution.

A local policy maps the current observation to an action chunk in the
bimanual Cartesian contract (``cartesian_chunk``: H x 14, first step relative
to the measured query pose). The runner decodes the chunk from the *same*
measured query pose, executes ``n_action_steps`` waypoints as absolute TCP
commands, then re-queries. It is the only command owner while active.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..sim.contracts import ARMS, BimanualTcpCommand, Observation
from .cartesian_chunk import ACTION_DIM, decode_bimanual_chunk, encode_bimanual_chunk, encode_state


class LocalPolicy(abc.ABC):
    """Frozen local skill policy. ``name`` and ``checkpoint_id`` version outcome labels."""

    name: str = "abstract"
    checkpoint_id: str = "none"
    is_vla: bool = False          # only real VLA checkpoints set this; baselines must not

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear observation/action history (fixed history-reset convention at handoff)."""

    @abc.abstractmethod
    def act(self, obs: Observation) -> np.ndarray:
        """Return an (H, 14) chunk encoded relative to ``obs``'s measured TCP poses."""

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "checkpoint_id": self.checkpoint_id, "is_vla": self.is_vla}


def obs_state(obs: Observation) -> np.ndarray:
    a = obs.arms
    return encode_state(a["left"].T_WE, a["right"].T_WE, a["left"].gripper_width or 0.0,
                        a["right"].gripper_width or 0.0)


class CommandArbiter:
    """Enforces one command owner per arm (planner during preparation, policy after handoff)."""

    def __init__(self):
        self.owner: Dict[str, Optional[str]] = {a: None for a in ARMS}
        self.log: List[Dict[str, Any]] = []

    def acquire(self, who: str, arms=ARMS) -> None:
        for a in arms:
            if self.owner[a] not in (None, who):
                raise RuntimeError(f"arm {a} is owned by {self.owner[a]!r}; {who!r} must wait for release")
        for a in arms:
            self.owner[a] = who
        self.log.append({"event": "acquire", "who": who, "arms": list(arms)})

    def release(self, who: str, arms=ARMS) -> None:
        for a in arms:
            if self.owner[a] == who:
                self.owner[a] = None
        self.log.append({"event": "release", "who": who, "arms": list(arms)})

    def check(self, who: str) -> None:
        bad = [a for a in ARMS if self.owner[a] != who]
        if bad:
            raise RuntimeError(f"{who!r} commanded arms {bad} it does not own ({self.owner})")


@dataclass
class RolloutResult:
    success: bool
    terminated_by: str                       # success | timeout | invalid_state | env_terminated | error
    steps: int
    sim_time_s: float
    policy_queries: int
    mean_query_latency_s: float
    final_obs: Optional[Observation] = None
    info: Dict[str, Any] = field(default_factory=dict)


def run_policy(backend, policy: LocalPolicy, obs: Observation, arbiter: CommandArbiter, max_steps: int,
               n_action_steps: int, success_fn: Callable[[Observation, Dict[str, Any]], bool],
               invalid_fn: Optional[Callable[[Observation], Optional[str]]] = None) -> RolloutResult:
    """Receding-horizon rollout with bounded steps. ``success_fn`` reads the
    step info / oracle predicates for *labelling only*; the policy never sees them."""
    arbiter.acquire("policy")
    policy.reset()
    steps, queries, lat = 0, 0, []
    t0 = obs.time
    reason = "timeout"
    success = False
    info: Dict[str, Any] = {}
    try:
        while steps < max_steps:
            q0 = time.perf_counter()
            chunk = np.asarray(policy.act(obs), dtype=np.float64).reshape(-1, ACTION_DIM)
            lat.append(time.perf_counter() - q0)
            queries += 1
            Tl, Tr, gl, gr = decode_bimanual_chunk(obs.arms["left"].T_WE, obs.arms["right"].T_WE, chunk)
            for k in range(min(n_action_steps, len(chunk))):
                arbiter.check("policy")
                cmd = BimanualTcpCommand({"left": Tl[k], "right": Tr[k]},
                                         {"left": float(np.clip(gl[k], 0, 1)), "right": float(np.clip(gr[k], 0, 1))})
                res = backend.step(cmd, obs)
                obs, steps, info = res.obs, steps + 1, res.info
                if success_fn(obs, info):
                    success, reason = True, "success"
                    break
                if invalid_fn is not None:
                    bad = invalid_fn(obs)
                    if bad:
                        reason = f"invalid_state:{bad}"
                        break
                if res.terminated or res.truncated or steps >= max_steps:
                    reason = "env_terminated" if res.terminated else ("timeout" if steps >= max_steps else "truncated")
                    break
            if reason != "timeout" or success:
                break
    finally:
        arbiter.release("policy")
    return RolloutResult(success=success, terminated_by=reason, steps=steps, sim_time_s=obs.time - t0,
                         policy_queries=queries, mean_query_latency_s=float(np.mean(lat)) if lat else 0.0,
                         final_obs=obs, info={"last_step_info_keys": sorted(info.keys())})


class MockOraclePolicy(LocalPolicy):
    """MOCK-backend stand-in for a frozen local policy (pipeline tests only).

    It reads oracle handle positions, so it is an *oracle* policy and is never
    reported as a VLA or learned-policy result."""

    name = "mock_oracle_policy"
    checkpoint_id = "mock-v1"

    def __init__(self, horizon: int = 8, step_m: float = 0.03, approach_cone_deg: float = 60.0):
        self.horizon = horizon
        self.step_m = step_m
        self.cone = np.deg2rad(approach_cone_deg)

    def reset(self) -> None:
        self._closing = 0

    def act(self, obs: Observation) -> np.ndarray:
        from ..sim.mock import MockBackend
        h = obs.oracle["handles"]
        yaw = float(np.arctan2(h["left"][1] - h["right"][1], h["left"][0] - h["right"][0])) - np.pi / 2
        targets, grips = {}, {}
        for a in ARMS:
            p = obs.arms[a].T_WE[:3, 3]
            # Only succeeds when started roughly above its handle (entry-sensitivity for tests).
            d = h[a] - p
            dn = float(np.linalg.norm(d))
            above = dn < 0.02 or np.arccos(np.clip(-d[2] / max(dn, 1e-9), -1, 1)) < self.cone
            seq, g = [], []
            cur = p.copy()
            attached = obs.oracle.get("attached", False)
            for _ in range(self.horizon):
                goal = h[a] + (np.array([0, 0, 0.12]) if attached else 0.0)
                if not above and not attached:
                    goal = p + np.array([0, 0, -0.02])      # wanders down: will time out
                step = goal - cur
                n = np.linalg.norm(step)
                cur = cur + (step if n < self.step_m else step / n * self.step_m)
                seq.append(MockBackend.downward_pose(cur, yaw))
                close = np.linalg.norm(cur - h[a]) < 0.01 or attached
                g.append(0.0 if close else 1.0)
            targets[a], grips[a] = np.stack(seq), np.array(g)
        return encode_bimanual_chunk(obs.arms["left"].T_WE, obs.arms["right"].T_WE, targets["left"],
                                     targets["right"], grips["left"], grips["right"])


class ExpertReplayPolicy(LocalPolicy):
    """Replays recorded demonstration targets. DIAGNOSTIC ONLY: never evidence
    of VLA success at perturbed entries."""

    name = "expert_replay_diagnostic"

    def __init__(self, targets_l: np.ndarray, targets_r: np.ndarray, grip_l: np.ndarray, grip_r: np.ndarray,
                 horizon: int = 8):
        self.tl, self.tr, self.gl, self.gr = targets_l, targets_r, grip_l, grip_r
        self.horizon = horizon
        self.k = 0

    def reset(self) -> None:
        self.k = 0

    def act(self, obs: Observation) -> np.ndarray:
        idx = np.minimum(np.arange(self.k, self.k + self.horizon), len(self.tl) - 1)
        self.k += self.horizon
        return encode_bimanual_chunk(obs.arms["left"].T_WE, obs.arms["right"].T_WE, self.tl[idx], self.tr[idx],
                                     self.gl[idx], self.gr[idx])
