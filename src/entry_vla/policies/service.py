"""Local policy service: run a frozen policy in the training env (lerobot, Python 3.12)
and query it from a simulator env (RCS 3.11 / RLBench 3.8).

Transport: ``multiprocessing.connection`` on localhost with an authkey (stdlib,
available on Python 3.8+). Messages are plain dicts of numpy arrays/strings,
versioned with ``SERVICE_VERSION``. The client never forwards privileged
oracle data unless ``send_oracle=True`` (allowed only for the labelled mock policy).
"""

from __future__ import annotations

import time
from multiprocessing.connection import Client, Listener
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..sim.contracts import ARMS, ArmState, CameraFrame, Observation
from .base import LocalPolicy

SERVICE_VERSION = "entry_vla.policy_service/1"


def encode_observation(obs: Observation, cameras: Sequence[str], send_depth: bool = False,
                       send_oracle: bool = False) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"step": obs.step, "time": obs.time, "instruction": obs.instruction,
                           "arms": {a: {"T_WE": obs.arms[a].T_WE, "q": obs.arms[a].q, "gripper_width": obs.arms[a].gripper_width}
                                    for a in ARMS}, "cameras": {}}
    for n in cameras:
        c = obs.cameras[n]
        msg["cameras"][n] = {"rgb": c.rgb, "K": c.K, "T_WC": c.T_WC, "width": c.width, "height": c.height,
                             "moving": c.moving, "depth": c.depth if send_depth else None,
                             "depth_valid": c.depth_valid if send_depth else None}
    if send_oracle:
        msg["oracle"] = obs.oracle
    return msg


def decode_observation(msg: Dict[str, Any]) -> Observation:
    arms = {a: ArmState(T_WE=np.asarray(v["T_WE"]), q=np.asarray(v["q"]), gripper_width=v["gripper_width"])
            for a, v in msg["arms"].items()}
    cams = {n: CameraFrame(name=n, K=np.asarray(c["K"]), T_WC=np.asarray(c["T_WC"]), width=c["width"], height=c["height"],
                           rgb=c["rgb"], depth=c["depth"], depth_valid=c["depth_valid"], moving=c["moving"],
                           depth_source_encoding="metric_m" if c["depth"] is not None else "unavailable")
            for n, c in msg["cameras"].items()}
    return Observation(step=msg["step"], time=msg["time"], arms=arms, cameras=cams, instruction=msg["instruction"],
                       oracle=msg.get("oracle", {}))


def serve(policy: LocalPolicy, address: Tuple[str, int], authkey: bytes, max_requests: Optional[int] = None,
          ready=None) -> Dict[str, Any]:
    """Serve one client connection. ``max_requests`` bounds the session (None = until 'close')."""
    if address[0] not in ("127.0.0.1", "localhost"):
        raise ValueError("policy service binds to localhost only")
    n = 0
    with Listener(address, authkey=authkey) as lst:
        if ready is not None:
            ready(lst.address)
        with lst.accept() as conn:
            while max_requests is None or n < max_requests:
                req = conn.recv()
                n += 1
                if req.get("version") != SERVICE_VERSION:
                    conn.send({"error": f"version mismatch: server {SERVICE_VERSION}, client {req.get('version')}"})
                    continue
                cmd = req["cmd"]
                if cmd == "describe":
                    conn.send({"ok": True, "policy": policy.describe(), "version": SERVICE_VERSION})
                elif cmd == "reset":
                    policy.reset()
                    conn.send({"ok": True})
                elif cmd == "act":
                    t0 = time.perf_counter()
                    chunk = np.asarray(policy.act(decode_observation(req["obs"])), dtype=np.float32)
                    conn.send({"ok": True, "chunk": chunk, "server_latency_s": time.perf_counter() - t0})
                elif cmd == "close":
                    conn.send({"ok": True})
                    break
                else:
                    conn.send({"error": f"unknown cmd {cmd!r}"})
    return {"requests": n}


class RemotePolicy(LocalPolicy):
    """Client-side LocalPolicy forwarding to a policy service."""

    def __init__(self, address: Tuple[str, int], authkey: bytes, cameras: Sequence[str], send_depth: bool = False,
                 send_oracle: bool = False, timeout_s: float = 30.0):
        self.conn = Client(address, authkey=authkey)
        self.cameras, self.send_depth, self.send_oracle, self.timeout = list(cameras), send_depth, send_oracle, timeout_s
        info = self._call({"cmd": "describe"})
        self.remote = info["policy"]
        self.name = f"remote:{self.remote['name']}"
        self.checkpoint_id = self.remote["checkpoint_id"]
        self.is_vla = bool(self.remote["is_vla"])

    def _call(self, req: Dict[str, Any]) -> Dict[str, Any]:
        self.conn.send({**req, "version": SERVICE_VERSION})
        if not self.conn.poll(self.timeout):
            raise TimeoutError("policy service did not answer in time")
        rep = self.conn.recv()
        if "error" in rep:
            raise RuntimeError(rep["error"])
        return rep

    def reset(self) -> None:
        self._call({"cmd": "reset"})

    def act(self, obs: Observation) -> np.ndarray:
        rep = self._call({"cmd": "act", "obs": encode_observation(obs, self.cameras, self.send_depth, self.send_oracle)})
        return rep["chunk"]

    def close(self) -> None:
        try:
            self._call({"cmd": "close"})
        finally:
            self.conn.close()
