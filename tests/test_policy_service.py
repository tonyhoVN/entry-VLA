import threading

import numpy as np

from entry_vla.policies.base import MockOraclePolicy
from entry_vla.policies.service import RemotePolicy, decode_observation, encode_observation, serve
from entry_vla.sim.mock import MockBackend


def test_remote_policy_roundtrip_matches_local():
    be = MockBackend({"name": "mock"}); be.make("mock_bimanual_reach")
    obs = be.reset(seed=3)
    local = MockOraclePolicy()
    local.reset()
    ref = local.act(obs)
    addr = {}
    ready = threading.Event()

    def on_ready(a):
        addr["a"] = a
        ready.set()
    th = threading.Thread(target=serve, args=(MockOraclePolicy(), ("127.0.0.1", 0), b"k"), kwargs={"ready": on_ready,
                                                                                                   "max_requests": 10})
    th.start()
    assert ready.wait(10)
    rp = RemotePolicy(addr["a"], b"k", cameras=["front"], send_oracle=True)   # oracle forwarding: mock-only
    rp.reset()
    out = rp.act(obs)
    rp.close()
    th.join(10)
    assert np.allclose(out, ref, atol=1e-5)
    assert rp.name == "remote:mock_oracle_policy" and rp.is_vla is False


def test_encode_excludes_oracle_by_default():
    be = MockBackend({"name": "mock"}); be.make("mock_bimanual_reach")
    obs = be.reset(seed=0)
    msg = encode_observation(obs, ["front"])
    assert "oracle" not in msg and msg["cameras"]["front"]["depth"] is None
    o2 = decode_observation(msg)
    assert o2.oracle == {} and np.allclose(o2.arms["left"].T_WE, obs.arms["left"].T_WE)
