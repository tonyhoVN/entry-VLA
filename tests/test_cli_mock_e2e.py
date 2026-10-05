"""CLI end-to-end on MOCK fixtures only (plumbing coverage, not benchmark evidence)."""
import json
import os
import subprocess
import sys

REPO = os.path.join(os.path.dirname(__file__), "..")


def cli(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    out = subprocess.run([sys.executable, "-m", "entry_vla.cli.main", *args], capture_output=True, text=True,
                         env=env, timeout=120, cwd=cwd or REPO)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)


def test_fixtures_annotate_split_rollout(tmp_path):
    root = str(tmp_path / "eps")
    r = cli("make-fixtures", "--out", root, "--n", "3")
    assert len(r["written"]) == 3
    r = cli("annotate", "--config", "configs/annotate.yaml", f"annotate.episodes_root={root}",
            "annotate.overrides=null", "annotate.params.visibility_cameras=[front]", "annotate.overlay_cameras=[front]")
    assert all(v["status"] == "proposed" for v in r["annotations"].values()), r
    r = cli("split", "--config", "configs/split.yaml", f"split.episodes_root={root}", f"split.out={root}/splits.json")
    assert sum(r["counts"].values()) == 3
    r = cli("rollout", "--config", "configs/rollout.yaml", "--max-trials", "2", f"trials.runs_root={tmp_path}/runs")
    s = r["summary"]
    assert s["requested_trials"] == 2 and s["backend"] == "mock"
    assert os.path.exists(os.path.join(r["run_dir"], "trials.jsonl"))
    assert os.path.exists(os.path.join(r["run_dir"], "manifest.json"))


def test_rollout_refuses_mock_policy_on_real_backend():
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    out = subprocess.run([sys.executable, "-m", "entry_vla.cli.main", "rollout", "--config", "configs/rollout.yaml",
                          "backend=duobench", "--dry-run"], capture_output=True, text=True, env=env, timeout=60, cwd=REPO)
    assert out.returncode != 0 and "mock_oracle" in out.stderr
