# DuoBench / RCS simulator environment (user-run setup)

Nothing here has been installed or executed by the agent.

| Component | Pin | Source |
|---|---|---|
| Python | 3.11 (RCS extensions do not build on 3.12+) | RCS README |
| robot-control-stack (`rcs-core`) | commit `c7e81ed359d95ff40713fcac37d71759ba10e8f8` (master, 2026-09-07) | github.com/RobotControlStack/robot-control-stack |
| duobench | commit `85fdb6cc0615c078af8247f49576072f1f36fa8b` (master, 2026-09-07) | github.com/RobotControlStack/duobench |
| mujoco / pin / gymnasium | resolved by RCS source pin (master pyproject: `mujoco==3.10.0`, `pin==3.7.0`, `gymnasium~=0.29.1`) | RCS pyproject |

The PyPI `rcs-core==0.7.2` wheel pins `mujoco==3.2.6`, unlike RCS master. Use the source pins
above together, and do not mix the PyPI wheel with DuoBench master.

```bash
conda create -n duobench python=3.11 -y && conda activate duobench
conda install -c conda-forge urdfdom urdfdom_headers glfw -y   # or: sudo apt install build-essential gcc libglfw3-dev liburdfdom-dev
git clone https://github.com/RobotControlStack/robot-control-stack.git ext/robot-control-stack
git -C ext/robot-control-stack checkout c7e81ed359d95ff40713fcac37d71759ba10e8f8
pip install 'pip>=25.1'
(cd ext/robot-control-stack && pip install --group build_deps && pip install -ve . --no-build-isolation)
git clone https://github.com/RobotControlStack/duobench.git ext/duobench
git -C ext/duobench checkout 85fdb6cc0615c078af8247f49576072f1f36fa8b
pip install -ve ext/duobench
pip install -e .[test,io]            # this repo (core is numpy/pyyaml/scipy)
pip install pyarrow pillow           # raw parquet inspection/export
```

**Implicit downloads:** importing DuoBench/RCS fetches assets into `~/.duobench` and `~/.rcs`
(override with `DUOBENCH_PREFIX` / `RCS_PREFIX`). Trigger this deliberately once, e.g.
`python -c "from duobench.tasks import carry_pot"`, before you run any smoke test.

## Data (selective download; not executed)

Raw recordings are one large parquet per task/split (5-38 GB each). Fetch a single file:

```bash
huggingface-cli download RobotControlStack/duobench_raw carry_pot/sim/2026-05-15_part-0_no_rotation.parquet \
  --repo-type dataset --local-dir data/raw/duobench_raw
# reduced LeRobot export (joint-space actions, 224x224 AV1 video), one task/split:
huggingface-cli download RobotControlStack/duobench --repo-type dataset --include "carry_pot/sim/*" \
  --local-dir data/raw/duobench_lerobot
```

Raw schema (parsed from the parquet footer): `obs.{left,right}.{tquat,joints,xyzrpy,gripper}`,
`obs.frames.{head,left_wrist,right_wrist}.rgb.{data(JPEG),intrinsics,extrinsics}`,
`info.{arm}.{sim_state,sim_state_schema,absolute_action,is_grasped,gripper_width,...}`,
`action.*` (= a_{k-1}, delayed copy), `env_action.*` (= a_k applied at this step), `uuid`, `step`,
`instruction`, `timestamp`. **No depth columns.** For RGB-D, re-render with `include_depth=true`
(see `entry-vla replay-render`).

## Smoke (user-run)

```bash
entry-vla smoke --config configs/smoke.yaml backend=duobench --out runs/smoke_duobench
pytest -m sim --run-sim tests/test_backends_offline.py::test_duobench_live_smoke
```
