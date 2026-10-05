# Implementation plan and dependency audit

## Plan

The pipeline is: a geometric planner runs the preparation motion, a verified stopped handoff
follows, and then a frozen local VLA runs the skill. Evaluation measures downstream autonomous
success, not just agreement with demonstrated entry poses.

| Milestone | Code deliverable | Status |
|---|---|---|
| M0 | Backend contract, DuoBench/RCS and PerAct2 adapters, capability metadata, smoke CLI | implemented; simulator runs pending (user) |
| M1 | Shared episode format, DuoBench raw / LeRobot v3 / PerAct2 readers, alignment checks, splits, RGB-D re-render | implemented; checked on synthetic fixtures + 10 real DuoBench rows (no images) |
| M2 | Skill-start annotation, overlays, overrides, stopped-entry preparation, local-policy runner, SmolVLA dataset/training/inference adapters | implemented; mock-only coverage |
| M3 | K-candidate paired entry model, min-over-K loss, offline evaluation, nearest-demo/canonical baselines | implemented; tiny CPU forward pass only |
| M4 | Probe collection, outcome labels, scorer training, joint/factorized scored selection, matched comparisons | implemented; mock-only coverage |
| M5 | Tilted dual-Franka scene spec, arrival-order scheduling, MoveIt hook, ablation configs | implemented; MoveIt untested (ROS 2 not installed) |

No milestone has been validated experimentally. All acceptance experiments are user-run.

## Environment split

The three Python stacks conflict, so the envs only share data through `entry_vla.episode/1`
directories, LeRobot v3 datasets and JSONL trial/probe records.

| Env | Python | Purpose | Setup |
|---|---|---|---|
| `duobench` | 3.11 | RCS/MuJoCo simulation, raw export, RGB-D re-render, trials | `envs/duobench/README.md` |
| `peract2` | 3.8 | CoppeliaSim 4.1 / PyRep / RLBench forks | `envs/peract2/README.md` |
| `entryvla-train` | 3.12 | lerobot 0.6.1 SmolVLA, entry model, scorer | `envs/train/README.md` |

`entry_vla` core needs only numpy and pyyaml (Python ≥3.8). Torch, pyarrow, PIL and matplotlib
are imported lazily where they are needed.

SmolVLA cannot be loaded inside the `duobench` process: lerobot needs Python ≥3.12 and RCS
extensions need 3.11. Instead, `entry-vla serve-policy` (training env) serves the frozen policy
on localhost (`entry_vla.policy_service/1`, authkey from `ENTRY_VLA_POLICY_AUTHKEY`).
Simulator-side configs then use `policy.type=remote`. The round trip is tested only with the mock
policy; whether real SmolVLA latency stays within the control period is unmeasured.

## Dependency audit (verified from source on 2026-10-05; nothing installed)

| Component | Pin | Verified facts used |
|---|---|---|
| DuoBench | `85fdb6cc0615c078af8247f49576072f1f36fa8b` (master, 2026-09-07) | 11 tasks registered as `duobench/<task>` on import; `<Task>EnvConfig().config()`; `gym.make(id, cfg=cfg)`; stage wrapper (CarryPot 3, TransferCube 4, BinSort 5 stages) |
| robot-control-stack | `c7e81ed359d95ff40713fcac37d71759ba10e8f8` (master, 2026-09-07) | obs `{left,right}.{joints,tquat(xyzw, shared base),gripper}`; frames rgb/depth with 3x4 K and world→camera extrinsics (z forward); `include_depth` gives uint16 mm (`physical_units=True`); `set_replay_state` restore |
| rcs-core (PyPI) | 0.7.2 | pins `mujoco==3.2.6`, but master pins 3.10.0: **do not mix** |
| duobench_raw (HF) | dataset sha `f94ae0f7` | parquet schema parsed; `action[k]==env_action[k-1]` (also checked on 10 real rows); no depth columns; ~30 Hz (one episode) |
| duobench LeRobot (HF) | sha `d96164d7` | v3.0, 30 fps, 16D joint state/action, 224² AV1 video |
| PerAct2 RLBench fork | markusgrotz/RLBench `8af748c51287989294e00c9c670e3330a0e35ed5` (**main**) | 13 bimanual tasks incl. `bimanual_handover_item_easy`, `bimanual_dual_push_buttons`, `bimanual_lift_tray`; 18D right-first xyzw action; `depth_in_meters=False` default; misc camera keys |
| PyRep fork | markusgrotz/PyRep `b8bd1d7a3182adcd570d001649c0849047ebf197` | negative-focal intrinsics |
| CoppeliaSim | 4.1.0 | per README |
| lerobot | 0.6.1 (tag → `7e241bd630a3719a56157a497ce5d08f244784f1`) | Python ≥3.12, torch `>=2.7,<2.12`; SmolVLA config fields; v3 dataset API; **no grad accumulation** in `lerobot-train` |
| smolvla_base | HF revision `d9f33c94` | 6D state/action, cameras `camera1..3` at 256²: the 20D/14D contract needs fine-tuning |

**Unresolved pins / facts**:
- Exact torch CUDA wheel.
- The Franka-hand gripper config type in RCS.
- Kinematic equivalence of FR3 and the Panda DH table.
- MuJoCo site names for oracle object points (read them from the task MJCF).
- moveit_py version.
- The PerAct2 dataset episode count per split, and whether the hosted pickles match the current generator.
- The CoppeliaSim timestep used for the time axis (assumed 0.05 s).

## Key design decisions

- **Frames:** `T_AB` maps B→A. W = MuJoCo world (DuoBench) / CoppeliaSim world (PerAct2). Cameras are
  OpenCV. Quaternion order is declared at every native boundary.
- **DuoBench frames:** `obs.tquat` lives in the shared Duo base frame. The exporter needs
  `T_world_shared` from the task config (`entry-vla duobench-frames`); `inspect --handeye` checks it
  (the wrist hand-eye transform must be constant).
- **Chunk encoding:** SmolVLA training re-encodes chunks from absolute targets (`target.abs`). LeRobot
  stacks per-frame actions, which would mix measured-anchored and waypoint-anchored increments.
- **Planner:** `entry_vla.jointspace_ik_interp/v2` is the simulator planner. It is not MoveIt and does
  not model object collisions. PerAct2's planning action mode also plans internally; this is recorded.
- **Labels:** outcome labels carry `label_version` (policy checkpoint, controller, reset protocol,
  timeout, planner). Scorers refuse to train on mixed versions and refuse to score under a different one.
- **Reporting:** benchmarks are never pooled (`summarize_trials` raises), and rejections and failures
  stay in the denominator.
