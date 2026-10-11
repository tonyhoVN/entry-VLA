# Implementation status

Last updated: 2026-10-05. Consult this file before exploring the repo again.

## Milestones

| | Code | Verified by agent | Pending (user-run) |
|---|---|---|---|
| M0 | `sim/{contracts,base,registry,duobench,peract2,mock,validate}.py`, `cli/smoke.py` | offline adapter tests with fake native payloads; smoke on the **mock** backend; dry-runs | live DuoBench and PerAct2 smoke (one reset + ≤5 steps) |
| M1 | `data/{episode,duobench_raw,duobench_replay,lerobot_v3,peract2_raw,alignment,splits,synthetic}.py`, `cli/data_cmds.py` | synthetic fixtures; a fake PerAct2 episode dir; a fake DuoBench parquet; **10 real DuoBench raw rows** (alignment only, see below) | `duobench-frames`, `inspect --handeye`, export, replay-render, live alignment check |
| M2 | `data/{segmentation,overlays}.py`, `sim/preparation.py`, `policies/{base,cartesian_chunk,smolvla_adapter,train_smolvla,service}.py`, `evaluation/{selectors,trials,metrics,failures,factory}.py`, `cli/policy_cmds.py` | mock-only: annotation, preparation, trials, policy service round trip, SmolVLA frame and chunk consistency (numpy) | policy-dataset, train-smolvla, serve-policy, canonical rollouts |
| M3 | `models/{entry_proposal,text,train_entry}.py`, `data/entry_dataset.py`, `cli/model_cmds.py` | tiny random-weight CPU forward pass (torch 2.5.1, `xvla` env); loss and decoding unit tests | tiny-overfit and full training; eval-entry |
| M4 | `evaluation/{probes,compare}.py`, `models/{scorer,train_scorer}.py`, `cli/eval_cmds.py` | mock-only probe records, scorer arrays, matched comparison | collect-probes, train-scorer, compare on simulators |
| M5 | `sim/{custom_scene,moveit_planner}.py`, `configs/scene/`, `configs/ablation/`, `scripts/ablation_args.py` | tilted-mount geometry/gravity tests; sequential arrival order on mock; every ablation condition applies as overrides | building the custom RCS scene, MoveIt runs, the ablations |

No experimental milestone has been validated, and no benchmark results exist.

## Commands actually executed

- 2026-10-05: created conda env `entryvla` (`envs/train/environment.yml`): Python 3.12, torch 2.11.0+cu128
  (CUDA available on the RTX 4090), lerobot[smolvla,dataset] 0.6.1, transformers 5.5.4, numpy 2.2.6,
  entry-vla installed editable with extras test,train,io,viz. `pip check` reports no conflicts. In that env:
  `pytest`: **92 passed, 8 skipped**; `entry-vla check-models` works; `train-smolvla --dry-run` finds
  lerobot and torch. No weights were downloaded and no GPU workload was run.

- `python3 -m pytest -q` (system Python 3.10, numpy 1.21, no torch/pyarrow): **87 passed, 10 skipped**
  (skipped: torch, pyarrow and opt-in simulator tests).
- `PYTHONPATH=<scratch pytest shim> ~/anaconda3/envs/xvla/bin/python -m pytest -q -p no:cacheprovider -p no:anyio`
  (torch 2.5.1, numpy 1.26, pyarrow 20): **92 passed, 8 skipped** (skipped: opt-in simulator tests).
  The shim is a scratch directory of symlinks to the system pytest packages; nothing was installed.
- `--dry-run` for all 18 subcommands. All exit 0 except `train-scorer`, which reports the placeholder
  probe file as missing (expected).
- `entry-vla check-models` (one CPU forward pass with random weights, 0 optimizer steps).
- One inspection of 10 rows from a DuoBench raw parquet sample (`carry_pot/sim`, one episode, image
  columns stripped by the research fetch):
  - frame spacing 33.4 ms, i.e. about 30 Hz;
  - `action[k] == env_action[k-1]` for 9 of 9 pairs, and never equal to the same-row action;
  - `info.gripper_width == obs.gripper` (normalised, not metres);
  - `env_action.tquat` is not in the frame of `obs.tquat`.
- Upstream source was read through the web at the pins listed in `docs/PLAN.md`. Nothing was
  installed or downloaded into the project, and no simulator was started.

## Decisions

The rationale is in `docs/PLAN.md`.
- DuoBench W is the MuJoCo world. `T_world_shared` is required, and `inspect --handeye` validates it.
- RCS `set_replay_state` is used for RGB-D re-render and diagnostic restore. It restores joint-level
  state only, so the replay is validated on every frame.
- SmolVLA chunks are re-encoded from `target.abs` in a custom loop that adds gradient accumulation.
  Running plain `lerobot-train` on these datasets is wrong.
- The simulator planner is `entry_vla.jointspace_ik_interp/v2` (via-point retries, sequential arrival
  order). It is not MoveIt.
- The PerAct2 adapter uses `BimanualMoveArmThenGripper(EndEffectorPoseViaPlanning, Discrete)`, which
  also plans internally.
- Simulator and training envs are separated. They exchange episode files, LeRobot datasets, JSONL
  records and the localhost policy service.

## Blockers and unverified items

- No simulator stack, CoppeliaSim, lerobot, ROS 2 or MoveIt is installed. Every simulator and
  training path is unexecuted.
- To fill in: RCS Franka-hand gripper config type; MuJoCo site names for oracle object points;
  FR3-vs-Panda DH equivalence and the Robotiq TCP offset for the planner arm models.
- To check on first user run: lerobot call details flagged UNVERIFIED in `policies/train_smolvla.py`;
  the moveit_py API version.
- RCS restore covers joint-level state only. Contacts, controller targets and task bookkeeping are
  not guaranteed, which matters for mid-grasp restoration (`tests/test_sim_integration.py`).
- The PerAct2 time axis assumes a 0.05 s step.

## Pending user-run commands (in order)

1. Create the simulator envs (`envs/duobench/README.md`, `envs/peract2/README.md`). The training env `entryvla` already exists.
2. `entry-vla smoke --config configs/smoke.yaml backend=duobench --out runs/smoke_duobench`, and the same with `backend=peract2` under `xvfb-run -a`.
3. Download one DuoBench parquet and one PerAct2 val squashfs (commands are in the env READMEs).
4. `entry-vla duobench-frames --config configs/export.yaml --allow-asset-import`, then `entry-vla inspect --config configs/export.yaml --handeye`.
5. `entry-vla export ...`, `entry-vla replay-render ... --max-episodes 1`, `entry-vla check-alignment --live backend=duobench`.
6. `entry-vla split`, `entry-vla annotate --overlays` (review the overlays, then add manual overrides).
7. `entry-vla policy-dataset`, `entry-vla train-smolvla` (check `--dry-run` first), `entry-vla serve-policy`, `entry-vla rollout`.
8. `entry-vla train-entry train.tiny_overfit=true train.steps=300`, then full training and `eval-entry`.
9. `entry-vla collect-probes`, `entry-vla train-scorer`, `entry-vla compare`, then the ablations.
10. `pytest --run-sim -m sim` once the backends are installed.

## CodeGraph

Indexed with `codegraph sync .` after each milestone commit and before handoff. The index covers
source, configs and docs only; `data/`, `runs/`, `.codegraph/` and the envs are excluded or empty.

## VLABench backend (contact phase split), 2026-10-09
- Source: `ext/VLABench` @ cf588fe (OpenMOSS/VLABench), `ext/rrt-algorithms` @ 14e4796 (editable; the
  non-editable git install lacks subpackages). Assets (12 GB, Google Drive via `scripts/download_assets.py`)
  extracted into `ext/VLABench/VLABench/assets`.
- Env `vlabench` (py3.10): mujoco 3.2.2, dm_control 1.0.22, gym 0.26.2, gymnasium 0.29.1, open3d 0.18.0,
  numpy 1.25.0, h5py 3.11.0, scipy 1.14.0, mediapy 1.2.0, openai (import only); lerobot/tensorflow skipped.
  `pip install -e ext/VLABench --no-deps`. `entryvla` env got h5py for the split.
- `scripts/vlabench_generator.py`: one expert demo per process -> `data/raw/vlabench/<task>/seed<S>/episodes/episode<N>/data.hdf5`
  (RGB, metric depth, geom-id segmentation, K, T_WC OpenCV, TCP T_W, MuJoCo finger/robot contacts, skill ends).
  4 cameras at 480 px: right, left, forward (static) + Franka wrist cam (moving, excluded from the split).
- `scripts/contact_phase_split_vlabench.py`, `scripts/run_vlabench_contact.sh` (timeout + retries per demo).
- Short (2-skill) tasks: select_fruit, select_chemistry_tube, select_poker, select_book, select_mahjong.
  Ran 1 seed-42 episode each (all succeeded first try); splits in `data/entry_process/vlabench_short5_seed42`.
  Geometric onset vs MuJoCo first robot contact: 0 to -3 frames.
- Known: depth/segmentation renders disagree on silhouette edges -> edge pixels dropped (3x3 single-id window).

## Single-arm entry-pose model (RLBench contact split), 2026-10-10
- `src/entry_vla/data/rlbench_entry.py`, `scripts/build_rlbench_entry_cache.py`: samples = free-phase frames
  (stride 5, <=12/episode) -> TCP pose at contact split (contact onset - 10). Exo-camera fused cloud (no masks),
  instruction (hashing encoder), TCP pose + gripper + joints. Split by seed: 15-17 val, 18-20 test.
- `src/entry_vla/models/single_entry.py`, `scripts/train_single_entry.py`: PointNet-lite K=4 relaxed WTA;
  rotation error symmetric under pi about TCP z (parallel gripper).
- Run v1 (local data, 5 tasks, 92 eps; 5 skipped with undefined split; 10k steps, best val step 3250, 108 s on 4090):
  val top-1 0.084 m / 0.20 rad (task-mean baseline 0.171 m / 0.32 rad); test 0.063 m / 0.16 rad (baseline 0.144 / 0.38).
  Offline agreement with demo entry poses only; no downstream policy success measured.

## Conditional entry-pose diffusion (RLBench), 2026-10-10
- Cache `scripts/build_entry_diffusion_cache.py` (oracle contact object = nearest non-structure mask id at frame 0
  to TCP at contact onset; skipped if > 0.15 m), `scripts/embed_vlm_features.py` (frozen Qwen3-VL-4B, last
  hidden state mean: bare instruction tokens / crop image tokens), `scripts/train_entry_diffusion.py`.
- Model `src/entry_vla/models/entry_diffusion.py` (+ `pointnet2.py`): pose [p_C/0.2, rot6d], cosine DDPM 100
  steps, FiLM MLP, CFG dropout 0.1, DDIM 25 steps, 16 samples, medoid selection. Centroid c also conditioned.
- Local data: 63/14/10 episodes (10 skipped). 20k steps, ~10 min each on the 4090. Medoid val / test errors:
  full 4.2 cm 0.24 rad / 7.2 cm 0.23 rad; no-vision 3.7 / 5.2 cm (0.24 / 0.18 rad); no-text 4.1 / 7.2 cm;
  task-mean baseline 7.6 cm 0.40 rad / 9.9 cm 0.35 rad. close_fridge test ~27 cm (whole-fridge object).
  Offline demo agreement only; tiny splits.
