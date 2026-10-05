# entry-vla

This project predicts entry states for planner→VLA handoff, simulation first. A geometric
planner moves both Franka arms to a predicted paired entry, verifies a stopped handoff, and a
frozen local VLA (SmolVLA, fine-tuned locally) runs the manipulation. Benchmarks are DuoBench/RCS
(MuJoCo, preferred for the dual-Franka pilot) and PerAct2 bimanual RLBench, behind a common
backend interface.

**Status:** code delivery only. No simulator, training or evaluation has been run.
[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) lists what was verified and what
is pending. [docs/PLAN.md](docs/PLAN.md) has the plan, the dependency audit and the design decisions.

## Layout

```
src/entry_vla/
  geometry/    transforms, rotations (explicit quaternion order), camera/depth, visibility, Franka FK/IK, shells
  sim/         backend contract, duobench.py, peract2.py, mock.py, preparation (planner + handoff), custom_scene, moveit_planner
  data/        episode format, DuoBench raw / LeRobot v3 / PerAct2 readers, replay re-render, alignment, splits, segmentation, overlays, entry samples
  policies/    Cartesian chunk contract, policy runner, SmolVLA dataset/train/inference adapters
  models/      entry proposal net + min-over-K loss, outcome scorer + arrival verifier, text encoders, training loops
  evaluation/  selectors (baselines), trial runner, probes, metrics, failure taxonomy, matched comparison
  cli/         `entry-vla <command>`
configs/  envs/  scripts/  tests/  docs/      data/ and runs/ are git-ignored
```

## Conventions

- Units are metres, radians and seconds. `T_AB` maps coordinates from B to A.
- Cameras use the OpenCV frame; depth is planar float32 metres with a validity mask; each source
  depth encoding is declared.
- Policy state is 20D: per arm, position, rot6D and measured gripper width.
- Policy actions are H×14: per arm, body-relative Δp, rotation vector and gripper command. The
  first step is anchored to the measured query pose; later steps to the preceding desired
  waypoint (sequential, not the SE(3) log).
- Experiments built on RGB-D simulation are RGB-D experiments. Depth on the real system is
  unconfirmed.

## Commands (all user-run; every command has `--help`, `--config`, `--seed` and `--dry-run`)

Install the core in any env with `pip install -e .`, or run with `PYTHONPATH=src python -m entry_vla.cli.main ...`.

| Step | Command |
|---|---|
| M0 smoke (1 reset, ≤5 steps) | `entry-vla smoke --config configs/smoke.yaml backend=duobench --out runs/smoke_duobench` (PerAct2: `xvfb-run -a entry-vla smoke ... backend=peract2`) |
| DuoBench frames | `entry-vla duobench-frames --config configs/export.yaml --allow-asset-import` → paste into `configs/export/duobench_raw.yaml` |
| Inspect one episode (≤10 frames) | `entry-vla inspect --config configs/export.yaml export=duobench_raw --handeye` / `export=peract2` / `export=duobench_lerobot` |
| Export | `entry-vla export --config configs/export.yaml export=duobench_raw --max-episodes 1` |
| RGB-D re-render (DuoBench) | `entry-vla replay-render --config configs/export.yaml --max-episodes 1` |
| Alignment | `entry-vla check-alignment --config configs/export.yaml --episode-dir <ep>`; live: `--live backend=duobench` (1 reset + 2 steps) |
| Splits | `entry-vla split --config configs/split.yaml` |
| Annotate starts | `entry-vla annotate --config configs/annotate.yaml --overlays` (manual overrides: `configs/overrides/skill_starts.yaml`) |
| SmolVLA data/training | `entry-vla policy-dataset --config configs/train/smolvla_4090.yaml`; `entry-vla train-smolvla --config configs/train/smolvla_4090.yaml` |
| Canonical rollouts | `entry-vla rollout --config configs/rollout.yaml backend=duobench policy.type=smolvla ... --max-trials 10` |
| Entry model | `entry-vla check-models`; tiny overfit: `entry-vla train-entry --config configs/train/entry_4090.yaml train.tiny_overfit=true train.steps=300`; full: drop the overrides; eval: `entry-vla eval-entry --checkpoint <pt> --split val` |
| Probes / scorer | `entry-vla collect-probes --config configs/probes.yaml backend=duobench policy.type=smolvla ... --max-scenes 10`; `entry-vla train-scorer --config configs/train/scorer_4090.yaml data.probe_files=[<run>/probes.jsonl]` |
| Matched comparison | `entry-vla compare --config configs/compare.yaml --max-trials 20` |
| Ablations | `entry-vla compare --config configs/compare.yaml $(python scripts/ablation_args.py configs/ablation/pause.yaml pause_10)` |
| Fixtures (mock) | `entry-vla make-fixtures --out data/episodes/mock --n 4` |

The mock backend (`backend=mock`) exists only to test pipeline plumbing. Its results are never
benchmark evidence.

## Tests

```bash
python -m pytest -q                       # offline: geometry, depth, chunks, export, splits, mock pipelines
python -m pytest -q --run-sim -m sim      # opt-in simulator integration (needs installed backend + assets)
```
