# VLA Entry-Pose Prediction: Project Instructions

## Execution policy: implement code, do not launch experiments

This is a code-delivery project. Implement runnable code, configs, tests, and exact user-run commands. Do not execute training, fine-tuning, hyperparameter searches, full evaluations, autonomous outcome collection, benchmark sweeps, or long simulator rollouts unless the user explicitly changes this policy. A milestone mentioning training means implement its training pipeline, not run it. Never start jobs in the background or submit scheduler jobs.

Small verification is allowed only when needed to check code or data structure:

- Syntax/import checks and targeted unit tests using synthetic fixtures.
- Inspect at most one already available episode and up to 10 frames; do not scan full datasets.
- A small CPU forward pass with a tiny randomly initialized model; no optimizer steps or pretrained-weight downloads.
- If the simulator is already installed with assets available, at most one reset and five steps for an adapter/observation smoke check. No policy evaluation or demonstration generation.
- Limit each verification command to 60 seconds and each task's verification to approximately five minutes. Use explicit timeouts and stop if scope grows. Do not repeat equivalent successful checks.

Do not automatically install large simulator stacks, download datasets/checkpoints/assets, replay entire episodes, export full datasets, or precompute embeddings. Write setup/download/export commands for the user instead. Account for implicit downloads on imports or environment creation; skip that test when required assets are absent. Missing dependencies are a reported blocker, not authorization for expensive setup.

Training and evaluation CLIs must have explicit bounded options and a `--dry-run` that validates configuration/schema without loading weights, iterating a dataset, starting a simulator, or executing optimization. Tests must not invoke training as a side effect. Deliver a concise change summary plus commands for the user; do not claim unexecuted validation or experimental results.

## GPU resource configuration

The user has one NVIDIA RTX 4090 available. Do not assume the earlier 32 GB deployment or 96 GB training figures. Treat available VRAM as a configurable resource; verify it from user-provided device information or a lightweight `nvidia-smi` query when relevant. Do not launch GPU workloads to discover capacity.

Default implementation configs to a single GPU (`cuda:0`), with CPU fallbacks for schema and geometry tests. Provide conservative user-run training presets for SmolVLA and the entry predictor: mixed precision where supported, small per-device batches, gradient accumulation, optional activation checkpointing, and optional backbone freezing/parameter-efficient adaptation supported by the pinned model. Expose image resolution, camera count, action horizon, batch size and precision in config rather than hard-code them. Profile memory only when the user explicitly requests runtime profiling; do not claim an untested configuration fits.

Keep preprocessing and simulator rendering separable from model training to avoid concurrent GPU contention. No multi-GPU/sharding assumptions. Larger VLA models are optional configurations requiring separate feasibility checks; do not silently select them. The presence of this GPU does not authorize training, fine-tuning, pretrained inference, benchmarking, or experiment execution. Existing code-only and small-verification limits remain in force.

## CodeGraph and context budget

The user has CodeGraph installed. Use the actual installed CLI or MCP interface for repository structure, symbol locations, dependencies, and caller/callee lookup. Discover its tools/configuration and read installed help once; do not invent command names or flags. If unavailable in this session, document that fact and use targeted `rg` searches.

Keep the graph current at task start if stale, after each coherent source-edit batch, after file/module moves or public API changes, and before final handoff. Prefer incremental updates of changed files. Update before the next graph query that depends on an edit; do not rebuild after every line or repeatedly rebuild an unchanged repository. If only a full rebuild exists and it exceeds the small verification budget, provide the refresh command and report the graph as stale rather than launching a long job.

Index project source and small configuration/documentation files only. Exclude datasets, videos, images, weights, logs, runs, virtual environments, build outputs, caches, and vendored simulator/model repositories unless a narrowly scoped external lookup is necessary. Never rebuild the graph over large data directories.

Query CodeGraph for a compact map, then read only relevant symbols and bounded source ranges. Bound query output, avoid dumping complete graphs or files, and verify graph references against source before editing. Maintain concise `docs/IMPLEMENTATION_STATUS.md` notes with changed modules, decisions, tests actually run, pending user-run commands, blockers, and graph freshness. Consult this note on continuation to avoid repeated exploration. Do not store dataset rows or large logs in the note. Graph upkeep is a development task, not a recurring background job.

## Objective and scope

Implement a simulation-first research project that predicts suitable entry states for a locally trained vision-language-action manipulation policy. A geometric planner executes preparation motion; a frozen local VLA executes approach/contact/manipulation after a verified handoff. Evaluate actual downstream success, not only agreement with demonstration poses.

The eventual hardware is two Franka arms mounted at approximately 45 degrees, default parallel grippers, ego/exo and two wrist cameras, and Cartesian end-effector trajectory chunks plus gripper actions. Depth availability on the real system is unconfirmed. RGB-D simulation experiments must be identified as RGB-D; do not claim RGB-only deployment without a tested perception adapter or sensor change.

This file is an implementation specification, not evidence that the code or experiments already work. Do not invent benchmark results, pretrained compatibility, task IDs, successful tests, or novelty claims. No real-robot execution is in the present scope.

## First actions

1. Inspect existing instructions, source, dependency manifests and CodeGraph; preserve existing work. Avoid broad environment inventory unless needed.
2. Write a concise implementation plan and dependency audit. Pin compatible versions/commit hashes from verified sources; mark unresolved pins rather than fabricate them.
3. Implement both DuoBench/RCS and PerAct2/bimanual RLBench behind a common backend interface. Prefer DuoBench for the dual-Franka pilot; maintain PerAct2 as an independent benchmark backend.
4. Inspect a tiny existing fixture or episode before designing the exporter. If data are absent, implement synthetic schema fixtures and provide selective download commands without executing them.
5. Implement one-task smoke-test commands for each backend. Execute only the bounded checks permitted above; otherwise leave simulator verification pending.

Proceed through routine reversible coding decisions autonomously. Document concrete blockers and complete independent components. Distinguish mocked tests from simulator tests and implemented milestones from experimentally validated milestones.

## Benchmark selection and adapters

### DuoBench / RCS

Official name: DuoBench (the user may write DouBench). Use its MuJoCo/Gymnasium implementation and compatible RCS version. Inspect pinned source for actual task IDs and APIs; initial task choices are Carry-Pot, Transfer-Cube and Bin-Sort. Do not assume display names equal registered environment IDs.

Use the raw recordings for geometry, Cartesian telemetry and replay research; support the reduced LeRobot export as a separate input format. Do not assume the reduced export retains simulator state, depth, calibration, or Cartesian command semantics. The paper's converted actions use joint space; inspect actual schemas and converter settings before adapting actions. Keep raw command targets and measured-future-derived targets distinct.

RCS supports depth rendering, but downloaded demonstration depth must be verified. Implement optional depth-enabled simulator rendering/replay as a user-run preprocessing command. Validate recorded replay state sufficiency, object motion, timestamps and camera calibration; partial replay state is not automatically a complete resumable physics snapshot. Inspect `include_depth` and depth encoding in the pinned RCS version. Documented uint16 millimeter depth becomes float32 meters by division by 1000; do not apply this conversion to already metric values.

DuoBench's published hardware uses Robotiq 2F-85 grippers and three cameras. The target real system has default Franka hands and four cameras. Keep the stock benchmark unchanged; implement a separately named custom scene/configuration for target mounts, grippers and cameras. Do not reuse demonstrations after geometry changes without explicit validation or recollection.

Use task-stage predicates for annotation and diagnostic evaluation only. Stage progress is not a policy-compatible entry label, and monotonically increasing stages do not prove current contact is intact. Privileged stage/object information must not leak into ordinary predictor inputs.

### PerAct2 / bimanual RLBench

Use PerAct2's compatible bimanual RLBench/PyRep forks, not ordinary upstream RLBench as a drop-in replacement. Verify versions and source APIs. Start with a task such as `bimanual_handover_item_easy` or `bimanual_dual_push_buttons` only after verifying the identifier in pinned source; later add lift-tray/drawer tasks.

Inspect saved RGB/depth encoding, camera intrinsics/extrinsics, per-arm state and demonstration provenance. Simulator-generated waypoint demonstrations must not be described as human teleoperation. Saved observations do not necessarily contain original commanded actions or full resumable snapshots.

### Shared contract

Use selectable configs such as `backend=duobench` and `backend=peract2`. Implement adapters for reset/step/close, observation normalization, camera calibration, raw action conversion, demonstration inspection/export, task predicates, preparation, and supported restore/replay. Unsupported capabilities must raise explicit errors or return declared capability metadata; never silently substitute a backend.

Keep simulator environments separate when dependencies conflict. Use a shared versioned episode format and thin adapters; retain backend-specific metadata. Benchmark task difficulty, grippers, cameras and demonstration provenance differ: report each benchmark separately instead of pooling success rates. Optional LIBERO/single-arm RLBench diagnostics are lower priority than both requested backends.

Keep stock scenes unchanged for benchmark experiments. The later tilted dual-Franka setup is a separately named custom evaluation with explicit mount transforms, gravity and gripper geometry.

## Repository organization

Use a small Python package and explicit CLI commands. Suggested structure:

- `src/entry_vla/sim/`: DuoBench and PerAct2 backend adapters, shared camera/control contracts, reset and preparation execution.
- `src/entry_vla/data/`: inspection, export, segmentation, splits, datasets.
- `src/entry_vla/geometry/`: transforms, depth backprojection, visibility, IK/collision wrappers.
- `src/entry_vla/models/`: entry proposal network and outcome scorer.
- `src/entry_vla/policies/`: local-policy adapters and fine-tuning integration.
- `src/entry_vla/evaluation/`: trial runner, baselines, metrics, failure taxonomy.
- `configs/`, `tests/`, `scripts/`, `docs/`.
- `data/` and `runs/`: ignored generated artifacts; raw demonstrations remain immutable.

Each CLI needs `--help`, a configuration file, deterministic seeds where supported, and actionable errors. Design exact commands after inspecting the code; document implemented user-run commands separately from commands actually verified.

## Data and geometry contracts

Use meters, radians, seconds. Define `T_AB` as a transform mapping coordinates from frame B to A. Use a shared world frame W and explicit frames for each arm, TCP, object reference, and camera. Declare quaternion order at every external boundary. Use rotation matrices or continuous 6D rotations inside learned pose representations and valid SO(3) reconstruction.

An exported episode must contain or explicitly mark as unavailable:

- Episode/task/variation IDs, instruction, split, random seed, simulation timestep, recording rate, provenance.
- RGB uint8 images, metric depth float32, valid-depth masks, image sizes.
- Per-camera intrinsics K and camera-to-world transforms T_WC; moving wrist transforms at each frame; near/far clipping and original depth encoding.
- Left/right measured TCP transforms, joint position/velocity, measured gripper state.
- Raw simulator actions with controller type, scale, action frame and quaternion convention; commanded targets if exposed.
- Source demonstration index/time, proposed skill-start interval, local-skill label, arm-role/contact annotations with confidence and provenance.
- Replay/reset metadata sufficient for the implemented recovery procedure.
- Privileged object poses/instance masks/task predicates in a separately identified oracle/annotation namespace.

Distinguish depth-buffer conversion from already metric depth. RLBench's camera configuration supports depth but its default `depth_in_meters` is false. Validate the pinned backend's saved-depth encoding and camera convention. Apply RGB flips/resizing consistently to depth, masks, and intrinsics. Test backprojection and reprojection against known simulator geometry.

Demonstration observations are not necessarily original commanded actions. If deriving targets from future measured poses, name them as derived targets, retain the source trajectory, and document the control assumption. Never silently equate measured future motion, commanded actions, keyframes, and raw simulator controls.

Implement a bounded one-step diagnostic to establish whether obs[t] is before or after action[t]; run it only under the execution policy. DuoBench documents delayed observations in its recording formulation; inspect each actual export rather than assuming identical alignment. Align all modalities accordingly. Handle variable frame spacing and terminal padding with masks. Normalize using training data only.

## Skill-start annotation

Use distance to the task-relevant interaction region, stable target visibility, compatible orientation, gripper/contact mode, and an interval before interaction onset to propose starts. A 0.20 m radius is a baseline parameter, not ground truth. Retain local approach before contact. Provide diagnostic videos/overlays and manual overrides.

Visibility is a weak readiness signal, not a success label. Whole-episode success does not imply every frame or approach candidate is a valid start. For bimanual tasks, annotate a paired state: both poses/configurations, role assignment, held objects and prerequisite contact state. Do not independently crop the two arms into incompatible phases.

Match preparation and demonstration boundary conditions: begin with stopped handoffs, bounded settling, zeroed/recorded velocity, a fixed policy history-reset convention, and explicit contact prerequisites. Record whether adding a pause changes the task. Preserve full demonstrations for full-episode policy baselines.

## Entry-pose models

First implement a supervised proposal model mapping current global RGB-D views, instruction/skill, current robot state, and available object geometry to K candidate paired entry poses and required gripper/role states. Use a lightweight point-cloud or RGB-D encoder, a frozen text encoder, and a small fusion network initially. Predict poses in a declared frame; use a point/region frame when full object orientation is not observable.

Start with nearest demonstrated entry as a baseline. Avoid averaging incompatible approach modes with a single MSE pose output. If multi-hypothesis prediction is needed, implement a documented assignment/minimum-over-K loss and monitor candidate collapse. Translation and rotation losses need explicit units/weights and valid rotation reconstruction.

The candidate generator may also sample task-conditioned partial shells around interaction regions. If sampling camera poses, transform to TCP using `T_WE = T_WC inverse(T_EC)`. Camera-facing orientation does not establish a usable grasp. Filter orientations, table intersections, IK branches, full-arm collision, attached objects, and paired clearance. Avoid requesting endpoint wrist images before physical arrival.

Second implement a small candidate success/duration scorer trained on autonomous outcomes from a frozen local policy. Pre-travel scoring uses observations available before travel; arrival verification uses actual endpoint observations. Separate models or clearly separated inputs are required. Include q_L/q_R and contact/history features where relevant.

Outcome labels depend on policy checkpoint, controller, reset protocol, and timeout. Version all of them; regenerate or revalidate labels after policy changes. Keep transit failures and manipulation failures separate. Do not interpret ensemble disagreement or a detector score as a calibrated safety certificate.

## Downstream local policy

Use locally fine-tuned SmolVLA as the initial VLA if its pinned implementation works with the backend. It requires a custom Cartesian/bimanual data adapter; do not claim zero-shot Franka or RLBench compatibility. Verify native checkpoint loading/horizon before modifying it. Begin with a small camera set, then measure the cost of additional views.

A diffusion policy or ACT can validate simulator plumbing but must be named as a visuomotor-policy baseline, not presented as the main VLA result. Never use expert demonstration replay as evidence of VLA success at perturbed entries.

For the intended Cartesian contract, state is two copies of position(3), rotation6D(6), and measured gripper width(1): 20D total. Actions are H x 14: each arm's body-relative translation(3), relative rotation vector(3), and desired gripper command(1). Keep native backend action conventions in an explicit adapter.

For each sequential chunk: `delta_p = R_prev.T @ (p_next - p_prev)` and `delta_rot = Log(R_prev.T @ R_next)`. The first previous pose is the measured query pose; subsequent previous poses are preceding desired waypoints. Decode from the same measured query pose. These decoupled increments are not the coupled SE(3) logarithm. Do not stack unrelated measured-to-commanded per-frame deltas as a sequential chunk. If commands are unavailable, document the measured-future-target variant separately.

Validate action scale, frame, gripper polarity, tracking, joint limits, and controller behavior before training. Check that planning-based action modes do not unintentionally perform the manipulation for the VLA: report any internal motion planning and keep it matched across baselines.

## Simulator interventions and outcome collection

Implement verified scene initialization/restoration. Saved observations alone are not full physical snapshots. Restore joints, object/articulation states, velocities, gripper attachment/contact state, task bookkeeping, and randomization as needed. If arbitrary snapshots are unsupported, reproduce the episode from initialization with a validated prefix or use a supported reset procedure. Test replay error and task predicates.

Actual main trials must physically execute preparation; teleporting is allowed only for explicitly labeled diagnostic interventions. Do not teleport a held object into an inconsistent grasp or claim preparation success from teleportation. Recheck the scene after arrival, cancel old commands, flush action queues, and enforce one command owner per arm.

Implement bounded candidate perturbations and autonomous skill rollouts with success, timeout, invalid-state, collision/clearance violation, transit-failure, and intervention labels. Do not deliberately create unsafe physical collisions; simulation labels can use collision checks and early termination. Cache pre-travel context, candidate decision, actual arrival and final outcome as distinct records.

Scheduling experiments must change physical preparation/contact/history to have meaning. Include a stable-scene/history-reset negative control where arrival order should not matter. Independent arm starts do not establish joint compatibility.

## Evaluation and leakage prevention

Split by episode and scene/layout seed, never adjacent frames. Keep descendants of the same source demonstration grouped when data are generated. Add held-out placements/approach directions and a separately reported held-out-task experiment if trained to support it. Do not tune on test outcomes.

Oracle poses/masks may bootstrap annotation and diagnostics. Separately evaluate learned perception or injected localization error. The main entry predictor must not receive future wrist images, future actions, or privileged state unless explicitly labeled an oracle condition.

Baselines: fixed canonical entry; distance shell; visibility plus geometric validity; nearest/demo-supported entry; supervised predicted entry; factorized per-arm outcome scoring; joint outcome scoring. Use identical frozen policies and controllers for entry-selector comparisons. Use full episodes for the VLA-only baseline. Share demonstration and probe budgets fairly.

Report entry position/rotation errors AND downstream autonomous success over all requested trials. Include acceptance coverage, capped episode time including failures, successful completion time separately, preparation/planning time, policy queries, latency, minimum clearance/tracking diagnostics, reset rejection rate, and failure categories. Report seeds, trial counts and uncertainty. Do not hide planner failures or rejected candidates from the main denominator. A rollout success estimate is not a formal safety guarantee.

## Milestones and acceptance checks

Every milestone below is a code deliverable. Full acceptance experiments are user-run; leave their status pending until results are supplied. Never interpret a milestone as permission to execute training or collection.

M0: implement pinned environment definitions and one-task headless reset/step/RGB-D smoke scripts for both DuoBench and PerAct2. Support capability checks and missing-asset errors without implicit downloads.

M1: implement tiny demonstration inspection, shared export and geometry/alignment checks for both formats. Verify with synthetic fixtures and at most one available episode subset. Provide full replay/export commands for user execution.

M2: implement start-interval annotation, overlays, manual overrides, stopped-entry preparation and local-policy adapters. Provide canonical local-policy rollout commands; do not run them.

M3: implement supervised paired-entry training, evaluation, multi-candidate losses and demo/shell baselines. Validate shapes and pose decoding with a tiny forward pass only. Provide tiny-overfit and training commands for the user; no optimizer execution by agents.

M4: implement frozen-policy candidate probe collection, outcome labels, scorer training and matched-policy comparisons. Provide bounded collection/training/evaluation configs; execute none of these experiments.

M5: implement multi-task and joint/posture ablation configs, separate tilted-Franka scenes and explicit planner/MoveIt integration as requested. Leave real-robot execution out of scope.

Targeted tests should cover transforms, depth units/reprojection, pose/chunk round trips, first-step anchoring, scaling, temporal alignment, splits and padded masks. Provide optional simulator integration tests for predicates, preparation, attached-state restoration and policy rollouts; keep them opt-in and outside default tests. Default tests must be lightweight, offline, and avoid training/model downloads. Mark skipped integrations and mock-only coverage honestly.

## Reporting and dependencies

Separate simulator and modern model-training environments if their Python/dependency requirements conflict; connect through a versioned file or local-service contract. Prefer a thin adapter over rewriting the benchmark. The initial simulator planner need not be MoveIt, but must be identified; later real-system claims require actual MoveIt integration.

Keep configurations, checkpoint hashes, dataset checksums, source commits, seeds, metrics and videos for each run. Use local logging by default. Finish implementation updates with what changed, commands actually executed, results, and unresolved blockers. Complete the requested runnable code without architecture-only stubs. Report optional integration checks as unrun when execution limits or missing dependencies prevent verification. Never relax execution limits merely to mark a milestone complete.

## Primary references to inspect

- DuoBench project: https://duobench.github.io/
- DuoBench implementation: https://github.com/RobotControlStack/duobench
- DuoBench raw data: https://huggingface.co/datasets/RobotControlStack/duobench_raw
- DuoBench LeRobot data: https://huggingface.co/datasets/RobotControlStack/duobench
- RCS implementation: https://github.com/RobotControlStack/robot-control-stack
- RCS documentation: https://robotcontrolstack.org/

- PerAct2 benchmark/data: https://bimanual.github.io/
- PerAct2 implementation and fork installation: https://github.com/markusgrotz/peract_bimanual
- Upstream RLBench: https://github.com/stepjam/RLBench
- RLBench observation configuration: https://github.com/stepjam/RLBench/blob/master/rlbench/observation_config.py
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- LIBERO rendering/conversion: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py
- SmolVLA implementation/docs: https://huggingface.co/docs/lerobot/smolvla
- SViP prior switching/feasibility method: https://arxiv.org/abs/2506.18825
- SkipVLA prior planner/VLA hybrid: https://arxiv.org/abs/2609.20648

Verify pinned source APIs before using reference commands. The listed methods establish relevant prior work; do not claim that planner/VLA switching, visibility constraints, or bimanual entry selection alone are new.

## Git
- Make a local git and add short commit after each success step so that you can comeback if need. No "Co-author Claude" in commit.
- each commit should name "fix/feat/chore: short intro", then follow by bullets explain briefly explain what commit does. 