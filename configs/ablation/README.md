# Ablation configs (M5)

Each file is a set of overrides applied on top of a base config. Run each condition
separately and report every benchmark on its own (never pool DuoBench and PerAct2).
None of these have been executed.

| File | Base | Varies |
|---|---|---|
| `multitask.yaml` | `train/entry_4090.yaml` | single-task vs multi-task training; per-task reporting; held-out task |
| `scorer_joint_vs_factorized.yaml` | `compare.yaml` | joint vs factorized outcome scoring over the same generator |
| `posture.yaml` | `train/entry_4090.yaml` | entry model with vs without q_L/q_R posture inputs |
| `arrival_order.yaml` | `compare.yaml` | simultaneous vs left-first vs right-first preparation (+ stable-scene negative control) |
| `pause.yaml` | `compare.yaml` | 0 vs N settle-pause steps after arrival (does a pause change the task?) |
| `region_source.yaml` | `compare.yaml` | oracle vs injected localisation error (sigma) vs learned perception |
| `cameras.yaml` | `train/smolvla_4090.yaml` | SmolVLA camera count (1 vs 3) |

Example: `entry-vla compare --config configs/compare.yaml $(python scripts/ablation_args.py configs/ablation/pause.yaml pause_10)`
