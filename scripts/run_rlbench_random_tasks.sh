#!/usr/bin/env bash
# Sample K random single-arm RLBench tasks (deterministic for a seed), generate one demo per
# task (random variation drawn from the same seed), and run the perception phase split on it.
# A task that fails to generate or process is logged and skipped. Run from the repo root.
#
#   bash scripts/run_rlbench_random_tasks.sh 10 42
#
# Args: K SEED. Env overrides: RAW_ROOT, OUT_ROOT, SIM_PY, PROC_PY, IMAGE_SIZE (default 256),
# EPISODES (default 1), TASKS="a b c" (skip sampling), SKIP_GENERATE=1.
#
# The privileged reference needs the task-specific object slot in task_low_dim_state, which is
# unknown for arbitrary tasks, so it is disabled (--target-slice none). Each comparison.json then
# holds the method's start frames and the task-agnostic state events (gripper toggles, arm stops).
set -uo pipefail

K=${1:?number of tasks}; SEED=${2:?seed}
IMAGE_SIZE=${IMAGE_SIZE:-256}; EPISODES=${EPISODES:-1}
RAW_ROOT=${RAW_ROOT:-data/raw/rlbench_single_seed${SEED}}
OUT_ROOT=${OUT_ROOT:-data/entry_process/rlbench_single_seed${SEED}}
SIM_PY=${SIM_PY:-$HOME/anaconda3/envs/peract2/bin/python}
PROC_PY=${PROC_PY:-$HOME/anaconda3/envs/entryvla/bin/python}
mkdir -p "$OUT_ROOT"
trap 'echo "interrupted"; exit 130' INT TERM   # Ctrl+C stops the whole run, not just the child

if [ -z "${TASKS:-}" ]; then
  TASKS=$(python3 - "$K" "$SEED" <<'EOF'
import os, random, sys
k, seed = int(sys.argv[1]), int(sys.argv[2])
d = "ext/RLBench/rlbench/tasks"
tasks = sorted(f[:-3] for f in os.listdir(d) if f.endswith(".py") and f != "__init__.py")
print(" ".join(random.Random(seed).sample(tasks, k)))
EOF
)
fi
echo "$TASKS" | tr ' ' '\n' > "$OUT_ROOT/tasks.txt"
echo "== tasks (seed $SEED): $TASKS"

export COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-$HOME/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

: > "$OUT_ROOT/status.tsv"
for t in $TASKS; do
  if [ "${SKIP_GENERATE:-0}" != "1" ]; then
    echo "== generate $t"
    if ! "$SIM_PY" scripts/rlbench_single_generator.py --task "$t" --variation -1 --episodes "$EPISODES" \
         --seed "$SEED" --image_size "$IMAGE_SIZE" "$IMAGE_SIZE" --save_path "$RAW_ROOT" \
         > "$OUT_ROOT/gen_$t.log" 2>&1; then
      echo -e "$t\tgenerate_failed" >> "$OUT_ROOT/status.tsv"; echo "   generate failed (see gen_$t.log)"; continue
    fi
  fi
  for ep in $(ls -d "$RAW_ROOT/$t"/variation*/episodes/episode* 2>/dev/null | sort -V); do
    var=$(basename "$(dirname "$(dirname "$ep")")")
    out="$OUT_ROOT/$t/$var/$(basename "$ep")"
    echo "== process $t/$var/$(basename "$ep")"
    mkdir -p "$out"
    if [ ! -f "$ep/low_dim_obs.pkl" ]; then   # interrupted/partial generation
      echo -e "$t\t$var\tincomplete_episode" >> "$OUT_ROOT/status.tsv"; echo "   skipped: no low_dim_obs.pkl"; continue
    fi
    "$PROC_PY" scripts/detect_transition_rlbench.py --episode "$ep" --out "$out" --no-video \
      > "$out/detect.log" 2>&1 || true   # needs a task-specific target slot; optional here
    if "$PROC_PY" scripts/perception_phase_split.py --episode "$ep" --out "$out/perception" \
         --target-slice none > "$out/perception.log" 2>&1; then
      echo -e "$t\t$var\tok" >> "$OUT_ROOT/status.tsv"
    else
      echo -e "$t\t$var\tperception_failed" >> "$OUT_ROOT/status.tsv"; echo "   perception failed (see $out/perception.log)"
    fi
  done
done

"$PROC_PY" - "$OUT_ROOT" <<'EOF'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(root, "*", "variation*", "episode*", "perception", "comparison.json"))):
    d = json.load(open(p))
    task = p.split(os.sep)[-5]
    labels = {c: (v["detection"] or {}).get("label") for c, v in d["per_camera"].items()}
    rows.append({"task": task, "instruction": d["instruction"], "frames": d["num_frames"],
                 "start_r0.20": d["pred_start_at_baseline"],
                 "start_r0.08": next(r["pred_start"] for r in d["radius_sweep"] if abs(r["radius_m"] - 0.08) < 1e-9),
                 "gripper_toggles": d["other_references"].get("gripper_toggle_frames"),
                 "arm_stops": d["other_references"].get("arm_stop_frames"), "detected": labels})
json.dump(rows, open(os.path.join(root, "summary.json"), "w"), indent=2)
print(f"\n{'task':32s} {'frames':>6s} {'R=.20':>6s} {'R=.08':>6s}  gripper toggles / arm stops / detected object")
for r in rows:
    print(f"{r['task']:32s} {r['frames']:6d} {str(r['start_r0.20']):>6s} {str(r['start_r0.08']):>6s}  "
          f"{r['gripper_toggles']} / {r['arm_stops']} / {sorted(set(r['detected'].values()), key=str)}")
EOF
echo "== done: $OUT_ROOT/{tasks.txt,status.tsv,summary.json}"
