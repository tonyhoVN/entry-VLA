#!/usr/bin/env bash
# For each "<task> <variation>" line of a task list (default docs/select_single.md, lines before "----"),
# generate one RLBench single-arm demo per seed SEED_FROM..SEED_TO (in order) and run the contact phase
# split on it with the manipulation start PRE_FRAMES frames before the contact event.
#
#   raw:       RAW_ROOT/<task>/variation<V>/seed<S>/episodes/episode0
#   processed: OUT_ROOT/<task>/variation<V>/seed<S>/episode0/contact/
#
# Resource limits: JOBS episodes run at once (default 1); with CPUS set (default 0-3) every process is
# pinned to those cores with `nice`; CPUS=all disables pinning. BLAS/OpenMP threads per process are
# capped at THREADS (default: number of CPUS cores, or 4 with CPUS=all); each step has a timeout.
# Finished steps are skipped, so the script can be re-run after an interruption. Run from the repo root, in the background:
#
#   nohup bash scripts/run_rlbench_select_single.sh > logs/rlbench_select_single.log 2>&1 &
#
# Workstation without limits: JOBS=12 CPUS=all bash scripts/run_rlbench_select_single.sh
# Env overrides: JOBS, THREADS, TASK_LIST, SEED_FROM (1), SEED_TO (20), PRE_FRAMES (10), CPUS (0-3), IMAGE_SIZE (256),
# GEN_TIMEOUT (900 s), SPLIT_TIMEOUT (900 s), RAW_ROOT, OUT_ROOT, SIM_PY, PROC_PY, SPLIT_ARGS, FORCE_SPLIT=1.
set -uo pipefail
TASK_LIST=${TASK_LIST:-docs/select_single.md}
SEED_FROM=${SEED_FROM:-1}; SEED_TO=${SEED_TO:-20}; PRE_FRAMES=${PRE_FRAMES:-10}
JOBS=${JOBS:-1}; CPUS=${CPUS:-0-3}; IMAGE_SIZE=${IMAGE_SIZE:-256}
GEN_TIMEOUT=${GEN_TIMEOUT:-900}; SPLIT_TIMEOUT=${SPLIT_TIMEOUT:-900}
RAW_ROOT=${RAW_ROOT:-data/raw/rlbench_single}
OUT_ROOT=${OUT_ROOT:-data/entry_process/rlbench_select_single}
SIM_PY=${SIM_PY:-$HOME/anaconda3/envs/peract2/bin/python}
PROC_PY=${PROC_PY:-$HOME/anaconda3/envs/entryvla/bin/python}

export COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-$HOME/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
if [ "$CPUS" = "all" ]; then
  NT=${THREADS:-4}; LIMIT=""
else
  NT=${THREADS:-$(echo "$CPUS" | tr ',' '\n' | awk -F- '{n += ($2 == "" ? 1 : $2 - $1 + 1)} END {print n}')}
  LIMIT="taskset -c $CPUS nice -n 10"
fi
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMEXPR_NUM_THREADS=$NT

mkdir -p "$OUT_ROOT"
trap 'echo "interrupted"; kill 0; exit 130' INT TERM
export STATUS=$OUT_ROOT/status.tsv LIMIT RAW_ROOT OUT_ROOT SIM_PY PROC_PY PRE_FRAMES IMAGE_SIZE GEN_TIMEOUT \
  SPLIT_TIMEOUT SPLIT_ARGS=${SPLIT_ARGS:-} FORCE_SPLIT=${FORCE_SPLIT:-0}
mapfile -t LINES < <(sed -n '/^----/q;p' "$TASK_LIST" | awk 'NF >= 2 {print $1, $2}')
echo "== ${#LINES[@]} tasks, seeds $SEED_FROM..$SEED_TO, pre-frames $PRE_FRAMES, jobs $JOBS, cpus $CPUS ($NT threads)"

run_one() {   # task variation seed
  local task=$1 var=$2 seed=$3
  local ep=$RAW_ROOT/$task/variation$var/seed$seed/episodes/episode0
  local out=$OUT_ROOT/$task/variation$var/seed$seed/episode0
  mkdir -p "$out"
  if [ ! -f "$ep/low_dim_obs.pkl" ]; then
    echo "== generate $task v$var seed$seed"
    rm -rf "$ep"   # partial episode from an interrupted run
    timeout --kill-after=30 "$GEN_TIMEOUT" $LIMIT "$SIM_PY" scripts/rlbench_single_generator.py \
      --task "$task" --variation "$var" --episodes 1 --seed "$seed" \
      --image_size "$IMAGE_SIZE" "$IMAGE_SIZE" --save_path "$RAW_ROOT" > "$out/generate.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$ep/low_dim_obs.pkl" ]; then
      echo -e "$task\t$var\t$seed\tgenerate_failed_$rc" >> "$STATUS"; echo "   $task seed$seed generate failed ($rc)"
      return 0
    fi
    echo -e "$task\t$var\t$seed\tgenerated" >> "$STATUS"
  fi
  if [ "$FORCE_SPLIT" != "1" ] && [ -f "$out/contact/contact.json" ]; then return 0; fi
  if timeout --kill-after=30 "$SPLIT_TIMEOUT" $LIMIT "$PROC_PY" scripts/contact_phase_split.py \
       --episode "$ep" --out "$out" --pre-frames "$PRE_FRAMES" $SPLIT_ARGS > "$out/contact.log" 2>&1; then
    echo -e "$task\t$var\t$seed\tsplit_ok" >> "$STATUS"; echo "== split ok $task v$var seed$seed: $(tail -n1 "$out/contact.log")"
  else
    echo -e "$task\t$var\t$seed\tsplit_failed" >> "$STATUS"; echo "   $task seed$seed split failed, see $out/contact.log"
  fi
}
export -f run_one

for line in "${LINES[@]}"; do
  read -r task var <<< "$line"
  for seed in $(seq "$SEED_FROM" "$SEED_TO"); do echo "$task $var $seed"; done
done | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _

"$PROC_PY" - "$OUT_ROOT" <<'PY'
import csv, glob, json, os, sys
root = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "variation*", "seed*", "episode*", "contact", "contact.json"))):
    j = json.load(open(f)); r = j["references"]
    task, var, seed, ep = f.split(os.sep)[-6:-2]
    rows.append({"task": task, "variation": var, "seed": seed, "episode": ep, "num_frames": j["num_frames"],
                 "first_contact_onset": j["first_contact_onset"], "split_frame": j.get("split_frame"),
                 "n_intervals": len(j["contact_intervals"]), "object_motion_onset": r["object_motion_onset"],
                 "touch_force_onset": r["touch_force_onset"],
                 "onset_minus_object_motion": j["onset_minus_object_motion"]})
if rows:
    out = os.path.join(root, "contact_summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} episodes -> {out}")
PY
echo "== done"
