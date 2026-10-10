#!/usr/bin/env bash
# For each "<task> <variation>" line of a task list (default docs/select_single.md, lines before "----"),
# generate one RLBench single-arm demo per seed SEED_FROM..SEED_TO (in order) and run the contact phase
# split on it with the manipulation start PRE_FRAMES frames before the contact event.
#
#   raw:       RAW_ROOT/<task>/variation<V>/seed<S>/episodes/episode0
#   processed: OUT_ROOT/<task>/variation<V>/seed<S>/episode0/contact/
#
# Resource limits: every step runs one process at a time, pinned to CPUS (default 4 cores, 0-3) with
# `nice`, BLAS/OpenMP threads capped, and a per-step timeout. Finished steps are skipped, so the script
# can be re-run after an interruption. Run from the repo root, in the background:
#
#   nohup bash scripts/run_rlbench_select_single.sh > logs/rlbench_select_single.log 2>&1 &
#
# Env overrides: TASK_LIST, SEED_FROM (1), SEED_TO (20), PRE_FRAMES (10), CPUS (0-3), IMAGE_SIZE (256),
# GEN_TIMEOUT (900 s), SPLIT_TIMEOUT (900 s), RAW_ROOT, OUT_ROOT, SIM_PY, PROC_PY, SPLIT_ARGS, FORCE_SPLIT=1.
set -uo pipefail
TASK_LIST=${TASK_LIST:-docs/select_single.md}
SEED_FROM=${SEED_FROM:-1}; SEED_TO=${SEED_TO:-20}; PRE_FRAMES=${PRE_FRAMES:-10}
CPUS=${CPUS:-0-3}; IMAGE_SIZE=${IMAGE_SIZE:-256}
GEN_TIMEOUT=${GEN_TIMEOUT:-900}; SPLIT_TIMEOUT=${SPLIT_TIMEOUT:-900}
RAW_ROOT=${RAW_ROOT:-data/raw/rlbench_single}
OUT_ROOT=${OUT_ROOT:-data/entry_process/rlbench_select_single}
SIM_PY=${SIM_PY:-$HOME/anaconda3/envs/peract2/bin/python}
PROC_PY=${PROC_PY:-$HOME/anaconda3/envs/entryvla/bin/python}

export COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-$HOME/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
NT=$(echo "$CPUS" | tr ',' '\n' | awk -F- '{n += ($2 == "" ? 1 : $2 - $1 + 1)} END {print n}')
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMEXPR_NUM_THREADS=$NT
LIMIT=(taskset -c "$CPUS" nice -n 10)

mkdir -p "$OUT_ROOT"
trap 'echo "interrupted"; exit 130' INT TERM
STATUS=$OUT_ROOT/status.tsv
mapfile -t LINES < <(sed -n '/^----/q;p' "$TASK_LIST" | awk 'NF >= 2 {print $1, $2}')
echo "== ${#LINES[@]} tasks, seeds $SEED_FROM..$SEED_TO, pre-frames $PRE_FRAMES, cpus $CPUS ($NT threads)"

for line in "${LINES[@]}"; do
  read -r task var <<< "$line"
  for seed in $(seq "$SEED_FROM" "$SEED_TO"); do
    ep=$RAW_ROOT/$task/variation$var/seed$seed/episodes/episode0
    out=$OUT_ROOT/$task/variation$var/seed$seed/episode0
    mkdir -p "$out"
    if [ ! -f "$ep/low_dim_obs.pkl" ]; then
      echo "== generate $task v$var seed$seed"
      rm -rf "$ep"   # partial episode from an interrupted run
      timeout --kill-after=30 "$GEN_TIMEOUT" "${LIMIT[@]}" "$SIM_PY" scripts/rlbench_single_generator.py \
        --task "$task" --variation "$var" --episodes 1 --seed "$seed" \
        --image_size "$IMAGE_SIZE" "$IMAGE_SIZE" --save_path "$RAW_ROOT" > "$out/generate.log" 2>&1
      rc=$?
      if [ $rc -ne 0 ] || [ ! -f "$ep/low_dim_obs.pkl" ]; then
        echo -e "$task\t$var\t$seed\tgenerate_failed_$rc" >> "$STATUS"; echo "   generate failed ($rc), see $out/generate.log"
        continue
      fi
      echo -e "$task\t$var\t$seed\tgenerated" >> "$STATUS"
    fi
    if [ "${FORCE_SPLIT:-0}" != "1" ] && [ -f "$out/contact/contact.json" ]; then continue; fi
    echo "== contact split $task v$var seed$seed"
    if timeout --kill-after=30 "$SPLIT_TIMEOUT" "${LIMIT[@]}" "$PROC_PY" scripts/contact_phase_split.py \
         --episode "$ep" --out "$out" --pre-frames "$PRE_FRAMES" ${SPLIT_ARGS:-} > "$out/contact.log" 2>&1; then
      echo -e "$task\t$var\t$seed\tsplit_ok" >> "$STATUS"; tail -n1 "$out/contact.log"
    else
      echo -e "$task\t$var\t$seed\tsplit_failed" >> "$STATUS"; echo "   split failed, see $out/contact.log"
    fi
  done
done

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
