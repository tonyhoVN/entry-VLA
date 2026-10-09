#!/usr/bin/env bash
# Collect VLABench scripted demos for short (2-skill) primitive tasks and run the contact phase split.
# Each demo runs in its own process under `timeout`, so a hang/crash costs one attempt, not the batch;
# failed demos (exit 2) are retried with the next index up to MAX_TRIES per task. Run from repo root:
#
#   nohup bash scripts/run_vlabench_contact.sh > logs/vlabench_contact.log 2>&1 &
#
# Env overrides: TASKS, EPISODES (successful demos per task, default 1), SEED (42), MAX_TRIES (5),
# EP_TIMEOUT (s, 1200), RAW_ROOT, OUT_ROOT, SIM_PY, PROC_PY, SKIP_GENERATE=1, EXTRA_ARGS (split options).
set -uo pipefail
TASKS=${TASKS:-"select_fruit select_chemistry_tube select_poker select_book select_mahjong"}
EPISODES=${EPISODES:-1}; SEED=${SEED:-42}; MAX_TRIES=${MAX_TRIES:-5}; EP_TIMEOUT=${EP_TIMEOUT:-1200}
RAW_ROOT=${RAW_ROOT:-data/raw/vlabench}
OUT_ROOT=${OUT_ROOT:-data/entry_process/vlabench_short5_seed${SEED}}
SIM_PY=${SIM_PY:-$HOME/anaconda3/envs/vlabench/bin/python}
PROC_PY=${PROC_PY:-$HOME/anaconda3/envs/entryvla/bin/python}
export MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p "$OUT_ROOT"
trap 'echo "interrupted"; exit 130' INT TERM
: > "$OUT_ROOT/status.tsv"

for t in $TASKS; do
  if [ "${SKIP_GENERATE:-0}" != "1" ]; then
    ok=$(ls -d "$RAW_ROOT/$t/seed$SEED"/episodes/episode*/data.hdf5 2>/dev/null | wc -l); idx=0; tries=0
    while [ "$ok" -lt "$EPISODES" ] && [ "$tries" -lt "$MAX_TRIES" ]; do
      if [ -f "$RAW_ROOT/$t/seed$SEED/episodes/episode$idx/data.hdf5" ]; then idx=$((idx+1)); continue; fi
      echo "== generate $t episode$idx (try $((tries+1))/$MAX_TRIES)"
      timeout --kill-after=30 "$EP_TIMEOUT" "$SIM_PY" scripts/vlabench_generator.py --task "$t" --index "$idx" \
        --seed "$SEED" --save_path "$RAW_ROOT" > "$OUT_ROOT/gen_${t}_$idx.log" 2>&1
      rc=$?; tail -n1 "$OUT_ROOT/gen_${t}_$idx.log"
      case $rc in
        0) ok=$((ok+1)); echo -e "$t\tepisode$idx\tgenerated" >> "$OUT_ROOT/status.tsv" ;;
        2) echo -e "$t\tepisode$idx\tdemo_failed" >> "$OUT_ROOT/status.tsv" ;;
        124|137) echo -e "$t\tepisode$idx\ttimeout" >> "$OUT_ROOT/status.tsv"; echo "   timeout" ;;
        *) echo -e "$t\tepisode$idx\terror_$rc" >> "$OUT_ROOT/status.tsv"; echo "   error $rc (see gen_${t}_$idx.log)" ;;
      esac
      idx=$((idx+1)); tries=$((tries+1))
    done
  fi
  for ep in $(ls -d "$RAW_ROOT/$t/seed$SEED"/episodes/episode* 2>/dev/null | sort -V); do
    [ -f "$ep/data.hdf5" ] || continue
    out="$OUT_ROOT/$t/$(basename "$ep")"; mkdir -p "$out"
    echo "== contact split $t/$(basename "$ep")"
    if "$PROC_PY" scripts/contact_phase_split_vlabench.py --episode "$ep" --out "$out" ${EXTRA_ARGS:-} \
         > "$out/contact.log" 2>&1; then
      tail -n1 "$out/contact.log"; echo -e "$t\t$(basename "$ep")\tsplit_ok" >> "$OUT_ROOT/status.tsv"
    else
      echo "   split failed (see $out/contact.log)"; echo -e "$t\t$(basename "$ep")\tsplit_failed" >> "$OUT_ROOT/status.tsv"
    fi
  done
done

"$PROC_PY" - "$OUT_ROOT" <<'PY'
import csv, glob, json, os, sys
root = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "episode*", "contact", "contact.json"))):
    j = json.load(open(f)); r = j["references"]
    rows.append({"task": j["task"], "episode": f.split(os.sep)[-3], "num_frames": j["num_frames"],
                 "first_contact_onset": j["first_contact_onset"], "split_frame": j["split_frame"],
                 "n_intervals": len(j["contact_intervals"]), "sim_robot_contact_onset": r["sim_robot_contact_onset"],
                 "sim_finger_contact_onset": r["sim_finger_contact_onset"],
                 "gripper_closing_onset": r["gripper_closing_onset"], "skill_end_frames": r["skill_end_frames"]})
if rows:
    out = os.path.join(root, "contact_summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} episodes -> {out}")
PY
echo "== done"
