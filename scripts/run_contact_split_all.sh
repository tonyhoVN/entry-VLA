#!/usr/bin/env bash
# Run scripts/contact_phase_split.py on every episode under RAW_ROOT and write
# OUT_ROOT/<task>/variation<v>/episode<i>/<SUBDIR>/contact.json, then an aggregate
# OUT_ROOT/contact_summary.csv. Episodes with an existing contact.json are skipped (FORCE=1 redoes).
# Run from the repo root, in the background:
#
#   nohup bash scripts/run_contact_split_all.sh > data/entry_process/rlbench_single_seed42/contact_all.log 2>&1 &
#
# Env overrides: RAW_ROOT, OUT_ROOT, PROC_PY, SUBDIR (contact), FORCE=1, TASKS="a b",
# EXTRA_ARGS (passed through, e.g. "--eps 0.005 --no-video").
set -uo pipefail

RAW_ROOT=${RAW_ROOT:-data/raw/rlbench_single_seed42}
OUT_ROOT=${OUT_ROOT:-data/entry_process/rlbench_single_seed42}
PROC_PY=${PROC_PY:-$HOME/anaconda3/envs/entryvla/bin/python}
SUBDIR=${SUBDIR:-contact}
TASKS=${TASKS:-$(ls -d "$RAW_ROOT"/*/ | xargs -n1 basename)}

ok=0; skip=0; fail=0
for task in $TASKS; do
  for ep in $(ls -d "$RAW_ROOT/$task"/variation*/episodes/episode* 2>/dev/null | sort -V); do
    var=$(basename "$(dirname "$(dirname "$ep")")")
    out="$OUT_ROOT/$task/$var/$(basename "$ep")"
    if [ "${FORCE:-0}" != "1" ] && [ -f "$out/$SUBDIR/contact.json" ]; then skip=$((skip+1)); continue; fi
    echo "== $task/$var/$(basename "$ep")"; mkdir -p "$out"
    if "$PROC_PY" scripts/contact_phase_split.py --episode "$ep" --out "$out" --subdir "$SUBDIR" \
         ${EXTRA_ARGS:-} > "$out/$SUBDIR.log" 2>&1; then
      ok=$((ok+1)); tail -n1 "$out/$SUBDIR.log"
    else
      fail=$((fail+1)); echo "   FAILED, see $out/$SUBDIR.log"; tail -n3 "$out/$SUBDIR.log"
    fi
  done
done
echo "== ok=$ok skipped=$skip failed=$fail"

"$PROC_PY" - "$OUT_ROOT" "$SUBDIR" <<'PY'
import csv, glob, json, os, sys
root, sub = sys.argv[1:]
rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "variation*", "episode*", sub, "contact.json"))):
    j = json.load(open(f)); r = j["references"]
    task, var, ep = f.split(os.sep)[-5:-2]
    rows.append({"task": task, "variation": var, "episode": ep, "num_frames": j["num_frames"],
                 "first_contact_onset": j["first_contact_onset"], "n_intervals": len(j["contact_intervals"]),
                 "object_motion_onset": r["object_motion_onset"], "touch_force_onset": r["touch_force_onset"],
                 "onset_minus_object_motion": j["onset_minus_object_motion"]})
out = os.path.join(root, f"{sub}_summary.csv")
if rows:
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} episodes -> {out}")
PY
