#!/usr/bin/env bash
# Generate N single-arm RLBench demos, then for every episode run the state-based
# transition detector and the perception method (Qwen3-VL + SAM 2.1 + distance field),
# compared with the privileged simulator reference. Run from the repo root.
#
#   bash scripts/run_rlbench_perception_batch.sh push_button 4 20 42
#
# Args: TASK VARIATION N_EPISODES SEED. Env overrides: RAW_ROOT, OUT_ROOT, SIM_PY, PROC_PY,
# SKIP_GENERATE=1 (reuse existing demos), IMAGE_SIZE (default 256).
set -euo pipefail

TASK=${1:?task}; VAR=${2:?variation}; N=${3:?episodes}; SEED=${4:?seed}
IMAGE_SIZE=${IMAGE_SIZE:-256}
RAW_ROOT=${RAW_ROOT:-data/raw/rlbench_single_seed${SEED}}
OUT_ROOT=${OUT_ROOT:-data/entry_process/${TASK}_seed${SEED}/variation${VAR}}
SIM_PY=${SIM_PY:-$HOME/anaconda3/envs/peract2/bin/python}
PROC_PY=${PROC_PY:-$HOME/anaconda3/envs/entryvla/bin/python}
EP_ROOT=$RAW_ROOT/$TASK/variation$VAR/episodes

if [ "${SKIP_GENERATE:-0}" != "1" ]; then
  export COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-$HOME/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT
  export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
  echo "== generate $N demos: $TASK variation $VAR seed $SEED -> $RAW_ROOT"
  "$SIM_PY" scripts/rlbench_single_generator.py --task "$TASK" --variation "$VAR" --episodes "$N" \
    --seed "$SEED" --image_size "$IMAGE_SIZE" "$IMAGE_SIZE" --save_path "$RAW_ROOT"
fi

echo "== state-based transition detector (stopped keyframe / contact)"
for ep in $(ls -d "$EP_ROOT"/episode* | sort -V); do
  "$PROC_PY" scripts/detect_transition_rlbench.py --episode "$ep" --out "$OUT_ROOT/$(basename "$ep")" --no-video >/dev/null
done

echo "== perception method vs simulator reference"
"$PROC_PY" scripts/perception_phase_split.py --episodes-root "$EP_ROOT" --out "$OUT_ROOT"

echo "== done: $OUT_ROOT/batch_summary.{json,csv}"
