#!/usr/bin/env bash
# Two-phase rollout of a fine-tuned X-VLA on a NEW RLBench episode (seed outside the training seeds):
#   1) record a scripted demo for the seed (scene restore),  2) contact split (oracle entry, for reference),
#   3) with ENTRY_CKPT: predict the entry from frame 0 (Qwen3-VL + SAM 2.1 object cloud -> entry diffusion),
#   4) start the X-VLA policy server (GPU),  5) planner -> entry pose (predicted if ENTRY_CKPT, else oracle),
#      then X-VLA closed loop; video + json.
# Run from the repo root (server):  bash scripts/xvla/run_xvla_rollout.sh
# Env overrides: TASK (close_microwave), VAR (0), SEED (random 1000-9999), CKPT, PORT (6010), PRE_FRAMES (10),
#   GPU (1), EXEC_STEPS (10), MAX_QUERIES (40), ENTRY_CKPT (entry diffusion best.pt; unset = oracle entry),
#   ENTRY_PY (entrytrain env python), OUT_TAG (suffix of the output folder), RAW_ROOT, PROC_ROOT, OUT_ROOT, SIM_PY, PROC_PY, XVLA_PY,
#   DISPLAY / XAUTHORITY (CoppeliaSim needs an X display), COPPELIASIM_ROOT.
set -uo pipefail
TASK=${TASK:-close_microwave}; VAR=${VAR:-0}; SEED=${SEED:-$((1000 + RANDOM % 9000))}
CKPT=${CKPT:-runs/xvla_${TASK}/ckpt-3000}; PORT=${PORT:-6010}; PRE_FRAMES=${PRE_FRAMES:-10}; GPU=${GPU:-1}
EXEC_STEPS=${EXEC_STEPS:-10}; MAX_QUERIES=${MAX_QUERIES:-40}
RAW_ROOT=${RAW_ROOT:-data/raw/rlbench_single}
PROC_ROOT=${PROC_ROOT:-data/entry_process/xvla_rollout}
OUT_ROOT=${OUT_ROOT:-runs/xvla_${TASK}/rollouts}
SIM_PY=${SIM_PY:-$HOME/miniconda3/envs/peract2/bin/python}
PROC_PY=${PROC_PY:-$HOME/miniconda3/envs/entryproc/bin/python}
XVLA_PY=${XVLA_PY:-$HOME/miniconda3/envs/xvla/bin/python}
ENTRY_PY=${ENTRY_PY:-$HOME/miniconda3/envs/entrytrain/bin/python}
ENTRY_CKPT=${ENTRY_CKPT:-}
export DISPLAY=${DISPLAY:-:1} XAUTHORITY=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
export COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-$HOME/ws_tony/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export PATH=$(dirname "$PROC_PY"):$PATH
EP=$RAW_ROOT/$TASK/variation$VAR/seed$SEED/episodes/episode0
PROC=$PROC_ROOT/$TASK/variation$VAR/seed$SEED/episode0
OUT=$OUT_ROOT/seed$SEED${ENTRY_CKPT:+_predicted_entry}${OUT_TAG:+_$OUT_TAG}
mkdir -p "$PROC" "$OUT"
echo "== $TASK variation $VAR seed $SEED (training seeds were 1-17; val 18-20) ckpt $CKPT"

if [ ! -f "$EP/low_dim_obs.pkl" ]; then
  echo "== record scripted demo (scene + entry label)"
  "$SIM_PY" scripts/rlbench_single_generator.py --task "$TASK" --variation "$VAR" --episodes 1 --seed "$SEED" \
    --image_size 256 256 --save_path "$RAW_ROOT" > "$OUT/generate.log" 2>&1 || { echo "generate failed"; exit 1; }
fi
echo "== contact split (pre-frames $PRE_FRAMES)"
"$PROC_PY" scripts/contact_phase_split.py --episode "$EP" --out "$PROC" --pre-frames "$PRE_FRAMES" --no-video \
  > "$OUT/contact.log" 2>&1 || { echo "contact split failed"; exit 1; }

ENTRY_ARGS=()
if [ -n "$ENTRY_CKPT" ]; then
  echo "== predict entry from frame 0 ($ENTRY_CKPT)"
  CUDA_VISIBLE_DEVICES=$GPU "$ENTRY_PY" scripts/predict_entry_diffusion.py --ckpt "$ENTRY_CKPT" --episode "$EP" \
    --out "$OUT/entry_prediction.json" > "$OUT/predict.log" 2>&1 || { echo "entry prediction failed"; tail "$OUT/predict.log"; exit 1; }
  tail -n 2 "$OUT/predict.log"
  ENTRY_ARGS=(--entry-json "$OUT/entry_prediction.json")
fi

echo "== start X-VLA server on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU "$XVLA_PY" scripts/xvla/xvla_policy_server.py --xvla-root ../X-VLA --ckpt "$CKPT" \
  --port "$PORT" > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT
for _ in $(seq 1 120); do grep -q "server ready" "$OUT/server.log" && break; sleep 2; done
grep -q "server ready" "$OUT/server.log" || { echo "server did not start"; tail "$OUT/server.log"; exit 1; }

echo "== rollout: planner -> entry, then X-VLA"
"$SIM_PY" scripts/xvla/rollout_xvla_rlbench.py --task "$TASK" --episode "$EP" --contact "$PROC/contact/contact.json" \
  "${ENTRY_ARGS[@]}" --port "$PORT" --exec-steps "$EXEC_STEPS" --max-queries "$MAX_QUERIES" --out "$OUT" > "$OUT/rollout.log" 2>&1
rc=$?
tail -n 25 "$OUT/rollout.log"
echo "== video: $OUT/rollout.mp4 (exit $rc)"
