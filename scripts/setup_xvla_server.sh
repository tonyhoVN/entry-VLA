#!/usr/bin/env bash
# Set up X-VLA fine-tuning on a Linux GPU server without sudo (run from the entry-VLA repo root).
#   - clones 2toINF/X-VLA at the pinned commit into $XVLA_ROOT (default ../X-VLA)
#   - creates conda env `xvla` (py3.10). X-VLA pins torch 2.5.1+cu124, which has no kernels for
#     Blackwell GPUs (sm_120); torch 2.7.1+cu128 is used instead. transformers stays <= 4.51.3 (X-VLA).
#
#   bash scripts/setup_xvla_server.sh
# Env overrides: CONDA_SH, XVLA_ROOT, XVLA_COMMIT, TORCH_INDEX.
set -e
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
XVLA_ROOT=${XVLA_ROOT:-$(cd .. && pwd)/X-VLA}
XVLA_COMMIT=${XVLA_COMMIT:-6bc2513}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
source "$CONDA_SH"
ENVS=$(cd "$(dirname "$(dirname "$(dirname "$CONDA_SH")")")/envs" && pwd)   # <conda root>/envs

[ -d "$XVLA_ROOT" ] || git clone -q https://github.com/2toinf/X-VLA.git "$XVLA_ROOT"
git -C "$XVLA_ROOT" checkout -q "$XVLA_COMMIT"
echo "X-VLA at $(git -C "$XVLA_ROOT" log --oneline -1)"

if ! conda env list | grep -q "^xvla "; then
  conda create -y -n xvla -c conda-forge python=3.10 pip
fi
PY=$ENVS/xvla/bin/python
$PY -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "$TORCH_INDEX"
$PY -m pip install "transformers==4.51.3" "accelerate==1.2.1" "mmengine==0.10.5" "timm==1.0.12" "einops==0.8.1" \
  "safetensors==0.4.5" "numpy==1.26.3" "scipy==1.15.0" "h5py==3.12.1" opencv-python-headless psutil tensorboard \
  "peft==0.17.1" "json_numpy==2.1.0" fastapi "uvicorn==0.34.3" "pyarrow==20.0.0" "av==15.0.0" "mediapy==1.2.4" pillow
$PY -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_arch_list()[-2:]); print('transformers', transformers.__version__)"
echo XVLA_SETUP_DONE
