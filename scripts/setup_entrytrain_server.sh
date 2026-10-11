#!/usr/bin/env bash
# Conda env `entrytrain` on a GPU server without sudo, for Qwen3-VL + SAM 2.1 detection, frozen VLM features
# and the entry diffusion model (needs transformers >= 4.57, unlike the X-VLA env). Mirrors the local
# `entryvla` env (torch cu128 wheels support Blackwell). Model weights must already be in the HF cache
# (copy ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct and models--facebook--sam2.1-hiera-small).
#   bash scripts/setup_entrytrain_server.sh
set -e
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
source "$CONDA_SH"
ENVS=$(cd "$(dirname "$(dirname "$(dirname "$CONDA_SH")")")/envs" && pwd)   # <conda root>/envs
conda env list | grep -q "^entrytrain " || conda create -y -n entrytrain -c conda-forge python=3.12 pip ffmpeg
PY=$ENVS/entrytrain/bin/python
$PY -m pip install torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX"
$PY -m pip install "transformers==5.5.4" "accelerate==1.15.0" "scipy==1.18.1" "numpy==2.2.6" pillow \
  opencv-python-headless matplotlib h5py safetensors
$PY -c "import torch, transformers; from transformers import Qwen3VLForConditionalGeneration, Sam2Model; print('torch', torch.__version__, torch.cuda.is_available(), 'transformers', transformers.__version__)"
echo ENTRYTRAIN_SETUP_DONE
