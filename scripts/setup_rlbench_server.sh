#!/usr/bin/env bash
# Set up RLBench single-arm data collection + contact processing on a Linux workstation without sudo.
# Run from the repo root after cloning the forks into ext/ and placing CoppeliaSim 4.1 (Ubuntu 20.04
# build) at $COPPELIASIM_ROOT:
#
#   git clone https://github.com/markusgrotz/PyRep.git ext/PyRep   && git -C ext/PyRep checkout b8bd1d7a
#   git clone https://github.com/markusgrotz/RLBench.git ext/RLBench && git -C ext/RLBench checkout 8af748c5
#   bash scripts/setup_rlbench_server.sh
#
# Creates conda envs `peract2` (CPython 3.8, simulator) and `entryproc` (py3.12, contact processing,
# ffmpeg from conda-forge). Env overrides: CONDA_SH, COPPELIASIM_ROOT.
set -e
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
export COPPELIASIM_ROOT=${COPPELIASIM_ROOT:-$HOME/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
source "$CONDA_SH"
ENVS=$(dirname "$(dirname "$CONDA_SH")")/../envs   # <conda root>/envs
ENVS=$(cd "$ENVS" && pwd)
# absolute interpreter paths: a venv or other python earlier on PATH must not receive the packages

if ! conda env list | grep -q "^peract2 "; then
  conda create -y -n peract2 -c conda-forge "python=3.8.*=*_cpython" pip
fi
PY=$ENVS/peract2/bin/python
$PY -m pip -V >/dev/null 2>&1 || $PY -m ensurepip --upgrade
$PY -m pip install "numpy==1.24.4" "scipy==1.10.1" "Pillow==9.5.0" "opencv-python-headless==4.10.0.84" "cffi==1.14.2" \
  "pyquaternion==0.9.9" "natsort==8.4.0" "gym==0.26.2" "PyYAML==6.0.3" "click==8.1.8" "pyarrow==17.0.0"
$PY -m pip install -e ext/PyRep
$PY -m pip install -e ext/RLBench --no-deps
$PY -c "import pyrep, rlbench; print('peract2 ok')"

if ! conda env list | grep -q "^entryproc "; then
  conda create -y -n entryproc -c conda-forge python=3.12 pip ffmpeg
fi
PY=$ENVS/entryproc/bin/python
$PY -m pip -V >/dev/null 2>&1 || $PY -m ensurepip --upgrade
$PY -m pip install numpy scipy pillow matplotlib opencv-python-headless h5py pyyaml
$PY -c "import numpy, scipy, cv2, matplotlib; print('entryproc ok')"
echo SETUP_DONE
