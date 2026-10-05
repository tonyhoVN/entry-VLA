# PerAct2 bimanual RLBench environment (user-run setup)

The agent installed this env as below (no demos generated). Upstream `stepjam/RLBench`/`PyRep`
do **not** support the bimanual tasks. Use the forks:

| Component | Pin |
|---|---|
| Python | 3.8 |
| CoppeliaSim | 4.1.0 (`CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz`) |
| markusgrotz/PyRep | `b8bd1d7a3182adcd570d001649c0849047ebf197` (main) |
| markusgrotz/RLBench | `8af748c51287989294e00c9c670e3330a0e35ed5` (**main**; `master` is a 2022 branch, and the `peract` branch linked from INSTALLATION.md does not exist) |
| markusgrotz/YARR | `6822ff78602c77878b27d4cfe759ce029c67bffb` (only for PerAct2 baselines) |
| markusgrotz/peract_bimanual | `bb0232a6ba3fe116566e9568f0c7af980ed6703d` (reference only) |

```bash
conda create -n peract2 python=3.8 -y && conda activate peract2
wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $HOME
export COPPELIASIM_ROOT=$HOME/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
conda install -c conda-forge cffi==1.14.2 -y
pip install "numpy<1.25" pyyaml pytest
git clone https://github.com/markusgrotz/PyRep.git ext/PyRep && git -C ext/PyRep checkout b8bd1d7a3182adcd570d001649c0849047ebf197
pip install --no-build-isolation -e ext/PyRep    # setup.py imports cffi at build time
git clone https://github.com/markusgrotz/RLBench.git ext/RLBench && git -C ext/RLBench checkout 8af748c51287989294e00c9c670e3330a0e35ed5
pip install -e ext/RLBench
# imported by the fork but not declared in its pyproject (natsort: rlbench.utils; absl: dataset generators)
pip install natsort absl-py "opencv-python-headless<4.11" "gym==0.26.2"
pip install -e .[test,io]
pip check
```

Installed on 2026-10-05 this way: Python 3.8.10, numpy 1.24.4, Pillow 9.5.0, PyRep 4.1.0.3, rlbench 1.2.1.
Not installed: `yarr` (only `tools/create_videos.py`, `task_builder_bimanual.py`) and `open3d`
(only `tools/visualize_dataset.py`). If a conda create is interrupted and later fails with
`JSONDecodeError`, delete the cache dirs in `~/anaconda3/pkgs` whose `info/index.json` is empty.

The fork's `MoveArmThenGripper` always passes `ignore_collisions` to the arm mode; only the
`*ViaPlanning` modes accept it, so the single-arm examples using `JointVelocity` fail on `step()`.

Headless rendering needs an X server or a virtual display (for example `xvfb-run -a`).

## Data (selective download; not executed)

Each task/split is hosted as one squashfs image with SHA256SUM files:
`https://dataset.cs.washington.edu/fox/bimanual/image_size_{128|256}/<task>.{train|val|test}.squashfs`
(~180 GB total at 128 px; `bimanual_handover_item_easy` train at 128 px is about 6 GB).

```bash
mkdir -p data/raw/peract2 && cd data/raw/peract2
wget https://dataset.cs.washington.edu/fox/bimanual/image_size_128/bimanual_handover_item_easy.val.squashfs
wget https://dataset.cs.washington.edu/fox/bimanual/image_size_128/SHA256SUM && sha256sum -c --ignore-missing SHA256SUM
unsquashfs -d bimanual_handover_item_easy.val bimanual_handover_item_easy.val.squashfs
```

Episode layout: `<task>/all_variations/episodes/episode<N>/{low_dim_obs.pkl, variation_number.pkl,
variation_descriptions.pkl, <cam>_rgb/rgb_0000.png, <cam>_depth/depth_0000.png, <cam>_mask/mask_0000.png}`.
Depth PNGs are 24-bit RGB-packed **linear** buffers (`DEPTH_SCALE=2**24-1`, `depth_in_meters=False`);
metric = near + d*(far-near), with near/far taken from `obs.misc`. Demonstrations are scripted
waypoint and motion-planning rollouts, **not** teleoperation.

## Smoke (user-run)

```bash
xvfb-run -a entry-vla smoke --config configs/smoke.yaml backend=peract2 --out runs/smoke_peract2
```
