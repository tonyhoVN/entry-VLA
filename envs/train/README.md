# Model-training environment (user-run setup)

This env is kept separate from both simulators: lerobot 0.6.1 needs Python >= 3.12, while
PerAct2 needs Python 3.8 and RCS needs Python 3.11. The envs exchange data only through the
versioned episode format (`entry_vla.episode/1`) written by the exporters.

| Component | Pin | Source |
|---|---|---|
| Python | 3.12 | lerobot `requires_python >=3.12` |
| lerobot[smolvla] | 0.6.1 (tag v0.6.1 = `7e241bd630a3719a56157a497ce5d08f244784f1`) | PyPI, GitHub |
| torch | `>=2.7,<2.12` per lerobot 0.6.1; resolved to 2.11.0+cu128 on this machine (driver 580, CUDA 13.0) | lerobot pyproject |
| transformers | `>=5.4.0,<5.6.0` (via the smolvla extra) | lerobot pyproject |
| numpy | `>=2.0,<2.3` | lerobot pyproject |
| checkpoint | `lerobot/smolvla_base` (HF revision `d9f33c94`; 6D state/action, cameras camera1..3 at 256x256) | HF |

The existing `vla` conda env (torch 2.13) is **outside** lerobot 0.6.1's torch range, so do not
reuse it for SmolVLA.

```bash
conda env create -f envs/train/environment.yml && conda activate entryvla   # or manually:
conda create -n entryvla -c conda-forge python=3.12 pip ffmpeg -y && conda activate entryvla
pip install "torch>=2.7,<2.12" torchvision --extra-index-url https://download.pytorch.org/whl/cu128
pip install "lerobot[smolvla,dataset]==0.6.1"
pip install -e .[test,train,viz]
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv   # record available VRAM
```

`lerobot-train` 0.6.1 has no gradient accumulation (Accelerator is built without
`gradient_accumulation_steps`), so `entry-vla train-smolvla` wraps the policy in its own bounded
loop with accumulation, AMP and optional freezing. It has not been executed.
