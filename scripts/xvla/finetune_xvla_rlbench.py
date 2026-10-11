#!/usr/bin/env python3
"""Fine-tune X-VLA on the RLBench manipulation phase (export: scripts/xvla/export_xvla_manip.py).

Registers the ``rlbench_manip`` domain (scripts/xvla/xvla_rlbench.py) into an unmodified X-VLA checkout,
then runs X-VLA's own train.py loop (group LRs, backbone freeze/warmup, checkpoints every
--save_interval). All train.py options pass through. Run with the `xvla` env, from any directory:

  CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --mixed_precision bf16 \
      scripts/xvla/finetune_xvla_rlbench.py --xvla-root ../X-VLA --models 2toINF/X-VLA-Pt \
      --train_metas_path data/xvla/rlbench_manip_close_microwave/meta_train.json \
      --output_dir runs/xvla_close_microwave --batch_size 16 --iters 3000 --freeze_steps 500 \
      --warmup_steps 500 --save_interval 1000 --learning_rate 1e-4 --learning_coef 0.1

  python scripts/xvla/finetune_xvla_rlbench.py --xvla-root ../X-VLA --dry-run --models X \
      --train_metas_path <meta_train.json>      # check meta + one sample, no model load
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--xvla-root", default=os.environ.get("XVLA_ROOT", "../X-VLA"))
    pre.add_argument("--domain-id", type=int, default=None, help="X-VLA soft-prompt domain id (default 11)")
    pre.add_argument("--dry-run", action="store_true")
    ours, rest = pre.parse_known_args()
    sys.path.insert(0, HERE)
    import xvla_rlbench
    xvla_rlbench.register(ours.xvla_root, ours.domain_id if ours.domain_id is not None else xvla_rlbench.DEFAULT_DOMAIN_ID)
    import train as xvla_train   # X-VLA train.py (repo root is on sys.path)

    parser = argparse.ArgumentParser("X-VLA fine-tuning on RLBench manipulation phase",
                                     parents=[xvla_train.get_args_parser()])
    args = parser.parse_args(rest)
    if ours.dry_run:
        from datasets.dataset import InfiniteDataReader
        r = InfiniteDataReader(args.train_metas_path, num_actions=30, num_views=3, training=True)
        s = next(iter(r))
        print({k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in s.items()})
        print("proprio", s["proprio"][:10].numpy().round(3), "action[0]", s["action"][0, :10].numpy().round(3))
        return
    os.makedirs(args.output_dir, exist_ok=True)
    xvla_train.main(args)


if __name__ == "__main__":
    main()
