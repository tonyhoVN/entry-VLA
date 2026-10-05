"""M3/M4 model commands: train-entry, eval-entry, check-models, train-scorer."""

from __future__ import annotations

import os

from ..utils.provenance import make_run_dir, run_manifest, seed_everything, write_json
from .common import add_common, emit, fail, load, module_available


def register(sub) -> None:
    p = sub.add_parser("train-entry", help="train the supervised paired-entry proposal model (user-run)")
    add_common(p, "configs/train/entry_4090.yaml")
    p.set_defaults(func=run_train_entry)

    p = sub.add_parser("eval-entry", help="offline entry-pose agreement vs demos (+nearest-demo/canonical baselines)")
    add_common(p, "configs/train/entry_4090.yaml")
    p.add_argument("--checkpoint", required=False)
    p.add_argument("--split", default="val", choices=["val", "test", "heldout"])
    p.set_defaults(func=run_eval_entry)

    p = sub.add_parser("check-models", help="tiny random-weight CPU forward pass (shapes, SO(3) decoding); no training")
    p.add_argument("--num-candidates", type=int, default=4)
    p.set_defaults(func=run_check_models)

    p = sub.add_parser("train-scorer", help="train the candidate outcome scorer on probe outcomes (user-run)")
    add_common(p, "configs/train/scorer_4090.yaml")
    p.set_defaults(func=run_train_scorer)


def run_train_entry(args) -> int:
    from ..models.train_entry import train, validate_config
    cfg = load(args)
    v = validate_config(cfg)
    if args.dry_run:
        emit({**v, "dry_run": True, "modules": module_available(["torch"])})
        return 0 if not v["problems"] else 1
    if v["problems"]:
        fail(str(v["problems"]))
    seed_everything(int(cfg["seed"]))
    run_dir = make_run_dir(cfg["train"].get("runs_root", "runs"), "entry_tiny" if cfg["train"].get("tiny_overfit") else "entry")
    write_json(os.path.join(run_dir, "manifest.json"), run_manifest(cfg))
    emit(train(cfg, run_dir))
    return 0


def run_eval_entry(args) -> int:
    from ..models.train_entry import evaluate, validate_config
    cfg = load(args)
    v = validate_config(cfg)
    if args.dry_run:
        emit({**v, "checkpoint": args.checkpoint, "split": args.split, "dry_run": True})
        return 0
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        fail("--checkpoint is required and must exist")
    if args.split == "test" and not cfg.get("eval", {}).get("allow_test", False):
        fail("test split evaluation requires eval.allow_test=true (do not tune on test outcomes)")
    emit(evaluate(cfg, args.checkpoint, args.split))
    return 0


def run_check_models(args) -> int:
    from ..models.train_entry import forward_check
    emit(forward_check(num_candidates=args.num_candidates))
    return 0


def run_train_scorer(args) -> int:
    from ..models.train_scorer import train, validate_config
    cfg = load(args)
    v = validate_config(cfg)
    if args.dry_run:
        emit({**v, "dry_run": True, "modules": module_available(["torch"])})
        return 0 if not v["problems"] else 1
    if v["problems"]:
        fail(str(v["problems"]))
    seed_everything(int(cfg["seed"]))
    run_dir = make_run_dir(cfg["train"].get("runs_root", "runs"), "scorer")
    write_json(os.path.join(run_dir, "manifest.json"), run_manifest(cfg))
    emit(train(cfg, run_dir))
    return 0
