#!/usr/bin/env python3
"""Print key=value overrides for one ablation condition: ablation_args.py <ablation.yaml> <condition>."""
import json
import sys

import yaml


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write(__doc__ + "\n")
        return 2
    spec = yaml.safe_load(open(sys.argv[1]))
    cond = spec["conditions"].get(sys.argv[2])
    if cond is None:
        sys.stderr.write(f"unknown condition; available: {sorted(spec['conditions'])}\n")
        return 2
    print(" ".join(f"{k}={json.dumps(v, separators=(',', ':'))}" for k, v in cond.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
