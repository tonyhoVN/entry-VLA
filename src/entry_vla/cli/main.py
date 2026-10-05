"""``entry-vla`` command-line entry point (``python -m entry_vla.cli.main``)."""

from __future__ import annotations

import argparse
import importlib
import sys

# subcommand modules; each exposes register(subparsers)
COMMANDS = [
    "smoke",
    "data_cmds",
    "policy_cmds",
    "model_cmds",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="entry-vla", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        importlib.import_module(f"entry_vla.cli.{name}").register(sub)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
