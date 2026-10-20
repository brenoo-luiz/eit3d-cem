from __future__ import annotations

import argparse
from typing import Callable, Dict, Optional, Sequence

from cem3d.steps import electrodes

STEPS: Dict[str, Callable[..., bool]] = {
    "electrodes": electrodes.run,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="eit3d-cem steps")
    parser.add_argument("step", choices=STEPS, help="step to run")
    parser.add_argument("--force", action="store_true", help="regenerate the mesh")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return 0 if STEPS[args.step](force=args.force) else 1
