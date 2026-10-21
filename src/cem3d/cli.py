from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cem3d.config import OUTPUTS_DIR
from cem3d.steps import consistency, electrodes, forward
from cem3d.utils import report


@dataclass(frozen=True)
class Step:
    output: str
    run: Callable[..., bool]


STEPS: dict[str, Step] = {
    "electrodes": Step("step1", electrodes.run),
    "forward": Step("step2", forward.run),
    "consistency": Step("step3", consistency.run),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="eit3d-cem steps")
    parser.add_argument("step", choices=STEPS, help="step to run")
    parser.add_argument("--force", action="store_true", help="regenerate the mesh")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    step = STEPS[args.step]
    with report.saved_to(OUTPUTS_DIR / f"{step.output}.txt"):
        ok = step.run(force=args.force)
    return 0 if ok else 1
