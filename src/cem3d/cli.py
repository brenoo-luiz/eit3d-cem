from __future__ import annotations

import argparse
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from cem3d.config import (
    OUTPUTS_DIR,
    ElectrodeConfig,
    MeshConfig,
    RowConfig,
    SetupConfig,
)
from cem3d.steps import consistency, derivative, electrodes, forward
from cem3d.utils import report


@dataclass(frozen=True)
class Step:
    output: str
    run: Callable[..., bool]


STEPS: dict[str, Step] = {
    "electrodes": Step("step1", electrodes.run),
    "forward": Step("step2", forward.run),
    "consistency": Step("step3", consistency.run),
    "derivative": Step("step4", derivative.run),
}
DIRECTIONS_STEP = "derivative"


def load_setup(path: Path) -> SetupConfig:
    with Path(path).open("rb") as file:
        data = tomllib.load(file)
    electrodes = dict(data.get("electrodes", {}))
    if "rows" in electrodes:
        electrodes["rows"] = tuple(RowConfig(**row) for row in electrodes["rows"])
    return SetupConfig(
        mesh=MeshConfig(**data.get("mesh", {})),
        electrodes=ElectrodeConfig(**electrodes),
        experiments=data.get("experiments"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="eit3d-cem steps")
    parser.add_argument("step", choices=STEPS, help="step to run")
    parser.add_argument("--force", action="store_true", help="regenerate the mesh")
    parser.add_argument(
        "--config", type=Path, help="TOML setup; outputs go to outputs/<file name>"
    )
    parser.add_argument(
        "--all-directions",
        action="store_true",
        help="derivative step only: run the a_k test for every direction (slow)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.all_directions and args.step != DIRECTIONS_STEP:
        parser.error(f"--all-directions only applies to {DIRECTIONS_STEP}")
    step = STEPS[args.step]
    options = {"all_directions": True} if args.all_directions else {}
    config = load_setup(args.config) if args.config else SetupConfig()
    outputs = OUTPUTS_DIR / args.config.stem if args.config else OUTPUTS_DIR
    with report.saved_to(outputs / f"{step.output}.txt"):
        ok = step.run(config=config, force=args.force, outputs=outputs, **options)
    return 0 if ok else 1
