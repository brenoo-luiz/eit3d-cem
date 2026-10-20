from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT_DIR / "cache"
OUTPUTS_DIR = ROOT_DIR / "outputs"

TOP_TAG = 1
BOTTOM_TAG = 2
GAP_TAG = 3
VOLUME_TAG = 10
ELECTRODE_TAG0 = 100


def electrode_tag(index: int) -> int:
    return ELECTRODE_TAG0 + index


@dataclass(frozen=True)
class ElectrodeConfig:
    count: int = 16
    coverage: float = 0.5
    rotation: float = 1.0
    clockwise: bool = True
    height: float = 1.0

    def __post_init__(self) -> None:
        if self.count < 2:
            raise ValueError("count must be at least 2")
        if not 0 < self.coverage < 1:
            raise ValueError("coverage must be in (0, 1)")
        if self.height <= 0:
            raise ValueError("height must be positive")
        object.__setattr__(self, "coverage", float(self.coverage))
        object.__setattr__(self, "rotation", float(self.rotation))
        object.__setattr__(self, "height", float(self.height))


@dataclass(frozen=True)
class MeshConfig:
    radius: float = 1.0
    height: float = 2.0
    size_max: float = 0.1
    size_min: float = 0.03

    def __post_init__(self) -> None:
        for name in ("radius", "height", "size_max", "size_min"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.radius <= 0 or self.height <= 0:
            raise ValueError("radius and height must be positive")
        if not 0 < self.size_min <= self.size_max:
            raise ValueError("need 0 < size_min <= size_max")


@dataclass(frozen=True)
class SetupConfig:
    mesh: MeshConfig = field(default_factory=MeshConfig)
    electrodes: ElectrodeConfig = field(default_factory=ElectrodeConfig)

    def __post_init__(self) -> None:
        if self.electrodes.height >= self.mesh.height:
            raise ValueError("electrodes must be shorter than the cylinder")
