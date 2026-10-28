from __future__ import annotations

import math
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

DEFAULT_ELECTRODE_ASPECT = 16 / math.pi
DEFAULT_THETA = math.degrees(1.0)


def electrode_tag(index: int) -> int:
    return ELECTRODE_TAG0 + index


@dataclass(frozen=True)
class RowConfig:
    count: int = 16
    coverage: float = 0.5
    theta: float = DEFAULT_THETA
    clockwise: bool = True

    def __post_init__(self) -> None:
        if self.count < 2:
            raise ValueError("count must be at least 2")
        if not 0 < self.coverage < 1:
            raise ValueError("coverage must be in (0, 1)")
        object.__setattr__(self, "coverage", float(self.coverage))
        object.__setattr__(self, "theta", float(self.theta))

    def width(self, radius: float) -> float:
        """Arc length of one electrode."""
        return self.coverage * math.tau * radius / self.count


@dataclass(frozen=True)
class ElectrodeConfig:
    aspect: float = DEFAULT_ELECTRODE_ASPECT  # electrode height / width
    rows: tuple[RowConfig, ...] = (RowConfig(),)  # top to bottom

    def __post_init__(self) -> None:
        object.__setattr__(self, "aspect", float(self.aspect))
        object.__setattr__(self, "rows", tuple(self.rows))
        if self.aspect <= 0:
            raise ValueError("aspect must be positive")
        if not self.rows:
            raise ValueError("need at least one row of electrodes")

    @property
    def count(self) -> int:
        return sum(row.count for row in self.rows)

    def heights(self, radius: float) -> tuple[float, ...]:
        return tuple(self.aspect * row.width(radius) for row in self.rows)


@dataclass(frozen=True)
class MeshConfig:
    radius: float = 1.0
    aspect: float = 2.0  # cylinder height / radius
    size_max: float = 0.1
    size_min: float = 0.03

    def __post_init__(self) -> None:
        for name in ("radius", "aspect", "size_max", "size_min"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.radius <= 0 or self.aspect <= 0:
            raise ValueError("radius and aspect must be positive")
        if not 0 < self.size_min <= self.size_max:
            raise ValueError("need 0 < size_min <= size_max")

    @property
    def height(self) -> float:
        return self.aspect * self.radius


@dataclass(frozen=True)
class SetupConfig:
    mesh: MeshConfig = field(default_factory=MeshConfig)
    electrodes: ElectrodeConfig = field(default_factory=ElectrodeConfig)

    def __post_init__(self) -> None:
        if sum(self.electrodes.heights(self.mesh.radius)) >= self.mesh.height:
            raise ValueError("the rows of electrodes must fit in the cylinder")


@dataclass(frozen=True)
class CEMConfig:
    impedance: float = 1e-3
    degree: int = 1
    rtol: float = 1e-12
    max_it: int = 1000

    def __post_init__(self) -> None:
        object.__setattr__(self, "impedance", float(self.impedance))
        if self.impedance <= 0:
            raise ValueError("impedance must be positive")
        if self.degree < 1:
            raise ValueError("degree must be at least 1")
        if self.rtol <= 0 or self.max_it < 1:
            raise ValueError("invalid solver tolerance")


@dataclass(frozen=True)
class InclusionConfig:
    center: tuple[float, float, float] = (-0.3, -0.3, 0.0)
    radius: float = 0.3
    inside: float = 2.0
    outside: float = 1.0

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.inside <= 0 or self.outside <= 0:
            raise ValueError("conductivity must be positive")
