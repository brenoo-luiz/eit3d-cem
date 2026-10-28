from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from cem3d.config import RowConfig, SetupConfig

TWO_PI = math.tau


@dataclass(frozen=True)
class Electrode:
    index: int
    theta_start: float
    theta_end: float
    z_min: float
    z_max: float

    @property
    def width(self) -> float:
        return self.theta_end - self.theta_start

    @property
    def theta_center(self) -> float:
        return (self.theta_start + self.theta_end) / 2

    @property
    def height(self) -> float:
        return self.z_max - self.z_min

    @property
    def z_center(self) -> float:
        return (self.z_min + self.z_max) / 2

    def area(self, radius: float) -> float:
        return radius * self.width * self.height


class ElectrodeRow:
    def __init__(
        self, config: RowConfig, number: int, first: int, z_min: float, z_max: float
    ) -> None:
        self._config = config
        self._number = number
        self._first = first
        self._theta = math.radians(config.theta)
        self.z_min = z_min
        self.z_max = z_max
        self._electrodes = tuple(self._electrode(j) for j in range(config.count))

    @property
    def config(self) -> RowConfig:
        return self._config

    @property
    def number(self) -> int:
        return self._number

    @property
    def first(self) -> int:
        return self._first

    @property
    def sector(self) -> float:
        return TWO_PI / self._config.count

    @property
    def width(self) -> float:
        return self.sector * self._config.coverage

    @property
    def sense(self) -> int:
        return -1 if self._config.clockwise else 1

    @property
    def z_center(self) -> float:
        return (self.z_min + self.z_max) / 2

    def __len__(self) -> int:
        return len(self._electrodes)

    def __iter__(self) -> Iterator[Electrode]:
        return iter(self._electrodes)

    def _electrode(self, j: int) -> Electrode:
        lead = self.sense * (self._theta + j * self.sector)
        trail = lead + self.sense * self.width
        return Electrode(
            index=self._first + j,
            theta_start=min(lead, trail),
            theta_end=max(lead, trail),
            z_min=self.z_min,
            z_max=self.z_max,
        )

    def _travelled(self, theta: float) -> float:
        return (self.sense * theta - self._theta) % TWO_PI % TWO_PI

    def contains(self, z: float) -> bool:
        return self.z_min <= z <= self.z_max

    def locate(self, theta: float) -> int:
        travelled = self._travelled(theta)
        slot = int(travelled // self.sector)
        if travelled - slot * self.sector > self.width:
            return 0
        return self._first + slot

    def clearance(self, theta: float) -> float:
        offset = self._travelled(theta) % self.sector
        if offset <= self.width:
            return 0.0
        return min(offset - self.width, self.sector - offset)

    def gap_centers(self) -> list[float]:
        middle = self._theta + (self.width + self.sector) / 2
        return [self.sense * (middle + j * self.sector) for j in range(len(self))]


class ElectrodeLayout:
    def __init__(self, config: SetupConfig) -> None:
        heights = config.electrodes.heights(config.mesh.radius)
        self._vertical_gap = (config.mesh.height - sum(heights)) / (len(heights) + 1)
        self._rows = self._build(config, heights)
        self._electrodes = tuple(e for row in self._rows for e in row)

    @property
    def rows(self) -> tuple[ElectrodeRow, ...]:
        return self._rows

    @property
    def vertical_gap(self) -> float:
        return self._vertical_gap

    def __len__(self) -> int:
        return len(self._electrodes)

    def __iter__(self) -> Iterator[Electrode]:
        return iter(self._electrodes)

    def __getitem__(self, index: int) -> Electrode:
        return self._electrodes[index - 1]

    def _build(
        self, config: SetupConfig, heights: tuple[float, ...]
    ) -> tuple[ElectrodeRow, ...]:
        rows = []
        top = config.mesh.height / 2 - self._vertical_gap
        first = 1
        for number, (row, height) in enumerate(
            zip(config.electrodes.rows, heights, strict=True), start=1
        ):
            rows.append(ElectrodeRow(row, number, first, top - height, top))
            first += row.count
            top -= height + self._vertical_gap
        return tuple(rows)

    def row_of(self, index: int) -> ElectrodeRow:
        for row in self._rows:
            if row.first <= index < row.first + len(row):
                return row
        raise IndexError(f"no electrode {index}")

    def locate(self, theta: float, z: float) -> int:
        for row in self._rows:
            if row.contains(z):
                return row.locate(theta)
        return 0

    def clearance(self, theta: float) -> float:
        return min(row.clearance(theta) for row in self._rows)

    @property
    def seam(self) -> float:
        candidates = [theta for row in self._rows for theta in row.gap_centers()]
        return max(candidates, key=self.clearance)
