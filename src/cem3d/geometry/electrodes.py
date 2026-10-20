from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from cem3d.config import ElectrodeConfig

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

    def area(self, radius: float) -> float:
        return radius * self.width * self.height


class ElectrodeLayout:
    def __init__(self, config: ElectrodeConfig) -> None:
        self._config = config
        self._electrodes = self._build()

    @property
    def config(self) -> ElectrodeConfig:
        return self._config

    def __len__(self) -> int:
        return len(self._electrodes)

    def __iter__(self) -> Iterator[Electrode]:
        return iter(self._electrodes)

    def __getitem__(self, index: int) -> Electrode:
        return self._electrodes[index - 1]

    @property
    def sector(self) -> float:
        return TWO_PI / self._config.count

    def _build(self) -> tuple[Electrode, ...]:
        cfg = self._config
        width = self.sector * cfg.coverage
        half_h = cfg.height / 2
        step = -self.sector if cfg.clockwise else self.sector
        return tuple(
            Electrode(
                index=j + 1,
                theta_start=cfg.rotation + j * step,
                theta_end=cfg.rotation + j * step + width,
                z_min=-half_h,
                z_max=half_h,
            )
            for j in range(cfg.count)
        )

    def locate(self, theta: float, z: float) -> int:
        """Index of the electrode containing the point (theta, z), 0 for none."""
        cfg = self._config
        if abs(z) > cfg.height / 2:
            return 0
        offset = (theta - cfg.rotation) % TWO_PI
        slot = int(offset // self.sector)
        if offset - slot * self.sector > self.sector * cfg.coverage:
            return 0
        return 1 + (-slot % cfg.count if cfg.clockwise else slot)
