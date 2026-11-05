from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag

from cem3d.geometry.electrodes import ElectrodeLayout


@dataclass(frozen=True)
class CurrentProtocol:
    currents: Callable[[ElectrodeLayout], np.ndarray]
    scope: str


def adjacent_pattern(count: int, experiments: int | None = None) -> np.ndarray:
    experiments = count if experiments is None else experiments
    if not 1 <= experiments <= count:
        raise ValueError("experiments must be in [1, count]")
    currents = np.zeros((experiments, count))
    rows = np.arange(experiments)
    currents[rows, rows] = 1.0
    currents[rows, (rows + 1) % count] = -1.0
    return currents


def adjacent_by_row(counts: Sequence[int]) -> np.ndarray:
    return block_diag(*(adjacent_pattern(count) for count in counts))


def _adjacent_by_row(layout: ElectrodeLayout) -> np.ndarray:
    return adjacent_by_row([len(row) for row in layout.rows])


def _adjacent_chain(layout: ElectrodeLayout) -> np.ndarray:
    return adjacent_pattern(len(layout))


PROTOCOLS: dict[str, CurrentProtocol] = {
    "adjacent_by_row": CurrentProtocol(_adjacent_by_row, "cada linha"),
    "adjacent_chain": CurrentProtocol(_adjacent_chain, "uma cadeia"),
}


def get_protocol(name: str) -> CurrentProtocol:
    if name not in PROTOCOLS:
        raise ValueError(f"unknown protocol {name!r}, choose from {sorted(PROTOCOLS)}")
    return PROTOCOLS[name]


def driving_electrodes(currents: np.ndarray) -> list[int]:
    return [int(np.argmax(row)) + 1 for row in np.atleast_2d(currents)]
