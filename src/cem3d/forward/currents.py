from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.linalg import block_diag


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


def driving_electrodes(currents: np.ndarray) -> list[int]:
    return [int(np.argmax(row)) + 1 for row in np.atleast_2d(currents)]
