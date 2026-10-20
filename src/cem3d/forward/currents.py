from __future__ import annotations

import numpy as np


def adjacent_pattern(count: int, experiments: int | None = None) -> np.ndarray:
    # Row i: +1 on e_{i+1}, -1 on e_{i+2}, cyclic.
    experiments = count if experiments is None else experiments
    if not 1 <= experiments <= count:
        raise ValueError("experiments must be in [1, count]")
    currents = np.zeros((experiments, count))
    rows = np.arange(experiments)
    currents[rows, rows] = 1.0
    currents[rows, (rows + 1) % count] = -1.0
    return currents
