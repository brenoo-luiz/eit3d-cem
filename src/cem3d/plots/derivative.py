from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from cem3d.plots.paper import COLUMN, PALETTE, STYLE, frame, save

REFERENCE = "#7f7f7f"
DASHED = (0, (4, 2))
DOTTED = (0, (1, 2))


def render_convergence(steps: np.ndarray, errors: np.ndarray, path: Path) -> Path:
    k = np.arange(1, len(errors) + 1)
    best = int(np.argmin(errors))
    down, up = slice(0, best + 1), slice(best, None)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(COLUMN, 2.8))
        ax.semilogy(
            k, errors, color=PALETTE[0], marker="o", markersize=2.5, label=r"$a_k$"
        )
        ax.semilogy(
            k[down],
            errors[0] * steps[down] / steps[0],
            color=REFERENCE,
            linestyle=DASHED,
            linewidth=0.9,
            label=r"$\mathcal{O}(t_k)$",
        )
        ax.semilogy(
            k[up],
            errors[-1] * steps[-1] / steps[up],
            color=REFERENCE,
            linestyle=DOTTED,
            linewidth=0.9,
            label=r"$\mathcal{O}(t_k^{-1})$",
        )
        ax.axvline(k[best], color=REFERENCE, linestyle=DOTTED, linewidth=0.6)
        ax.annotate(
            f"$k = {k[best]}$",
            (k[best], errors[best]),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=8,
        )
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$a_k$")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, borderaxespad=0
        )
        frame(ax)
        return save(fig, path)
