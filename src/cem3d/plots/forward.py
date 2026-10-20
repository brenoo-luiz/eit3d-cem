from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from cem3d.plots.paper import COLUMN, FRAME, PALETTE, STYLE, frame, legend_above, save


def render_voltages(
    homogeneous: np.ndarray, inclusion: np.ndarray, pattern: int, path: Path
) -> Path:
    electrodes = np.arange(1, homogeneous.size + 1)
    with plt.rc_context(STYLE):
        fig, (top, bottom) = plt.subplots(
            2,
            1,
            figsize=(COLUMN, 3.6),
            sharex=True,
            gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.12},
        )
        top.plot(
            electrodes,
            homogeneous,
            color=PALETTE[0],
            linewidth=2.4,
            marker="o",
            markersize=4,
            label=r"$\gamma \equiv 1$",
        )
        top.plot(
            electrodes,
            inclusion,
            color=PALETTE[1],
            marker="o",
            markersize=2.6,
            label=r"$\gamma$ com inclusão",
        )
        top.set_ylabel(rf"$U^{{({pattern})}}_j$")
        bottom.plot(
            electrodes,
            inclusion - homogeneous,
            color=PALETTE[2],
            marker="o",
            markersize=2.6,
        )
        bottom.axhline(0, color=FRAME, linewidth=0.8)
        bottom.set_ylabel(r"$\Delta U_j$")
        bottom.set_xlabel(r"eletrodo $j$")
        bottom.xaxis.set_major_locator(MaxNLocator(integer=True))
        bottom.set_xlim(0.5, electrodes[-1] + 0.5)
        for ax in (top, bottom):
            frame(ax)
        fig.align_ylabels((top, bottom))
        legend_above(top, ncol=2)
        return save(fig, path)


def render_difference(difference: np.ndarray, path: Path) -> Path:
    limit = np.abs(difference).max()
    n_patterns, n_electrodes = difference.shape
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(COLUMN, COLUMN))
        image = ax.imshow(
            difference,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(0.5, n_electrodes + 0.5, n_patterns + 0.5, 0.5),
        )
        ax.set_xticks(range(1, n_electrodes + 1, 3))
        ax.set_yticks(range(1, n_patterns + 1, 3))
        ax.set_xlabel(r"eletrodo $j$")
        ax.set_ylabel(r"experimento $i$")
        ax.set_title(r"$U^{(i)}_j(\gamma) - U^{(i)}_j(1)$")
        ax.grid(False)
        frame(ax)
        bar = fig.colorbar(
            image, ax=ax, orientation="horizontal", fraction=0.05, pad=0.12, aspect=40
        )
        bar.outline.set_edgecolor(FRAME)
        bar.ax.tick_params(size=0)
        bar.formatter.set_powerlimits((-2, 2))
        return save(fig, path)
