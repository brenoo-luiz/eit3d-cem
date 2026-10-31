from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.typing import RcKeyType

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#222222"
FRAME = "#bfbfbf"
GRID = "#e3e3e3"
PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
COLUMN = 4.8

STYLE: dict[RcKeyType, Any] = {
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "STIXGeneral"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": FRAME,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "lines.linewidth": 1.1,
    "legend.frameon": True,
    "legend.edgecolor": FRAME,
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "figure.facecolor": "white",
}


def frame(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
