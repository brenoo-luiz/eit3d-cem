from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cem3d.geometry.electrodes import TWO_PI
from cem3d.plots.paper import COLUMN, FRAME, PALETTE, STYLE, frame, save

ELECTRODE, GAP = PALETTE[1], PALETTE[0]
SHADE = "#f2f2f2"
MARKER = 1.6
LINE = 0.7
SPAN = 0.8


def _row_colors(count: int) -> list:
    palette = plt.get_cmap("tab20").colors
    return [palette[i % len(palette)] for i in range(count)]


def _rows(ax, values: np.ndarray) -> None:
    experiments, count = values.shape
    colors = _row_colors(experiments)
    for i, (row, color) in enumerate(zip(values, colors, strict=True), start=1):
        if i % 2 == 0:
            ax.axvspan(i - 0.1, i + SPAN + 0.1, color=SHADE, linewidth=0, zorder=0)
        ax.plot(
            np.linspace(i, i + SPAN, count),
            row,
            color=color,
            marker="o",
            markersize=MARKER,
            linewidth=LINE,
        )
    ax.axhline(0, color=FRAME, linewidth=0.6, zorder=1)
    centers = np.arange(1, experiments + 1)
    ax.set_xticks(centers + SPAN / 2, [str(i) for i in centers])
    ax.set_xlim(0.8, experiments + 1)
    ax.grid(axis="x", visible=False)
    ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))
    frame(ax)


def render_matrix(panels: list[tuple[np.ndarray, str]], path: Path) -> Path:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=(COLUMN, 1.7 * len(panels) + 0.4),
            sharex=True,
            squeeze=False,
            gridspec_kw={"hspace": 0.12},
        )
        axes = axes.ravel()
        for ax, (values, label) in zip(axes, panels, strict=True):
            _rows(ax, values)
            ax.set_ylabel(label)
        axes[-1].set_xlabel(r"experimento $i$")
        axes[-1].tick_params(axis="x", labelsize=7)
        fig.align_ylabels(axes)
        return save(fig, path)


def _band(points: np.ndarray, height: float, z: float) -> np.ndarray:
    radius = np.hypot(points[:, 0], points[:, 1])
    lateral = np.abs(radius - radius.max()) < 1e-6
    return lateral & (np.abs(points[:, 2] - z) < height)


def render_boundary(
    points: np.ndarray,
    values: list[np.ndarray],
    on_electrode: np.ndarray,
    band: float,
    label: str,
    path: Path,
    z: float = 0.0,
) -> Path:
    mask = _band(points, band, z)
    theta = np.arctan2(points[mask, 1], points[mask, 0]) % TWO_PI
    order = np.argsort(theta)
    theta, electrode = theta[order], on_electrode[mask][order]
    limit = 1.1 * max(np.abs(v[mask]).max() for v in values)
    columns = 4
    lines = int(np.ceil(len(values) / columns))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            lines,
            columns,
            figsize=(2 * COLUMN, 1.25 * lines + 0.5),
            sharex=True,
            sharey=True,
            squeeze=False,
            gridspec_kw={"hspace": 0.28, "wspace": 0.06},
        )
        for i, (ax, value) in enumerate(zip(axes.ravel(), values, strict=False), 1):
            y = value[mask][order]
            ax.plot(theta, y, color=FRAME, linewidth=0.4, zorder=1)
            ax.plot(
                theta[~electrode],
                y[~electrode],
                ".",
                color=GAP,
                markersize=MARKER,
                label="gaps",
            )
            ax.plot(
                theta[electrode],
                y[electrode],
                ".",
                color=ELECTRODE,
                markersize=MARKER,
                label="eletrodos",
            )
            ax.set_title(f"$i = {i}$", loc="left", fontsize=7, pad=2)
            ax.set_xlim(0, TWO_PI)
            ax.set_ylim(-limit, limit)
            ax.set_xticks([0, np.pi, TWO_PI], ["0", r"$\pi$", r"$2\pi$"])
            ax.grid(axis="x", visible=False)
            frame(ax)
        for ax in axes.ravel()[len(values) :]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel(r"$\theta$")
        fig.supylabel(label, x=0.06, fontsize=9)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            markerscale=3,
            bbox_to_anchor=(0.5, axes[0, 0].get_position().y1 + 0.035),
            borderaxespad=0,
        )
        return save(fig, path)


def render_contact(
    points: np.ndarray,
    profile: np.ndarray,
    voltage: float,
    difference: np.ndarray,
    band: float,
    path: Path,
    z: float = 0.0,
) -> Path:
    mask = np.abs(points[:, 2] - z) < band
    theta = np.arctan2(points[mask, 1], points[mask, 0])
    order = np.argsort(theta)
    with plt.rc_context(STYLE):
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(COLUMN, 3.9), gridspec_kw={"hspace": 0.5}
        )
        top.axhline(
            voltage,
            color=FRAME,
            linewidth=0.9,
            linestyle=(0, (4, 2)),
            label=r"$U^{(1)}_1$",
        )
        top.plot(
            theta[order],
            profile[mask][order],
            color=PALETTE[0],
            marker="o",
            markersize=MARKER + 0.6,
            linewidth=LINE + 0.2,
            label=r"$U^{\star(1)}_1(x)$",
        )
        top.set_xlabel(r"$\theta$ em $E_1$")
        top.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, borderaxespad=0
        )
        top.yaxis.get_major_formatter().set_useOffset(False)
        frame(top)
        _rows(bottom, difference)
        bottom.set_ylabel(r"$U - U^\star$")
        bottom.set_xlabel(r"experimento $i$")
        bottom.tick_params(axis="x", labelsize=7)
        return save(fig, path)
