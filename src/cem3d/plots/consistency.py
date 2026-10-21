from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from cem3d.geometry.electrodes import TWO_PI
from cem3d.plots.paper import COLUMN, FRAME, PALETTE, STYLE, frame, save

ROWS = plt.get_cmap("tab20").colors
ELECTRODE, GAP = PALETTE[1], PALETTE[0]
MARKER = 2.2


def _rows(ax, values: np.ndarray) -> None:
    count = values.shape[1]
    for i, row in enumerate(values, start=1):
        x = np.linspace(i, i + 0.8, count)
        ax.plot(
            x,
            row,
            color=ROWS[(i - 1) % len(ROWS)],
            marker="o",
            markersize=MARKER,
            linewidth=0.8,
        )
    ax.axhline(0, color=FRAME, linewidth=0.6, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0.7, len(values) + 1)
    ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))
    frame(ax)


def render_matrix(panels: list[tuple[np.ndarray, str]], path: Path) -> Path:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=(COLUMN, 1.8 * len(panels) + 0.4),
            sharex=True,
            squeeze=False,
            gridspec_kw={"hspace": 0.15},
        )
        axes = axes.ravel()
        for ax, (values, label) in zip(axes, panels, strict=True):
            _rows(ax, values)
            ax.set_ylabel(label)
        axes[-1].set_xlabel(r"experimento $i$")
        fig.align_ylabels(axes)
        return save(fig, path)


def _band(points: np.ndarray, height: float) -> np.ndarray:
    radius = np.hypot(points[:, 0], points[:, 1])
    return (np.abs(radius - radius.max()) < 1e-6) & (np.abs(points[:, 2]) < height)


def render_flux(
    points: np.ndarray,
    fluxes: list[np.ndarray],
    on_electrode: np.ndarray,
    band: float,
    path: Path,
) -> Path:
    mask = _band(points, band)
    theta = np.arctan2(points[mask, 1], points[mask, 0]) % TWO_PI
    order = np.argsort(theta)
    theta, electrode = theta[order], on_electrode[mask][order]
    columns = 4
    lines = int(np.ceil(len(fluxes) / columns))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            lines,
            columns,
            figsize=(2 * COLUMN, 1.25 * lines + 0.5),
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0.35, "wspace": 0.08},
        )
        for i, (ax, flux) in enumerate(zip(axes.ravel(), fluxes, strict=False), 1):
            values = flux[mask][order]
            ax.plot(
                theta[electrode],
                values[electrode],
                ".",
                color=ELECTRODE,
                markersize=MARKER,
                label="eletrodos",
            )
            ax.plot(
                theta[~electrode],
                values[~electrode],
                ".",
                color=GAP,
                markersize=MARKER,
                label="gaps",
            )
            ax.set_title(rf"$\gamma\,\partial u^{{({i})}}/\partial\eta$", pad=3)
            ax.set_xlim(0, TWO_PI)
            ax.set_xticks([0, np.pi, TWO_PI], ["0", r"$\pi$", r"$2\pi$"])
            frame(ax)
        for ax in axes.ravel()[len(fluxes) :]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel(r"$\theta$")
        fig.subplots_adjust(top=0.88)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=2,
            markerscale=3,
        )
        return save(fig, path)


def render_contact(
    points: np.ndarray,
    profile: np.ndarray,
    voltage: float,
    difference: np.ndarray,
    band: float,
    path: Path,
) -> Path:
    mask = np.abs(points[:, 2]) < band
    theta = np.arctan2(points[mask, 1], points[mask, 0])
    order = np.argsort(theta)
    with plt.rc_context(STYLE):
        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(COLUMN, 4.0), gridspec_kw={"hspace": 0.45}
        )
        top.plot(
            theta[order],
            profile[mask][order],
            color=PALETTE[0],
            marker="o",
            markersize=MARKER,
            linewidth=0.8,
            label=r"$U^{\star(1)}_1(x)$",
        )
        top.axhline(
            voltage,
            color=FRAME,
            linewidth=0.9,
            linestyle=(0, (4, 2)),
            label=r"$U^{(1)}_1$",
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
        return save(fig, path)
