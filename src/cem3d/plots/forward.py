from __future__ import annotations

from pathlib import Path

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from cem3d.config import InclusionConfig
from cem3d.geometry.electrodes import TWO_PI, ElectrodeLayout
from cem3d.plots.paper import COLUMN, FRAME, INK, STYLE, save
from cem3d.utils.sampling import horizontal_slice

CONTOURS = 17
SATURATION = 0.6


def _electrodes(ax, layout: ElectrodeLayout, radius: float, z: float) -> None:
    t = np.linspace(0, TWO_PI, 361)
    ax.plot(radius * np.cos(t), radius * np.sin(t), color=FRAME, linewidth=0.8)
    for e in layout:
        if not e.z_min <= z <= e.z_max:
            continue
        arc = np.linspace(e.theta_start, e.theta_end, 30)
        ax.plot(
            radius * np.cos(arc),
            radius * np.sin(arc),
            color=INK,
            linewidth=2.6,
            solid_capstyle="butt",
        )
        c = e.theta_center
        ax.text(
            1.17 * radius * np.cos(c),
            1.17 * radius * np.sin(c),
            f"$e_{{{e.index}}}$",
            ha="center",
            va="center",
            fontsize=6,
        )


def _sphere(ax, inclusion: InclusionConfig, z: float) -> None:
    x, y, zc = inclusion.center
    if abs(z - zc) >= inclusion.radius:
        return
    section = np.sqrt(inclusion.radius**2 - (z - zc) ** 2)
    t = np.linspace(0, TWO_PI, 200)
    ax.plot(
        x + section * np.cos(t),
        y + section * np.sin(t),
        color=INK,
        linewidth=0.7,
        linestyle=(0, (3, 2)),
    )


def _panel(fig, ax, X, Y, V, cmap, limit, title, label, radius, contours=None):
    image = ax.pcolormesh(
        X,
        Y,
        V,
        cmap=cmap,
        norm=TwoSlopeNorm(0, -limit, limit),
        shading="auto",
        rasterized=True,
    )
    if contours is not None:
        ax.contour(X, Y, V, levels=contours, colors="white", linewidths=0.4, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3 * radius, 1.3 * radius)
    ax.set_ylim(-1.3 * radius, 1.3 * radius)
    ax.axis("off")
    ax.set_title(title)
    bar = fig.colorbar(
        image, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=30
    )
    bar.outline.set_edgecolor(FRAME)
    bar.ax.tick_params(size=0, labelsize=7)
    bar.set_label(label, fontsize=8)
    bar.formatter.set_powerlimits((-2, 2))
    bar.update_ticks()


def render_potential(
    potential: dolfinx.fem.Function,
    effect: dolfinx.fem.Function,
    layout: ElectrodeLayout,
    inclusion: InclusionConfig,
    radius: float,
    path: Path,
    z: float = 0.0,
) -> Path:
    X, Y, (u, du) = horizontal_slice([potential, effect], radius, z)
    u_lim = SATURATION * np.nanmax(np.abs(u))
    du_lim = np.nanmax(np.abs(du))
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(1.6 * COLUMN, 3.4))
        _panel(
            fig,
            left,
            X,
            Y,
            u,
            "RdBu_r",
            u_lim,
            rf"potencial $u$ em $z = {z:g}$",
            r"$u$",
            radius,
            contours=np.linspace(-u_lim, u_lim, CONTOURS),
        )
        _panel(
            fig,
            right,
            X,
            Y,
            du,
            "PuOr_r",
            du_lim,
            r"efeito da inclusão, $u(\gamma) - u(1)$",
            r"$\Delta u$",
            radius,
        )
        for ax in (left, right):
            _electrodes(ax, layout, radius, z)
            _sphere(ax, inclusion, z)
        return save(fig, path)
