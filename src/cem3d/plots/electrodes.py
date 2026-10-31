from __future__ import annotations

from pathlib import Path

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from cem3d.config import GAP_TAG, electrode_tag
from cem3d.geometry.electrodes import TWO_PI, ElectrodeLayout
from cem3d.geometry.mesh import MeshData
from cem3d.plots.paper import COLUMN, FRAME, INK, PALETTE, STYLE, frame, save

ELECTRODE_FILL = "#cfe0f1"
EDGE = "#9a9a9a"
RING_STEP = 0.16
SHIFTS = (-360.0, 0.0, 360.0)


def _facet_triangles(data: MeshData, tag: int) -> np.ndarray:
    facets = data.facet_tags.find(tag)
    nodes = dolfinx.mesh.entities_to_geometry(data.mesh, 2, facets)
    return data.mesh.geometry.x[nodes]


def _unroll(triangles: np.ndarray) -> np.ndarray:
    theta = np.arctan2(triangles[..., 1], triangles[..., 0]) % TWO_PI
    ref = theta[:, :1]
    theta = np.where(theta - ref > np.pi, theta - TWO_PI, theta)
    theta = np.where(ref - theta > np.pi, theta + TWO_PI, theta)
    return np.stack([np.degrees(theta), triangles[..., 2]], axis=-1)


def _shifted(triangles: np.ndarray) -> np.ndarray:
    return np.concatenate([triangles + [shift, 0.0] for shift in SHIFTS])


def render_layout(data: MeshData, radius: float, height: float, path: Path) -> Path:
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(2 * COLUMN, 2.5))
        top = fig.add_axes((0.0, 0.0, 0.26, 1.0))
        side = fig.add_axes((0.33, 0.12, 0.67, 0.78))
        _top_view(top, data.layout, radius)
        _side_view(side, data, height)
        return save(fig, path)


def _arrow(ax, start: float, sense: int, r: float) -> None:
    arc = np.linspace(start, start + sense * 0.9, 30)
    ax.plot(r * np.cos(arc), r * np.sin(arc), color=INK, linewidth=0.7)
    ax.annotate(
        "",
        xy=(r * np.cos(arc[-1]), r * np.sin(arc[-1])),
        xytext=(r * np.cos(arc[-3]), r * np.sin(arc[-3])),
        arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 0.7},
    )


def _top_view(ax, layout: ElectrodeLayout, radius: float) -> None:
    t = np.linspace(0, TWO_PI, 361)
    single = len(layout.rows) == 1
    ax.plot(
        [0, 1.35 * radius], [0, 0], color=FRAME, linewidth=0.6, linestyle=(0, (3, 2))
    )
    for row in layout.rows:
        r = radius * (1 - RING_STEP * (row.number - 1))
        color = PALETTE[(row.number - 1) % len(PALETTE)]
        ax.plot(r * np.cos(t), r * np.sin(t), color=FRAME, linewidth=0.8)
        electrodes = list(row)
        for e in electrodes:
            arc = np.linspace(e.theta_start, e.theta_end, 30)
            ax.plot(
                r * np.cos(arc),
                r * np.sin(arc),
                color=color,
                linewidth=3 if single else 2,
                solid_capstyle="butt",
            )
        for e in electrodes if single else electrodes[:1]:
            c = e.theta_center
            ax.text(
                1.2 * radius * np.cos(c),
                1.2 * radius * np.sin(c),
                f"$e_{{{e.index}}}$",
                ha="center",
                va="center",
                fontsize=7,
                color=INK if single else color,
            )
        start = electrodes[0].theta_center + row.sense * 0.25
        _arrow(ax, start, row.sense, 0.7 * radius if single else r - 0.07 * radius)
    ax.set_aspect("equal")
    ax.set_xlim(-1.4 * radius, 1.4 * radius)
    ax.set_ylim(-1.4 * radius, 1.4 * radius)
    ax.axis("off")


def _side_view(ax, data: MeshData, height: float) -> None:
    gap = _shifted(_unroll(_facet_triangles(data, GAP_TAG)))
    ax.add_collection(
        PolyCollection(gap, facecolor="white", edgecolor=EDGE, linewidth=0.15)
    )
    for e in data.layout:
        tri = _shifted(_unroll(_facet_triangles(data, electrode_tag(e.index))))
        ax.add_collection(
            PolyCollection(
                tri, facecolor=ELECTRODE_FILL, edgecolor=PALETTE[0], linewidth=0.15
            )
        )
        ax.text(
            np.degrees(e.theta_center) % 360,
            e.z_max + 0.03 * height,
            f"$e_{{{e.index}}}$",
            ha="center",
            va="bottom",
            fontsize=7 if len(data.layout.rows) == 1 else 5,
        )
    ax.set_xlim(0, 360)
    ax.set_ylim(-height / 2, height / 2)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel(r"$\theta$ (graus, a partir da referência $\theta = 0$)")
    ax.set_ylabel(r"$z$")
    ax.grid(False)
    frame(ax)
