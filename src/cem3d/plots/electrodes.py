from __future__ import annotations

from pathlib import Path

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from cem3d.config import GAP_TAG, electrode_tag
from cem3d.geometry.electrodes import TWO_PI
from cem3d.geometry.mesh import MeshData
from cem3d.plots.paper import COLUMN, FRAME, INK, PALETTE, STYLE, frame, save

ELECTRODE_FILL = "#cfe0f1"
EDGE = "#9a9a9a"


def _facet_triangles(data: MeshData, tag: int) -> np.ndarray:
    facets = data.facet_tags.find(tag)
    nodes = dolfinx.mesh.entities_to_geometry(data.mesh, 2, facets)
    return data.mesh.geometry.x[nodes]


def _unroll(triangles: np.ndarray, cut: float) -> np.ndarray:
    """Maps lateral triangles to (theta, z), with theta measured from the cut angle."""
    theta = (np.arctan2(triangles[..., 1], triangles[..., 0]) - cut) % TWO_PI
    # Keeps the three vertices of a triangle on the same side of the cut.
    ref = theta[:, :1]
    theta = np.where(theta - ref > np.pi, theta - TWO_PI, theta)
    theta = np.where(ref - theta > np.pi, theta + TWO_PI, theta)
    return np.stack([np.degrees(theta), triangles[..., 2]], axis=-1)


def render_layout(data: MeshData, radius: float, height: float, path: Path) -> Path:
    layout = data.layout
    cfg = layout.config
    cut = cfg.rotation + layout.sector * (1 + cfg.coverage) / 2
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(2 * COLUMN, 2.5))
        top = fig.add_axes([0.0, 0.0, 0.26, 1.0])
        side = fig.add_axes([0.33, 0.12, 0.67, 0.78])
        _top_view(top, layout, radius)
        _side_view(side, data, cut, height)
        return save(fig, path)


def _top_view(ax, layout, radius: float) -> None:
    t = np.linspace(0, TWO_PI, 361)
    ax.plot(radius * np.cos(t), radius * np.sin(t), color=FRAME, linewidth=0.8)
    for e in layout:
        arc = np.linspace(e.theta_start, e.theta_end, 30)
        ax.plot(radius * np.cos(arc), radius * np.sin(arc), color=PALETTE[0], linewidth=3, solid_capstyle="butt")
        c = e.theta_center
        ax.text(1.2 * radius * np.cos(c), 1.2 * radius * np.sin(c), f"$e_{{{e.index}}}$",
                ha="center", va="center", fontsize=7)
    sense = -1 if layout.config.clockwise else 1
    start = layout[1].theta_center + sense * 0.25
    arc = np.linspace(start, start + sense * 0.9, 30)
    ax.plot(0.7 * radius * np.cos(arc), 0.7 * radius * np.sin(arc), color=INK, linewidth=0.7)
    ax.annotate("", xy=(0.7 * radius * np.cos(arc[-1]), 0.7 * radius * np.sin(arc[-1])),
                xytext=(0.7 * radius * np.cos(arc[-3]), 0.7 * radius * np.sin(arc[-3])),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=0.7))
    ax.set_aspect("equal")
    ax.set_xlim(-1.4 * radius, 1.4 * radius)
    ax.set_ylim(-1.4 * radius, 1.4 * radius)
    ax.axis("off")


def _side_view(ax, data: MeshData, cut: float, height: float) -> None:
    gap = _unroll(_facet_triangles(data, GAP_TAG), cut)
    ax.add_collection(PolyCollection(gap, facecolor="white", edgecolor=EDGE, linewidth=0.15))
    for e in data.layout:
        tri = _unroll(_facet_triangles(data, electrode_tag(e.index)), cut)
        ax.add_collection(PolyCollection(tri, facecolor=ELECTRODE_FILL, edgecolor=PALETTE[0], linewidth=0.15))
        x = tri[..., 0].mean()
        ax.text(x, e.z_max + 0.12, f"$e_{{{e.index}}}$", ha="center", va="bottom", fontsize=7)
    ax.set_xlim(0, 360)
    ax.set_ylim(-height / 2, height / 2)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel(r"$\theta - \theta_0$ (graus)")
    ax.set_ylabel(r"$z$")
    ax.grid(False)
    frame(ax)
