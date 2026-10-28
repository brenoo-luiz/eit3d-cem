import dolfinx
import matplotlib.pyplot as plt
import numpy as np

from cem3d.config import InclusionConfig
from cem3d.plots.consistency import render_boundary, render_contact, render_matrix
from cem3d.plots.electrodes import render_layout
from cem3d.plots.forward import render_potential
from cem3d.plots.paper import STYLE, frame, save


def test_save_creates_folder_and_file(tmp_path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        frame(ax)
        path = save(fig, tmp_path / "sub" / "fig.png")
    assert path.exists() and path.stat().st_size > 0


def test_layout_figure(mesh_data, tmp_path):
    path = render_layout(mesh_data, 1.0, 2.0, tmp_path / "layout.png")
    assert path.exists() and path.stat().st_size > 0


def test_potential_figure(mesh_data, tmp_path):
    V = dolfinx.fem.functionspace(mesh_data.mesh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] - x[1])
    du = dolfinx.fem.Function(V)
    du.interpolate(lambda x: 1e-3 * x[0] * x[1])
    path = render_potential(
        u, du, mesh_data.layout, InclusionConfig(), 1.0, tmp_path / "potential.png"
    )
    assert path.exists() and path.stat().st_size > 0


def test_consistency_figures(tmp_path):
    rng = np.random.default_rng(0)
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    electrode = np.sin(8 * theta) > 0
    paths = [
        render_matrix(
            [(rng.normal(size=(16, 16)), "A"), (rng.normal(size=(16, 16)), "B")],
            tmp_path / "matrix.png",
        ),
        render_boundary(
            points,
            [rng.normal(size=200)] * 16,
            electrode,
            0.1,
            "flux",
            tmp_path / "flux.png",
        ),
        render_contact(
            points[:20],
            rng.normal(size=20),
            0.5,
            rng.normal(size=(16, 16)),
            0.1,
            tmp_path / "contact.png",
        ),
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_boundary_figure_uses_one_height_per_panel(tmp_path):
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    ring = np.column_stack([np.cos(theta), np.sin(theta)])
    points = np.vstack(
        [
            np.column_stack([ring, np.full(100, 0.5)]),
            np.column_stack([ring, -0.5 * np.ones(100)]),
        ]
    )
    values = [np.r_[np.ones(100), np.zeros(100)], np.r_[np.zeros(100), np.ones(100)]]
    path = render_boundary(
        points,
        values,
        np.zeros(200, dtype=bool),
        0.1,
        "flux",
        tmp_path / "rows.png",
        levels=[0.5, -0.5],
    )
    assert path.exists() and path.stat().st_size > 0
