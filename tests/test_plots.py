import matplotlib.pyplot as plt
import numpy as np

from cem3d.plots.electrodes import render_layout
from cem3d.plots.forward import render_difference, render_voltages
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


def test_forward_figures(tmp_path):
    rng = np.random.default_rng(0)
    u = rng.normal(size=16)
    paths = [
        render_voltages(u, u + 1e-3, 1, tmp_path / "voltages.png"),
        render_difference(rng.normal(size=(16, 16)), tmp_path / "difference.png"),
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
