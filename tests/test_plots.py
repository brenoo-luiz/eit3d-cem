import matplotlib.pyplot as plt

from cem3d.plots.electrodes import render_layout
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
