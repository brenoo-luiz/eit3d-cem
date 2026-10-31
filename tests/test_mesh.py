from types import SimpleNamespace

import numpy as np
import pytest
from dolfinx.io import gmsh as gmshio

from cem3d.config import (
    BOTTOM_TAG,
    GAP_TAG,
    TOP_TAG,
    ElectrodeConfig,
    MeshConfig,
    RowConfig,
    SetupConfig,
    electrode_tag,
)
from cem3d.geometry.electrodes import TWO_PI
from cem3d.geometry.mesh import CylinderWithElectrodes, facet_area, facet_centroid


def test_every_electrode_is_tagged(mesh_data):
    for e in mesh_data.layout:
        assert mesh_data.facet_tags.find(electrode_tag(e.index)).size > 0


def test_electrode_areas(mesh_data):
    for e in mesh_data.layout:
        assert facet_area(mesh_data, electrode_tag(e.index)) == pytest.approx(
            e.area(1.0), rel=1e-2
        )


def test_electrode_positions(mesh_data):
    for e in mesh_data.layout:
        c = facet_centroid(mesh_data, electrode_tag(e.index))
        diff = (np.arctan2(c[1], c[0]) - e.theta_center + np.pi) % TWO_PI - np.pi
        assert abs(diff) < 1e-2
        assert abs(c[2] - e.z_center) < 1e-6


def test_boundary_is_fully_tagged(mesh_data):
    tags = [TOP_TAG, BOTTOM_TAG, GAP_TAG] + [
        electrode_tag(e.index) for e in mesh_data.layout
    ]
    total = sum(facet_area(mesh_data, t) for t in tags)
    assert total == pytest.approx(TWO_PI * 2 + 2 * np.pi, rel=1e-2)


def test_electrode_crossing_zero_angle_is_found(mesh_data):
    (crossing,) = [
        e for e in mesh_data.layout if e.theta_start % TWO_PI > e.theta_end % TWO_PI
    ]
    assert facet_area(mesh_data, electrode_tag(crossing.index)) == pytest.approx(
        crossing.area(1.0), rel=1e-2
    )


def test_electrode_meshes_are_alike(mesh_data):
    counts = [
        mesh_data.facet_tags.find(electrode_tag(e.index)).size for e in mesh_data.layout
    ]
    assert max(counts) / min(counts) < 1.1


def test_mesh_is_cached(mesh_config, tmp_path):
    first = CylinderWithElectrodes(mesh_config, cache_dir=tmp_path)
    first.get()
    assert first.file.exists()
    again = CylinderWithElectrodes(mesh_config, cache_dir=tmp_path).get()
    assert facet_area(again, electrode_tag(1)) == pytest.approx(
        again.layout[1].area(1.0), rel=1e-2
    )


def test_mesh_without_facet_tags_is_rejected(mesh_config, tmp_path, monkeypatch):
    untagged = SimpleNamespace(mesh=None, facet_tags=None)
    monkeypatch.setattr(gmshio, "model_to_mesh", lambda *args, **kwargs: untagged)
    with pytest.raises(RuntimeError):
        CylinderWithElectrodes(mesh_config, cache_dir=tmp_path).get()


@pytest.fixture(scope="module")
def staggered(tmp_path_factory):
    rows = (RowConfig(theta=0.0), RowConfig(theta=11.25))
    config = SetupConfig(
        mesh=MeshConfig(),
        electrodes=ElectrodeConfig(
            aspect=2.0, rows=rows, top_margin=0.4, bottom_margin=0.4
        ),
    )
    return CylinderWithElectrodes(
        config, cache_dir=tmp_path_factory.mktemp("cache")
    ).get()


def test_staggered_rows_are_tagged_with_exact_areas(staggered):
    assert staggered.layout.clearance(staggered.layout.seam) == 0.0
    for e in staggered.layout:
        assert facet_area(staggered, electrode_tag(e.index)) == pytest.approx(
            e.area(1.0), rel=1e-3
        )


def test_staggered_rows_are_in_place(staggered):
    for e in staggered.layout:
        c = facet_centroid(staggered, electrode_tag(e.index))
        diff = (np.arctan2(c[1], c[0]) - e.theta_center + np.pi) % TWO_PI - np.pi
        assert abs(diff) < 1e-2
        assert abs(c[2] - e.z_center) < 1e-6


def test_electrode_cut_by_the_seam_is_meshed_like_the_others(staggered):
    counts = [
        staggered.facet_tags.find(electrode_tag(e.index)).size for e in staggered.layout
    ]
    assert max(counts) / min(counts) < 1.1
