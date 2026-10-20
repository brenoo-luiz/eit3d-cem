import numpy as np
import pytest

from cem3d.config import BOTTOM_TAG, GAP_TAG, TOP_TAG, electrode_tag
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
        assert abs(c[2]) < 1e-6


def test_boundary_is_fully_tagged(mesh_data):
    tags = [TOP_TAG, BOTTOM_TAG, GAP_TAG] + [
        electrode_tag(e.index) for e in mesh_data.layout
    ]
    total = sum(facet_area(mesh_data, t) for t in tags)
    assert total == pytest.approx(TWO_PI * 2 + 2 * np.pi, rel=1e-2)


def test_electrode_crossing_seam_is_found(mesh_data):
    e4 = mesh_data.layout[4]
    assert e4.theta_start % TWO_PI > e4.theta_end % TWO_PI
    assert facet_area(mesh_data, electrode_tag(4)) == pytest.approx(
        e4.area(1.0), rel=1e-2
    )


def test_mesh_is_cached(mesh_config, tmp_path):
    first = CylinderWithElectrodes(mesh_config, cache_dir=tmp_path)
    first.get()
    assert first.file.exists()
    again = CylinderWithElectrodes(mesh_config, cache_dir=tmp_path).get()
    assert facet_area(again, electrode_tag(1)) == pytest.approx(
        again.layout[1].area(1.0), rel=1e-2
    )
