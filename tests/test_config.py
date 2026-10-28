import math

import pytest

from cem3d.config import (
    ELECTRODE_TAG0,
    CEMConfig,
    ElectrodeConfig,
    InclusionConfig,
    MeshConfig,
    RowConfig,
    SetupConfig,
    electrode_tag,
)


def test_default_setup_follows_the_advisor_sheet():
    cfg = SetupConfig()
    assert (cfg.mesh.radius, cfg.mesh.height) == (1.0, 2.0)
    (row,) = cfg.electrodes.rows
    assert (row.count, row.coverage, row.clockwise) == (16, 0.5, True)
    assert math.radians(row.theta) == pytest.approx(1.0)
    assert cfg.electrodes.heights(1.0) == pytest.approx((1.0,))


def test_cylinder_height_follows_the_aspect_ratio():
    assert MeshConfig(radius=1.5, aspect=3.0).height == pytest.approx(4.5)


def test_row_width_is_the_arc_length_of_one_electrode():
    assert RowConfig(count=8, coverage=0.4).width(2.0) == pytest.approx(
        0.4 * 2 * math.pi * 2.0 / 8
    )


def test_electrode_height_follows_the_aspect_ratio():
    rows = (RowConfig(count=16, coverage=0.5), RowConfig(count=8, coverage=0.5))
    heights = ElectrodeConfig(aspect=2.0, rows=rows).heights(1.0)
    assert heights == pytest.approx((2 * math.pi / 16, 2 * math.pi / 8))


def test_count_adds_every_row():
    rows = (RowConfig(count=16), RowConfig(count=8), RowConfig(count=12))
    assert ElectrodeConfig(aspect=1.0, rows=rows).count == 36


def test_rows_given_as_a_list_are_stored_as_a_tuple():
    assert isinstance(ElectrodeConfig(rows=[RowConfig()]).rows, tuple)


def test_electrode_tags_are_unique_and_offset():
    tags = [electrode_tag(j) for j in range(1, 17)]
    assert tags[0] == ELECTRODE_TAG0 + 1
    assert len(set(tags)) == 16


def test_cem_defaults_follow_the_advisor_sheet():
    assert CEMConfig().impedance == 1e-3


def test_values_are_stored_as_float():
    assert MeshConfig(radius=1).radius == 1.0
    assert RowConfig(theta=30).theta == 30.0
    assert ElectrodeConfig(aspect=2).aspect == 2.0


@pytest.mark.parametrize("kwargs", [{"count": 1}, {"coverage": 0.0}, {"coverage": 1.0}])
def test_invalid_row_config(kwargs):
    with pytest.raises(ValueError):
        RowConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"aspect": 0.0}, {"rows": ()}])
def test_invalid_electrode_config(kwargs):
    with pytest.raises(ValueError):
        ElectrodeConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"radius": 0}, {"aspect": -1}, {"size_min": 0.2, "size_max": 0.1}]
)
def test_invalid_mesh_config(kwargs):
    with pytest.raises(ValueError):
        MeshConfig(**kwargs)


def test_electrodes_must_fit_in_cylinder():
    with pytest.raises(ValueError):
        SetupConfig(mesh=MeshConfig(aspect=1.0))


def test_rows_together_must_fit_in_cylinder():
    two = ElectrodeConfig(rows=(RowConfig(), RowConfig()))
    with pytest.raises(ValueError):
        SetupConfig(electrodes=two)


@pytest.mark.parametrize("kwargs", [{"impedance": 0}, {"degree": 0}, {"rtol": 0}])
def test_invalid_cem_config(kwargs):
    with pytest.raises(ValueError):
        CEMConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"radius": 0}, {"inside": 0}, {"outside": -1}])
def test_invalid_inclusion(kwargs):
    with pytest.raises(ValueError):
        InclusionConfig(**kwargs)
