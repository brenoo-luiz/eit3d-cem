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

MARGINS = {"top_margin": 0.3, "bottom_margin": 0.3}
TWO_ROWS = (RowConfig(), RowConfig())


def test_default_setup_follows_the_advisor_sheet():
    cfg = SetupConfig()
    assert (cfg.mesh.radius, cfg.mesh.height) == (1.0, 2.0)
    (row,) = cfg.electrodes.rows
    assert (row.count, row.clockwise) == (16, True)
    assert cfg.electrodes.coverage == 0.5
    assert math.radians(row.theta) == pytest.approx(1.0)
    assert cfg.electrodes.height(1.0) == pytest.approx(1.0)


def test_cylinder_height_follows_the_aspect_ratio():
    assert MeshConfig(radius=1.5, aspect=3.0).height == pytest.approx(4.5)


def test_width_is_set_by_the_first_row():
    rows = (RowConfig(count=8), RowConfig(count=4))
    config = ElectrodeConfig(coverage=0.4, rows=rows, **MARGINS)
    assert config.width(2.0) == pytest.approx(0.4 * 2 * math.pi * 2.0 / 8)


def test_electrode_height_follows_the_aspect_ratio():
    rows = (RowConfig(count=16), RowConfig(count=8))
    config = ElectrodeConfig(aspect=2.0, rows=rows, **MARGINS)
    assert config.height(1.0) == pytest.approx(2 * math.pi / 16)


def test_count_adds_every_row():
    rows = (RowConfig(count=16), RowConfig(count=8), RowConfig(count=12))
    assert ElectrodeConfig(aspect=1.0, rows=rows, **MARGINS).count == 36


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
    two = ElectrodeConfig(rows=TWO_ROWS, top_margin=1, bottom_margin=1)
    assert (two.top_margin, two.bottom_margin) == (1.0, 1.0)
    assert isinstance(two.top_margin, float)


def test_margins_of_several_rows():
    two = ElectrodeConfig(rows=TWO_ROWS, top_margin=1, bottom_margin=0.5)
    assert two.margins() == (1.0, 0.5)


def test_single_row_has_no_margins():
    with pytest.raises(ValueError):
        ElectrodeConfig().margins()


def test_invalid_row_config():
    with pytest.raises(ValueError):
        RowConfig(count=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"aspect": 0.0},
        {"rows": ()},
        {"coverage": 0.0},
        {"coverage": 1.0},
        {"top_margin": 0.3},
        {"rows": TWO_ROWS},
        {"rows": TWO_ROWS, "top_margin": 0.3},
        {"rows": TWO_ROWS, "top_margin": 0.0, "bottom_margin": 0.3},
        {"rows": (RowConfig(count=8), RowConfig(count=16)), **MARGINS},
    ],
    ids=[
        "aspect",
        "no rows",
        "coverage 0",
        "coverage 1",
        "margin on one row",
        "no margins",
        "one margin",
        "zero margin",
        "row does not fit around",
    ],
)
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
    two = ElectrodeConfig(rows=TWO_ROWS, top_margin=0.1, bottom_margin=0.1)
    with pytest.raises(ValueError):
        SetupConfig(electrodes=two)


def test_margins_must_leave_room_between_the_rows():
    def setup(margin: float) -> SetupConfig:
        return SetupConfig(
            electrodes=ElectrodeConfig(
                aspect=1.0, rows=TWO_ROWS, top_margin=margin, bottom_margin=margin
            )
        )

    assert setup(0.7).electrodes.top_margin == 0.7
    with pytest.raises(ValueError):
        setup(0.9)


@pytest.mark.parametrize("kwargs", [{"impedance": 0}, {"degree": 0}, {"rtol": 0}])
def test_invalid_cem_config(kwargs):
    with pytest.raises(ValueError):
        CEMConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"radius": 0}, {"inside": 0}, {"outside": -1}])
def test_invalid_inclusion(kwargs):
    with pytest.raises(ValueError):
        InclusionConfig(**kwargs)
