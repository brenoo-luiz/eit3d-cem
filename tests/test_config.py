import pytest

from cem3d.config import (
    ELECTRODE_TAG0,
    ElectrodeConfig,
    MeshConfig,
    SetupConfig,
    electrode_tag,
)


def test_default_setup_follows_the_advisor_sheet():
    cfg = SetupConfig()
    assert (cfg.mesh.radius, cfg.mesh.height) == (1.0, 2.0)
    e = cfg.electrodes
    assert (e.count, e.coverage, e.height, e.clockwise) == (16, 0.5, 1.0, True)


def test_electrode_tags_are_unique_and_offset():
    tags = [electrode_tag(j) for j in range(1, 17)]
    assert tags[0] == ELECTRODE_TAG0 + 1
    assert len(set(tags)) == 16


def test_values_are_stored_as_float():
    assert MeshConfig(radius=1).radius == 1.0
    assert ElectrodeConfig(rotation=1).rotation == 1.0


@pytest.mark.parametrize(
    "kwargs", [{"count": 1}, {"coverage": 0.0}, {"coverage": 1.0}, {"height": 0.0}]
)
def test_invalid_electrode_config(kwargs):
    with pytest.raises(ValueError):
        ElectrodeConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"radius": 0}, {"height": -1}, {"size_min": 0.2, "size_max": 0.1}]
)
def test_invalid_mesh_config(kwargs):
    with pytest.raises(ValueError):
        MeshConfig(**kwargs)


def test_electrodes_must_fit_in_cylinder():
    with pytest.raises(ValueError):
        SetupConfig(mesh=MeshConfig(height=1.0), electrodes=ElectrodeConfig(height=1.0))
