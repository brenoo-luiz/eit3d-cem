import numpy as np
import pytest

from cem3d.config import ElectrodeConfig
from cem3d.geometry.electrodes import TWO_PI, ElectrodeLayout


def layout(**kwargs) -> ElectrodeLayout:
    return ElectrodeLayout(ElectrodeConfig(**kwargs))


def test_sizes_follow_coverage():
    lay = layout(count=16, coverage=0.5)
    assert len(lay) == 16
    for e in lay:
        assert e.width == pytest.approx(TWO_PI / 16 * 0.5)
        assert e.height == pytest.approx(1.0)
        assert e.area(1.0) == pytest.approx(np.pi / 16)


def test_electrodes_are_centered_in_height():
    for e in layout(height=1.0):
        assert (e.z_min, e.z_max) == pytest.approx((-0.5, 0.5))


def test_first_electrode_starts_at_rotation():
    assert layout(rotation=1.0)[1].theta_start == pytest.approx(1.0)


def test_clockwise_numbering():
    lay = layout(count=16, clockwise=True)
    assert lay[2].theta_start == pytest.approx(lay[1].theta_start - TWO_PI / 16)


def test_anticlockwise_numbering():
    lay = layout(count=16, clockwise=False)
    assert lay[2].theta_start == pytest.approx(lay[1].theta_start + TWO_PI / 16)


@pytest.mark.parametrize("clockwise", [True, False])
def test_locate_finds_each_electrode(clockwise):
    lay = layout(rotation=1.0, clockwise=clockwise)
    for e in lay:
        assert lay.locate(e.theta_center, 0.0) == e.index
        assert lay.locate(e.theta_center + TWO_PI, 0.0) == e.index


def test_locate_gap_and_outside_height():
    lay = layout(rotation=0.0, coverage=0.5)
    e = lay[1]
    assert lay.locate(e.theta_end + 0.01, 0.0) == 0
    assert lay.locate(e.theta_center, 0.6) == 0


def test_electrodes_do_not_overlap():
    lay = layout(count=16, coverage=0.9)
    starts = sorted(e.theta_start % TWO_PI for e in lay)
    gaps = np.diff(starts + [starts[0] + TWO_PI])
    assert np.all(gaps > lay[1].width)


@pytest.mark.parametrize("clockwise", [True, False])
@pytest.mark.parametrize("rotation", [0.0, 1.0, -2.5])
def test_gap_center_is_off_every_electrode(clockwise, rotation):
    lay = layout(rotation=rotation, clockwise=clockwise)
    assert lay.locate(lay.gap_center, 0.0) == 0
    gap = lay.sector - lay[1].width
    for e in lay:
        start = (lay.gap_center - e.theta_start) % TWO_PI
        end = (e.theta_end - lay.gap_center) % TWO_PI
        assert min(start - e.width, end) >= gap / 2 - 1e-12
