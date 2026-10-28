import math
from itertools import pairwise

import numpy as np
import pytest

from cem3d.config import ElectrodeConfig, MeshConfig, RowConfig, SetupConfig
from cem3d.geometry.electrodes import TWO_PI, ElectrodeLayout


def layout(*rows: RowConfig, aspect: float = 1.0, height: float = 2.0):
    electrodes = (
        ElectrodeConfig(aspect=aspect, rows=rows) if rows else ElectrodeConfig()
    )
    return ElectrodeLayout(
        SetupConfig(mesh=MeshConfig(aspect=height), electrodes=electrodes)
    )


def test_default_sizes_follow_coverage():
    lay = layout()
    assert len(lay) == 16
    for e in lay:
        assert e.width == pytest.approx(TWO_PI / 16 * 0.5)
        assert e.height == pytest.approx(1.0)
        assert e.area(1.0) == pytest.approx(np.pi / 16)


def test_single_row_is_centered_in_height():
    for e in layout():
        assert (e.z_min, e.z_max) == pytest.approx((-0.5, 0.5))


def test_anticlockwise_row_starts_at_theta():
    lay = layout(RowConfig(theta=30.0, clockwise=False))
    assert lay[1].theta_start == pytest.approx(math.radians(30.0))
    assert lay[2].theta_start == pytest.approx(lay[1].theta_start + TWO_PI / 16)


def test_clockwise_row_starts_at_theta_turning_clockwise():
    lay = layout(RowConfig(theta=30.0, clockwise=True))
    assert lay[1].theta_end == pytest.approx(-math.radians(30.0))
    assert lay[2].theta_end == pytest.approx(lay[1].theta_end - TWO_PI / 16)


def test_clockwise_row_mirrors_the_anticlockwise_row():
    cw = layout(RowConfig(theta=40.0, clockwise=True))
    acw = layout(RowConfig(theta=40.0, clockwise=False))
    for a, b in zip(cw, acw, strict=True):
        assert (a.theta_start, a.theta_end) == pytest.approx(
            (-b.theta_end, -b.theta_start)
        )


def test_old_layout_is_the_clockwise_row_at_minus_one_rad_minus_width():
    width = TWO_PI / 16 * 0.5
    lay = layout(RowConfig(theta=-math.degrees(1 + width)), aspect=16 / np.pi)
    for j, e in enumerate(lay):
        start = 1 - j * TWO_PI / 16
        assert (e.theta_start, e.theta_end) == pytest.approx((start, start + width))


def test_height_follows_the_aspect_ratio_of_each_row():
    lay = layout(RowConfig(count=8), RowConfig(count=16), aspect=1.5)
    for e in lay:
        assert e.height == pytest.approx(1.5 * e.width)
    assert lay[1].height == pytest.approx(2 * lay[9].height)


def test_rows_are_numbered_from_top_to_bottom():
    lay = layout(RowConfig(count=8), RowConfig(count=12), RowConfig(count=16))
    assert [len(row) for row in lay.rows] == [8, 12, 16]
    assert [e.index for e in lay] == list(range(1, 37))
    assert [next(iter(row)).index for row in lay.rows] == [1, 9, 21]
    centers = [row.z_center for row in lay.rows]
    assert centers == sorted(centers, reverse=True)


def test_vertical_gaps_are_all_equal():
    lay = layout(RowConfig(count=8), RowConfig(count=16), RowConfig(count=12))
    rows = lay.rows
    gaps = [1.0 - rows[0].z_max]
    gaps += [upper.z_min - lower.z_max for upper, lower in pairwise(rows)]
    gaps.append(rows[-1].z_min + 1.0)
    assert gaps == pytest.approx([lay.vertical_gap] * 4)
    assert lay.vertical_gap > 0


@pytest.mark.parametrize("clockwise", [True, False])
def test_locate_finds_each_electrode(clockwise):
    lay = layout(
        RowConfig(theta=57.0, clockwise=clockwise),
        RowConfig(count=12, theta=10.0, clockwise=not clockwise),
    )
    for e in lay:
        assert lay.locate(e.theta_center, e.z_center) == e.index
        assert lay.locate(e.theta_center + TWO_PI, e.z_center) == e.index


def test_locate_gap_and_outside_height():
    lay = layout(RowConfig(theta=0.0, clockwise=False))
    e = lay[1]
    assert lay.locate(e.theta_end + 0.01, e.z_center) == 0
    assert lay.locate(e.theta_center, e.z_max + 0.01) == 0


def test_locate_on_the_leading_edge_stays_in_the_row():
    lay = layout(
        RowConfig(theta=0.0, clockwise=False), RowConfig(theta=0.0, clockwise=False)
    )
    assert lay.rows[0].locate(-1e-18) == 1


def test_electrodes_do_not_overlap():
    lay = layout(RowConfig(coverage=0.9))
    starts = sorted(e.theta_start % TWO_PI for e in lay)
    gaps = np.diff(starts + [starts[0] + TWO_PI])
    assert np.all(gaps > lay[1].width)


@pytest.mark.parametrize("clockwise", [True, False])
@pytest.mark.parametrize("theta", [0.0, 57.3, -143.0])
def test_seam_is_in_the_middle_of_a_gap(clockwise, theta):
    lay = layout(RowConfig(theta=theta, clockwise=clockwise))
    row = lay.rows[0]
    assert lay.locate(lay.seam, 0.0) == 0
    assert lay.clearance(lay.seam) == pytest.approx((row.sector - row.width) / 2)


def test_seam_avoids_every_row_when_they_share_a_gap():
    lay = layout(RowConfig(count=16, theta=0.0), RowConfig(count=8, theta=0.0))
    assert lay.clearance(lay.seam) > 0
    for row in lay.rows:
        assert row.locate(lay.seam) == 0


def test_staggered_rows_share_no_gap():
    lay = layout(RowConfig(theta=0.0), RowConfig(theta=11.25))
    assert lay.clearance(lay.seam) == 0.0
