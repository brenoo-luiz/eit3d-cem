import numpy as np
import pytest

from cem3d.cli import load_setup
from cem3d.config import ROOT_DIR, SetupConfig
from cem3d.forward.currents import (
    PROTOCOLS,
    adjacent_by_row,
    adjacent_pattern,
    driving_electrodes,
    get_protocol,
)
from cem3d.geometry.electrodes import ElectrodeLayout

SETUPS = {
    "single_row": SetupConfig(),
    "two_rows_8": load_setup(ROOT_DIR / "configs" / "two_rows_8.toml"),
    "three_rows": load_setup(ROOT_DIR / "configs" / "three_rows.toml"),
}


@pytest.fixture(params=SETUPS.values(), ids=SETUPS.keys())
def layout(request):
    return ElectrodeLayout(request.param)


@pytest.mark.parametrize("name", PROTOCOLS)
def test_every_current_of_a_protocol_sums_to_zero(name, layout):
    currents = get_protocol(name).currents(layout)
    assert currents.shape == (len(layout), len(layout))
    assert np.allclose(currents.sum(axis=1), 0)


def test_rank_by_row_is_electrodes_minus_rows(layout):
    currents = get_protocol("adjacent_by_row").currents(layout)
    assert np.linalg.matrix_rank(currents) == len(layout) - len(layout.rows)


def test_rank_of_the_chain_is_electrodes_minus_one(layout):
    currents = get_protocol("adjacent_chain").currents(layout)
    assert np.linalg.matrix_rank(currents) == len(layout) - 1


def test_chain_crosses_the_rows_and_closes_the_cycle():
    currents = get_protocol("adjacent_chain").currents(
        ElectrodeLayout(SETUPS["two_rows_8"])
    )
    for pattern, (positive, negative) in ((7, (7, 8)), (15, (15, 0))):
        assert currents[pattern, positive] == 1
        assert currents[pattern, negative] == -1


def test_by_row_never_crosses_the_rows():
    currents = get_protocol("adjacent_by_row").currents(
        ElectrodeLayout(SETUPS["two_rows_8"])
    )
    assert np.all(currents[:8, 8:] == 0) and np.all(currents[8:, :8] == 0)


def test_protocols_coincide_on_a_single_row():
    single = ElectrodeLayout(SETUPS["single_row"])
    assert np.array_equal(
        get_protocol("adjacent_by_row").currents(single),
        get_protocol("adjacent_chain").currents(single),
    )


def test_unknown_protocol_is_refused():
    with pytest.raises(ValueError):
        get_protocol("nothing")


def test_first_pattern_matches_the_advisor_example():
    assert adjacent_pattern(16)[0].tolist() == [1, -1] + [0] * 14


def test_sixteen_cyclic_patterns():
    currents = adjacent_pattern(16)
    assert currents.shape == (16, 16)
    assert currents[15, 15] == 1 and currents[15, 0] == -1


def test_every_pattern_sums_to_zero():
    assert np.allclose(adjacent_pattern(16).sum(axis=1), 0)


@pytest.mark.parametrize("experiments", [0, 17])
def test_invalid_number_of_experiments(experiments):
    with pytest.raises(ValueError):
        adjacent_pattern(16, experiments)


def test_one_row_gives_the_adjacent_pattern():
    assert np.array_equal(adjacent_by_row([16]), adjacent_pattern(16))


def test_patterns_cycle_inside_each_row():
    currents = adjacent_by_row([3, 2])
    assert currents.tolist() == [
        [1, -1, 0, 0, 0],
        [0, 1, -1, 0, 0],
        [-1, 0, 1, 0, 0],
        [0, 0, 0, 1, -1],
        [0, 0, 0, -1, 1],
    ]


def test_driving_electrodes_follow_the_rows():
    currents = adjacent_by_row([3, 4])
    assert driving_electrodes(currents) == list(range(1, 8))
