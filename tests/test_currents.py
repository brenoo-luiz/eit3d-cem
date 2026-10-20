import numpy as np
import pytest

from cem3d.forward.currents import adjacent_pattern


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
