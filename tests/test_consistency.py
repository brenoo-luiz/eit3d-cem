import numpy as np
import pytest

from cem3d.config import InclusionConfig
from cem3d.fields import sphere_conductivity
from cem3d.forward.cem import CEMForward
from cem3d.forward.consistency import ConsistencyTests
from cem3d.forward.currents import adjacent_pattern


@pytest.fixture(scope="module")
def case(mesh_data):
    gamma = sphere_conductivity(mesh_data.mesh, InclusionConfig())
    forward = CEMForward(mesh_data, gamma)
    solution = forward.solve(adjacent_pattern(16))
    return ConsistencyTests(forward, mesh_data, gamma), solution


def test_divergence_vanishes(case):
    tests, solution = case
    assert tests.divergence(solution).max() < 1e-8


def test_consistent_currents_match_the_injected(case):
    tests, solution = case
    assert np.allclose(tests.currents(solution, True), solution.currents, atol=1e-8)


def test_gradient_currents_have_the_book_error(case):
    tests, solution = case
    error = np.abs(tests.currents(solution, False) - solution.currents).max()
    assert 1e-2 < error < 1.0


def test_no_current_through_the_gaps(case):
    tests, solution = case
    assert tests.gap_flux(solution, True) < 1e-8


def test_contact_voltages_match_U(case):
    tests, solution = case
    difference = tests.contact_voltages(solution, True) - solution.voltages
    assert np.abs(difference).max() < 2e-2 * np.abs(solution.voltages).max()


def test_contact_profile_lies_on_the_electrode(case, mesh_data):
    tests, solution = case
    points, values = tests.contact_profile(solution.voltages[0], 1, True)
    assert len(points) == len(values) > 0
    theta = np.arctan2(points[:, 1], points[:, 0])
    e1 = mesh_data.layout[1]
    offset = (theta - e1.theta_start + 1e-6) % (2 * np.pi)
    inside = offset <= e1.width + 2e-6
    assert np.all(inside)


def test_electrode_nodes_are_marked(case):
    tests, _ = case
    assert 0 < tests.on_electrode.sum() < tests.on_electrode.size
