import dataclasses
from typing import NamedTuple

import dolfinx
import numpy as np
import pytest
import ufl

from cem3d.cli import load_setup
from cem3d.config import ROOT_DIR, CEMConfig, DirectionConfig, InclusionConfig
from cem3d.fields import (
    cell_indicator,
    midpoint_direction,
    sphere_conductivity,
    sphere_direction,
)
from cem3d.forward.cem import CEMForward
from cem3d.forward.currents import adjacent_by_row, adjacent_pattern
from cem3d.forward.derivative import CEMDerivative
from cem3d.geometry.mesh import CylinderWithElectrodes, MeshData
from cem3d.steps.derivative import (
    _alignment,
    _alignment_note,
    _summarize_direction,
)

EXPERIMENTS = 3
STEP_IMPEDANCE = 1e-3
TOL_IDENTITY = 1e-9
TOL_HOMOGENEITY = 1e-8
TOL_RANK = 1e-9
FEW_EXPONENTS = np.array([8, 12, 16])


@pytest.fixture(scope="module")
def forward(mesh_data):
    return CEMForward(mesh_data, sphere_conductivity(mesh_data.mesh, InclusionConfig()))


@pytest.fixture(scope="module")
def derivative(forward):
    return CEMDerivative(forward, adjacent_pattern(16, EXPERIMENTS))


@pytest.fixture(scope="module")
def eta(mesh_data):
    return sphere_direction(mesh_data.mesh, DirectionConfig())


@pytest.fixture(scope="module")
def variational(derivative, eta):
    return derivative.directional(eta)


@pytest.fixture(scope="module")
def jacobian(derivative):
    return derivative.jacobian()


def test_derivative_has_one_row_per_current(variational):
    assert variational.shape == (EXPERIMENTS, 16)


def test_derivative_voltages_sum_to_zero(variational):
    assert np.allclose(variational.sum(axis=1), 0, atol=1e-12)


def test_power_of_the_derivative_is_minus_the_weighted_energy(
    forward, derivative, eta, variational
):
    for currents, voltages, W in zip(
        adjacent_pattern(16, EXPERIMENTS), derivative.voltages, variational, strict=True
    ):
        u = forward.potential(voltages)
        energy = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(eta * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx)
        )
        assert currents @ W == pytest.approx(-energy, rel=1e-8)


def test_jacobian_has_one_column_per_cell(jacobian, eta):
    assert jacobian.shape == (EXPERIMENTS * 16, eta.x.array.size)


def test_jacobian_times_direction_matches_the_variational_derivative(
    jacobian, eta, variational
):
    assert np.allclose(
        jacobian @ eta.x.array,
        variational.ravel(),
        rtol=1e-8,
        atol=1e-10 * np.abs(variational).max(),
    )


def test_jacobian_column_is_the_derivative_along_the_cell(derivative, jacobian, eta):
    cells = [0, eta.x.array.size // 2, eta.x.array.size - 1]
    for cell in cells:
        indicator = dolfinx.fem.Function(eta.function_space)
        indicator.x.array[cell] = 1.0
        column = derivative.directional(indicator).ravel()
        assert np.allclose(
            jacobian[:, cell], column, rtol=1e-6, atol=1e-8 * np.abs(column).max()
        )


def test_difference_quotient_converges_at_first_order(derivative, eta, variational):
    errors = [
        np.linalg.norm(derivative.difference_quotient(eta, t) - variational)
        for t in (2.0**-6, 2.0**-7)
    ]
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.05)
    assert errors[1] < 1e-2 * np.linalg.norm(variational)


@pytest.fixture(scope="module")
def two_rows_data(tmp_path_factory):
    config = load_setup(ROOT_DIR / "configs" / "two_rows_8.toml")
    coarse = dataclasses.replace(config.mesh, size_max=0.25, size_min=0.08)
    return CylinderWithElectrodes(
        dataclasses.replace(config, mesh=coarse),
        cache_dir=tmp_path_factory.mktemp("cache"),
    ).get()


@pytest.fixture(scope="module")
def two_rows_forward(two_rows_data):
    gamma = sphere_conductivity(two_rows_data.mesh, InclusionConfig())
    return CEMForward(two_rows_data, gamma)


@pytest.fixture(scope="module")
def two_rows(two_rows_data, two_rows_forward):
    return two_rows_data, CEMDerivative(two_rows_forward, adjacent_by_row([8, 8]))


@pytest.fixture(scope="module")
def two_rows_jacobian(two_rows):
    return two_rows[1].jacobian()


def test_two_rows_jacobian_has_sixteen_blocks(two_rows, two_rows_jacobian):
    data, _ = two_rows
    assert two_rows_jacobian.shape == (
        16 * 16,
        data.mesh.topology.index_map(3).size_local,
    )


@pytest.mark.parametrize(
    "profile",
    [lambda x: np.ones(len(x)), lambda x: x[:, 2], lambda x: np.sin(5 * x[:, 0])],
    ids=["constant", "linear", "oscillating"],
)
def test_two_rows_jacobian_times_any_direction_matches_the_derivative(
    two_rows, two_rows_jacobian, profile
):
    data, derivative = two_rows
    eta = midpoint_direction(data.mesh, profile)
    expected = derivative.directional(eta).ravel()
    assert np.allclose(
        two_rows_jacobian @ eta.x.array,
        expected,
        rtol=1e-8,
        atol=1e-10 * np.abs(expected).max(),
    )


def test_two_rows_jacobian_column_is_the_derivative_along_the_cell(
    two_rows, two_rows_jacobian
):
    data, derivative = two_rows
    cell = two_rows_jacobian.shape[1] // 2
    expected = derivative.directional(cell_indicator(data.mesh, cell)).ravel()
    assert np.allclose(
        two_rows_jacobian[:, cell],
        expected,
        rtol=1e-6,
        atol=1e-8 * np.abs(expected).max(),
    )


def test_two_rows_difference_quotient_converges_at_first_order(two_rows):
    data, derivative = two_rows
    eta = sphere_direction(data.mesh, DirectionConfig())
    expected = derivative.directional(eta)
    errors = [
        np.linalg.norm(derivative.difference_quotient(eta, t) - expected)
        for t in (2.0**-6, 2.0**-7)
    ]
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.05)


class TheoryCase(NamedTuple):
    data: MeshData
    forward: CEMForward
    derivative: CEMDerivative
    currents: np.ndarray
    eta: dolfinx.fem.Function
    variation: np.ndarray


@pytest.fixture(scope="module", params=["default", "two_rows_8"])
def theory(request, mesh_data, forward, two_rows_data, two_rows_forward, two_rows):
    if request.param == "default":
        data, solver, currents = mesh_data, forward, adjacent_by_row([16])
        shared = CEMDerivative(solver, currents)
    else:
        data, solver, currents = (
            two_rows_data,
            two_rows_forward,
            adjacent_by_row([8, 8]),
        )
        shared = two_rows[1]
    eta = sphere_direction(data.mesh, DirectionConfig())
    return TheoryCase(data, solver, shared, currents, eta, shared.directional(eta))


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _power_matrix(case: TheoryCase) -> np.ndarray:
    return case.variation @ case.currents.T


def _energy_matrix(case: TheoryCase) -> np.ndarray:
    space = case.forward.function_space
    left, right = dolfinx.fem.Function(space), dolfinx.fem.Function(space)
    energy = dolfinx.fem.form(
        case.eta * ufl.inner(ufl.grad(left), ufl.grad(right)) * ufl.dx
    )
    potentials = [case.forward.potential(U).x.array for U in case.derivative.voltages]
    matrix = np.empty((len(potentials), len(potentials)))
    for j, u_j in enumerate(potentials):
        left.x.array[:] = u_j
        for k, u_k in enumerate(potentials):
            right.x.array[:] = u_k
            matrix[j, k] = dolfinx.fem.assemble_scalar(energy)
    return matrix


def test_derivative_is_reciprocal_for_every_pair_of_currents(theory):
    power = _power_matrix(theory)
    assert _relative_error(power, power.T) < TOL_IDENTITY


def test_power_is_minus_the_energy_for_every_pair_of_currents(theory):
    assert _relative_error(_power_matrix(theory), -_energy_matrix(theory)) < (
        TOL_IDENTITY
    )


def test_derivative_along_the_conductivity_is_homogeneous(theory):
    gamma = theory.forward.conductivity
    impedance = CEMConfig().impedance

    def voltages(scale: float) -> np.ndarray:
        config = CEMConfig(impedance=scale * impedance)
        solver = CEMForward(theory.data, gamma, config)
        return solver.solve(theory.currents).voltages

    impedance_term = (voltages(1 + STEP_IMPEDANCE) - voltages(1 - STEP_IMPEDANCE)) / (
        2 * STEP_IMPEDANCE
    )
    expected = -theory.derivative.voltages + impedance_term
    assert _relative_error(theory.derivative.directional(gamma), expected) < (
        TOL_HOMOGENEITY
    )


@pytest.mark.parametrize("matrix", [_energy_matrix, _power_matrix])
def test_energy_rank_is_electrodes_minus_rows(theory, matrix):
    values = matrix(theory)
    expected = len(theory.data.layout) - len(theory.data.layout.rows)
    assert (
        np.linalg.matrix_rank(values, tol=TOL_RANK * np.abs(values).max()) == expected
    )


def test_order_is_one_along_a_direction_not_parallel_to_the_conductivity(theory):
    eta = midpoint_direction(theory.data.mesh, lambda x: x[:, 2])
    summary = _summarize_direction(
        theory.derivative, eta, theory.derivative.directional(eta), FEW_EXPONENTS
    )
    assert abs(summary.alignment) < 0.5
    assert summary.order == pytest.approx(1.0, abs=0.01)


def test_spheres_are_nearly_parallel_to_the_conductivity(theory):
    summary = _summarize_direction(
        theory.derivative, theory.eta, theory.variation, FEW_EXPONENTS
    )
    t = 0.5 ** FEW_EXPONENTS[-1]
    assert summary.alignment > 0.99
    assert summary.smallest == pytest.approx(t / (1 + t), rel=0.05)


def test_note_explains_a_k_for_the_spheres_nearly_parallel_to_the_conductivity(theory):
    alignment = _alignment(theory.variation, theory.derivative.voltages)
    note = _alignment_note(alignment)
    assert note.startswith(f"cos(W, -U) = {alignment:.4f}")
    assert "a_k ≈ t/(1 + t)" in note


def test_note_shows_only_the_cosine_for_a_direction_not_parallel(theory):
    eta = midpoint_direction(theory.data.mesh, lambda x: x[:, 2])
    variation = theory.derivative.directional(eta)
    alignment = _alignment(variation, theory.derivative.voltages)
    assert _alignment_note(alignment) == f"cos(W, -U) = {alignment:.4f}"


def test_rejects_a_constant_conductivity(mesh_data):
    gamma = dolfinx.fem.Constant(mesh_data.mesh, 1.0)
    with pytest.raises(TypeError):
        CEMDerivative(CEMForward(mesh_data, gamma), adjacent_pattern(16, 1))
