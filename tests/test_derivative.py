import dataclasses

import dolfinx
import numpy as np
import pytest
import ufl

from cem3d.cli import load_setup
from cem3d.config import ROOT_DIR, DirectionConfig, InclusionConfig
from cem3d.fields import (
    cell_indicator,
    midpoint_direction,
    sphere_conductivity,
    sphere_direction,
)
from cem3d.forward.cem import CEMForward
from cem3d.forward.currents import adjacent_by_row, adjacent_pattern
from cem3d.forward.derivative import CEMDerivative
from cem3d.geometry.mesh import CylinderWithElectrodes

EXPERIMENTS = 3


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
def two_rows(tmp_path_factory):
    config = load_setup(ROOT_DIR / "configs" / "two_rows_8.toml")
    coarse = dataclasses.replace(config.mesh, size_max=0.25, size_min=0.08)
    data = CylinderWithElectrodes(
        dataclasses.replace(config, mesh=coarse),
        cache_dir=tmp_path_factory.mktemp("cache"),
    ).get()
    gamma = sphere_conductivity(data.mesh, InclusionConfig())
    return data, CEMDerivative(CEMForward(data, gamma), adjacent_by_row([8, 8]))


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


def test_rejects_a_constant_conductivity(mesh_data):
    gamma = dolfinx.fem.Constant(mesh_data.mesh, 1.0)
    with pytest.raises(TypeError):
        CEMDerivative(CEMForward(mesh_data, gamma), adjacent_pattern(16, 1))
