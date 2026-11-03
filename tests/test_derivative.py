import dolfinx
import numpy as np
import pytest
import ufl

from cem3d.config import DirectionConfig, InclusionConfig
from cem3d.fields import sphere_conductivity, sphere_direction
from cem3d.forward.cem import CEMForward
from cem3d.forward.currents import adjacent_pattern
from cem3d.forward.derivative import CEMDerivative

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


def test_difference_quotient_converges_at_first_order(derivative, eta, variational):
    errors = [
        np.linalg.norm(derivative.difference_quotient(eta, t) - variational)
        for t in (2.0**-6, 2.0**-7)
    ]
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.05)
    assert errors[1] < 1e-2 * np.linalg.norm(variational)


def test_rejects_a_constant_conductivity(mesh_data):
    gamma = dolfinx.fem.Constant(mesh_data.mesh, 1.0)
    with pytest.raises(TypeError):
        CEMDerivative(CEMForward(mesh_data, gamma), adjacent_pattern(16, 1))
