import dolfinx
import dolfinx.fem.petsc
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import ufl

from cem3d.config import (
    ElectrodeConfig,
    InclusionConfig,
    MeshConfig,
    RowConfig,
    SetupConfig,
    electrode_tag,
)
from cem3d.fields import constant_conductivity, sphere_conductivity
from cem3d.forward.cem import CEMForward
from cem3d.forward.checks import (
    antisymmetry_error,
    circulant_error,
    extremes_off_electrodes,
    injected_currents,
)
from cem3d.forward.currents import adjacent_by_row, adjacent_pattern
from cem3d.geometry.mesh import CylinderWithElectrodes, boundary_measure


@pytest.fixture(scope="module")
def gamma(mesh_data):
    return sphere_conductivity(mesh_data.mesh, InclusionConfig())


@pytest.fixture(scope="module")
def forward(mesh_data, gamma):
    return CEMForward(mesh_data, gamma)


@pytest.fixture(scope="module")
def solution(forward):
    return forward.solve(adjacent_pattern(16))


def monolithic_solve(data, gamma, z, currents, source=None):
    V = dolfinx.fem.functionspace(data.mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds = boundary_measure(data)
    tags = [electrode_tag(e.index) for e in data.layout]
    a = ufl.inner(gamma * ufl.grad(u), ufl.grad(v)) * ufl.dx
    a += sum((1 / z) * u * v * ds(t) for t in tags)
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
    A.assemble()
    A = sp.csr_matrix(A.getValuesCSR()[::-1], shape=A.getSize())
    C = np.column_stack(
        [
            dolfinx.fem.petsc.assemble_vector(
                dolfinx.fem.form((1 / z) * v * ds(t))
            ).array
            for t in tags
        ]
    )
    one = dolfinx.fem.Constant(data.mesh, 1.0)
    areas = [dolfinx.fem.assemble_scalar(dolfinx.fem.form(one * ds(t))) for t in tags]
    n = len(tags)
    K = sp.bmat(
        [
            [A, -C, None],
            [-C.T, np.diag(np.array(areas) / z), np.ones((n, 1))],
            [None, np.ones((1, n)), None],
        ],
        format="csc",
    )
    N = A.shape[0]
    rhs = np.zeros((K.shape[0], len(currents)))
    rhs[N : N + n] = currents.T
    if source is not None:
        rhs[:N] = source[:, None]
    x = spla.splu(K).solve(rhs)
    return x[:N].T, x[N : N + n].T


def test_matches_monolithic_solve(mesh_data, gamma, solution):
    currents = adjacent_pattern(16, 3)
    _, expected = monolithic_solve(mesh_data, gamma, 1e-3, currents)
    assert np.allclose(solution.voltages[:3], expected, rtol=1e-6, atol=1e-8)


def test_potential_matches_monolithic_solve(mesh_data, gamma, forward, solution):
    currents = adjacent_pattern(16, 3)
    expected, _ = monolithic_solve(mesh_data, gamma, 1e-3, currents)
    for voltages, u in zip(solution.voltages[:3], expected, strict=True):
        assert np.allclose(forward.potential(voltages).x.array, u, rtol=1e-6, atol=1e-8)


def test_solve_source_matches_monolithic_solve(mesh_data, gamma, forward):
    V = forward.function_space
    f = dolfinx.fem.Function(V)
    f.interpolate(lambda x: x[0] * x[1] - 0.5 * x[2])
    v = ufl.TestFunction(V)
    source = dolfinx.fem.petsc.assemble_vector(
        dolfinx.fem.form(ufl.inner(ufl.grad(f), ufl.grad(v)) * ufl.dx)
    )
    omega, voltages = forward.solve_source(source)
    potential, expected = monolithic_solve(
        mesh_data, gamma, 1e-3, np.zeros((1, 16)), source=source.array
    )
    assert np.allclose(voltages, expected[0], rtol=1e-6, atol=1e-10)
    assert np.allclose(omega.x.array, potential[0], rtol=1e-6, atol=1e-8)
    assert abs(voltages.sum()) < 1e-12


def test_with_conductivity_keeps_mesh_and_contact(mesh_data, forward):
    doubled = constant_conductivity(mesh_data.mesh, 2.0)
    other = forward.with_conductivity(doubled)
    assert other.conductivity is doubled
    assert np.allclose(other.areas, forward.areas)
    assert np.allclose(other.impedances, forward.impedances)


def test_injected_currents_match(forward, mesh_data, solution):
    assert np.allclose(
        injected_currents(forward, mesh_data, solution), solution.currents, atol=1e-8
    )


def test_voltages_sum_to_zero(solution):
    assert np.allclose(solution.voltages.sum(axis=1), 0, atol=1e-12)


def test_reciprocity(solution):
    power = solution.currents @ solution.voltages.T
    assert np.allclose(power, power.T, atol=1e-12)


def test_schur_has_constant_null_space(forward):
    assert np.allclose(forward.schur @ np.ones(16), 0, atol=1e-8)


def test_equal_voltages_give_the_constant_potential(forward):
    assert np.allclose(forward.potential(np.ones(16)).x.array, 1.0, atol=1e-8)


def smallest_eigenvalue(data, gamma, electrode_term):
    V = dolfinx.fem.functionspace(data.mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds = boundary_measure(data)
    a = ufl.inner(gamma * ufl.grad(u), ufl.grad(v)) * ufl.dx
    if electrode_term:
        a += sum(1e3 * u * v * ds(electrode_tag(e.index)) for e in data.layout)
    matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
    matrix.assemble()
    matrix = sp.csr_matrix(matrix.getValuesCSR()[::-1], shape=matrix.getSize())
    return spla.eigsh(matrix, k=1, sigma=-1e-3, return_eigenvectors=False)[0]


def test_electrode_term_makes_the_interior_matrix_positive_definite(mesh_data, gamma):
    assert abs(smallest_eigenvalue(mesh_data, gamma, False)) < 1e-10
    assert smallest_eigenvalue(mesh_data, gamma, True) > 1e-3


def test_homogeneous_antisymmetry(mesh_data):
    sol = CEMForward(mesh_data, constant_conductivity(mesh_data.mesh, 1.0)).solve(
        adjacent_pattern(16, 1)
    )
    assert antisymmetry_error(sol.voltages[0]) < 2e-2


def test_homogeneous_matrix_is_circulant(mesh_data):
    sol = CEMForward(mesh_data, constant_conductivity(mesh_data.mesh, 1.0)).solve(
        adjacent_pattern(16)
    )
    assert circulant_error(sol.voltages) < 2e-2


def test_staggered_rows_keep_the_mirror_symmetry(tmp_path):
    rows = (RowConfig(theta=0.0), RowConfig(theta=11.25))
    config = SetupConfig(
        mesh=MeshConfig(size_max=0.25, size_min=0.08),
        electrodes=ElectrodeConfig(
            aspect=2.0, rows=rows, top_margin=0.4, bottom_margin=0.4
        ),
    )
    data = CylinderWithElectrodes(config, cache_dir=tmp_path).get()
    homogeneous = CEMForward(data, constant_conductivity(data.mesh, 1.0))
    voltages = homogeneous.solve(adjacent_by_row([16, 16])).voltages
    n = 16
    image = np.r_[n + np.arange(n), (np.arange(n) + 1) % n]
    mapped = np.empty_like(voltages)
    mapped[np.ix_(image, image)] = voltages
    assert np.abs(mapped - voltages).max() < 3e-2 * np.abs(voltages).max()


def test_extremes_at_the_injecting_electrodes(solution):
    assert extremes_off_electrodes(solution.voltages, solution.currents) == 0


def test_circulant_error_detects_a_broken_row():
    row = np.arange(16.0) - 7.5
    voltages = np.array([np.roll(row, i) for i in range(16)])
    assert circulant_error(voltages) == 0
    voltages[3, 5] += 1.0
    assert circulant_error(voltages) > 0


def test_extremes_detects_a_wrong_row():
    currents = adjacent_pattern(4)
    voltages = currents.copy()
    voltages[2] = [0.0, 0.0, -1.0, 1.0]
    assert extremes_off_electrodes(voltages, currents) == 1


def test_inclusion_changes_the_data(mesh_data, solution):
    homogeneous = CEMForward(mesh_data, constant_conductivity(mesh_data.mesh, 1.0))
    assert (
        np.abs(
            homogeneous.solve(adjacent_pattern(16)).voltages - solution.voltages
        ).max()
        > 1e-4
    )


def test_contact_voltage_drop(mesh_data, forward, solution):
    u = forward.potential(solution.voltages[0])
    ds = boundary_measure(mesh_data)
    mean = (
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(u * ds(electrode_tag(1))))
        / forward.areas[0]
    )
    drop = forward.impedances[0] * solution.currents[0, 0] / forward.areas[0]
    assert mean == pytest.approx(solution.voltages[0, 0] - drop, rel=1e-8)


def test_rejects_invalid_currents(forward):
    with pytest.raises(ValueError):
        forward.solve(np.ones(16))
    with pytest.raises(ValueError):
        forward.solve(np.zeros(15))
