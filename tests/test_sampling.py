import dolfinx
import numpy as np
import pytest

from cem3d.utils.sampling import horizontal_slice


@pytest.fixture(scope="module")
def linear(mesh_data):
    V = dolfinx.fem.functionspace(mesh_data.mesh, ("Lagrange", 1))
    f = dolfinx.fem.Function(V)
    f.interpolate(lambda x: 2 * x[0] - x[1] + 3 * x[2])
    return f


def test_slice_reproduces_a_linear_function(linear):
    X, Y, (V,) = horizontal_slice([linear], 1.0, 0.25, n=41)
    inside = ~np.isnan(V)
    assert np.allclose(V[inside], 2 * X[inside] - Y[inside] + 0.75)


def test_slice_is_empty_outside_the_disk(linear):
    X, Y, (V,) = horizontal_slice([linear], 1.0, 0.0, n=41)
    assert np.all(np.isnan(V[X**2 + Y**2 > 1.0]))
    assert V.shape == (41, 41)


def test_points_that_miss_the_mesh_are_empty(linear):
    X, Y, (V,) = horizontal_slice([linear], 1.1, 0.1, n=41)
    assert np.all(np.isnan(V[X**2 + Y**2 > 1.0]))
    inside = ~np.isnan(V)
    assert inside.sum() > 0.9 * np.sum(X**2 + Y**2 < 0.95)
    assert np.allclose(V[inside], 2 * X[inside] - Y[inside] + 0.3)
