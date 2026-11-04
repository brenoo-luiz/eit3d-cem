import dolfinx
import numpy as np
import pytest
import ufl

from cem3d.config import DirectionConfig, InclusionConfig
from cem3d.fields import (
    cell_indicator,
    constant_conductivity,
    midpoint_direction,
    sphere_conductivity,
    sphere_direction,
)


def test_midpoint_direction_evaluates_the_profile_at_the_cell_midpoints(mesh_data):
    eta = midpoint_direction(mesh_data.mesh, lambda x: x[:, 2])
    midpoints = dolfinx.mesh.compute_midpoints(
        mesh_data.mesh, 3, np.arange(eta.x.array.size, dtype=np.int32)
    )
    assert np.array_equal(eta.x.array, midpoints[:, 2])


def test_cell_indicator_is_one_only_on_the_cell(mesh_data):
    eta = cell_indicator(mesh_data.mesh, 7)
    assert eta.x.array[7] == 1.0
    assert eta.x.array.sum() == 1.0


def test_constant_conductivity(mesh_data):
    gamma = constant_conductivity(mesh_data.mesh, 3.0)
    assert np.all(gamma.x.array == 3.0)


def test_sphere_takes_both_values(mesh_data):
    gamma = sphere_conductivity(
        mesh_data.mesh, InclusionConfig(inside=5.0, outside=1.0)
    )
    assert set(np.unique(gamma.x.array)) == {1.0, 5.0}


def test_direction_is_one_outside_and_two_in_both_spheres(mesh_data):
    direction = DirectionConfig(centers=((0.4, 0.0, 0.4), (-0.4, 0.0, -0.4)))
    eta = sphere_direction(mesh_data.mesh, direction)
    midpoints = dolfinx.mesh.compute_midpoints(
        mesh_data.mesh, 3, np.arange(eta.x.array.size, dtype=np.int32)
    )
    assert set(np.unique(eta.x.array)) == {1.0, 2.0}
    for center in direction.centers:
        inside = np.linalg.norm(midpoints - center, axis=1) < direction.radius
        assert inside.any() and np.all(eta.x.array[inside] == 2.0)


def test_sphere_volume(mesh_data):
    inc = InclusionConfig(center=(0.0, 0.0, 0.0), radius=0.5, inside=2.0, outside=1.0)
    gamma = sphere_conductivity(mesh_data.mesh, inc)
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form((gamma - 1.0) * ufl.dx))
    assert volume == pytest.approx(4 / 3 * np.pi * 0.5**3, rel=0.15)
