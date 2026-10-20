import dolfinx
import numpy as np
import pytest
import ufl

from cem3d.config import InclusionConfig
from cem3d.fields import constant_conductivity, sphere_conductivity


def test_constant_conductivity(mesh_data):
    gamma = constant_conductivity(mesh_data.mesh, 3.0)
    assert np.all(gamma.x.array == 3.0)


def test_sphere_takes_both_values(mesh_data):
    gamma = sphere_conductivity(
        mesh_data.mesh, InclusionConfig(inside=5.0, outside=1.0)
    )
    assert set(np.unique(gamma.x.array)) == {1.0, 5.0}


def test_sphere_volume(mesh_data):
    inc = InclusionConfig(center=(0.0, 0.0, 0.0), radius=0.5, inside=2.0, outside=1.0)
    gamma = sphere_conductivity(mesh_data.mesh, inc)
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form((gamma - 1.0) * ufl.dx))
    assert volume == pytest.approx(4 / 3 * np.pi * 0.5**3, rel=0.15)
