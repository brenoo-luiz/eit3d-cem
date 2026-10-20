from __future__ import annotations

import dolfinx
import numpy as np

from cem3d.config import InclusionConfig


def constant_conductivity(
    mesh: dolfinx.mesh.Mesh, value: float
) -> dolfinx.fem.Function:
    gamma = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 0)))
    gamma.x.array[:] = value
    return gamma


def sphere_conductivity(
    mesh: dolfinx.mesh.Mesh, inclusion: InclusionConfig
) -> dolfinx.fem.Function:
    gamma = constant_conductivity(mesh, inclusion.outside)
    cells = np.arange(gamma.x.array.size, dtype=np.int32)
    centers = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim, cells)
    inside = (
        np.linalg.norm(centers - np.asarray(inclusion.center), axis=1)
        < inclusion.radius
    )
    gamma.x.array[inside] = inclusion.inside
    return gamma
