from __future__ import annotations

from collections.abc import Callable, Sequence

import dolfinx
import numpy as np

from cem3d.config import DirectionConfig, InclusionConfig


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
    gamma.x.array[_cells_inside(gamma, [inclusion.center], inclusion.radius)] = (
        inclusion.inside
    )
    return gamma


def sphere_direction(
    mesh: dolfinx.mesh.Mesh, direction: DirectionConfig
) -> dolfinx.fem.Function:
    eta = constant_conductivity(mesh, direction.outside)
    eta.x.array[_cells_inside(eta, direction.centers, direction.radius)] = (
        direction.inside
    )
    return eta


def midpoint_direction(
    mesh: dolfinx.mesh.Mesh, profile: Callable[[np.ndarray], np.ndarray]
) -> dolfinx.fem.Function:
    eta = constant_conductivity(mesh, 0.0)
    eta.x.array[:] = profile(_midpoints(eta))
    return eta


def cell_indicator(mesh: dolfinx.mesh.Mesh, cell: int) -> dolfinx.fem.Function:
    eta = constant_conductivity(mesh, 0.0)
    eta.x.array[cell] = 1.0
    return eta


def _midpoints(field: dolfinx.fem.Function) -> np.ndarray:
    mesh = field.function_space.mesh
    cells = np.arange(field.x.array.size, dtype=np.int32)
    return dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim, cells)


def _cells_inside(
    field: dolfinx.fem.Function,
    centers: Sequence[tuple[float, float, float]],
    radius: float,
) -> np.ndarray:
    midpoints = _midpoints(field)
    inside = np.zeros(len(midpoints), dtype=bool)
    for center in centers:
        inside |= np.linalg.norm(midpoints - np.asarray(center), axis=1) < radius
    return inside
