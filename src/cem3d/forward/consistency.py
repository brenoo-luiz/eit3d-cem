from __future__ import annotations

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import ufl
from mpi4py import MPI

from cem3d.config import electrode_tag
from cem3d.forward.cem import CEMForward, CEMSolution
from cem3d.geometry.mesh import MeshData, boundary_measure


def _to_scipy(A) -> sp.csr_matrix:
    indptr, indices, values = A.getValuesCSR()
    return sp.csr_matrix((values, indices, indptr), shape=A.getSize())


class ConsistencyTests:
    def __init__(
        self, forward: CEMForward, data: MeshData, gamma: dolfinx.fem.Function
    ) -> None:
        self._forward = forward
        V = forward.function_space
        mesh = data.mesh
        self._comm = mesh.comm
        self._u = dolfinx.fem.Function(V)
        p, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        ds = boundary_measure(data)
        n = ufl.FacetNormal(mesh)
        tags = [electrode_tag(e.index) for e in data.layout]

        self._gradient_flux = dolfinx.fem.form(
            gamma * ufl.dot(ufl.grad(self._u), n) * v * ufl.ds
        )
        self._weak_flux = dolfinx.fem.form(
            gamma * ufl.inner(ufl.grad(self._u), ufl.grad(v)) * ufl.dx
        )
        self._electrode_fluxes = [
            dolfinx.fem.form(gamma * ufl.dot(ufl.grad(self._u), n) * ds(t))
            for t in tags
        ]

        mesh.topology.create_connectivity(2, 3)
        facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
        self._boundary = dolfinx.fem.locate_dofs_topological(V, 2, facets)
        local = {dof: k for k, dof in enumerate(self._boundary)}
        mass = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(p * v * ufl.ds))
        mass.assemble()
        sub = _to_scipy(mass)[self._boundary][:, self._boundary]
        mass.destroy()
        self._mass = spla.splu(sub.tocsc())
        self._lumped = np.asarray(sub.sum(axis=1)).ravel()

        self._electrode_dofs = [
            dolfinx.fem.locate_dofs_topological(V, 2, data.facet_tags.find(t))
            for t in tags
        ]
        self._electrode_local = [
            np.array([local[d] for d in dofs]) for dofs in self._electrode_dofs
        ]
        self._on_electrode = np.zeros(self._boundary.size, dtype=bool)
        for k in self._electrode_local:
            self._on_electrode[k] = True
        self._points = V.tabulate_dof_coordinates()[self._boundary]

        u2 = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 2)))
        self._u2 = u2
        q = ufl.TestFunction(dolfinx.fem.functionspace(mesh, ("DG", 0)))
        self._divergence = dolfinx.fem.form(
            ufl.div(gamma * ufl.grad(u2)) * q / ufl.CellVolume(mesh) * ufl.dx
        )

    @property
    def boundary_points(self) -> np.ndarray:
        return self._points

    @property
    def on_electrode(self) -> np.ndarray:
        return self._on_electrode

    def _set(self, voltages: np.ndarray) -> None:
        self._u.x.array[:] = self._forward.potential(voltages).x.array

    def divergence(self, solution: CEMSolution) -> np.ndarray:
        result = []
        for voltages in solution.voltages:
            self._set(voltages)
            self._u2.interpolate(self._u)
            cells = dolfinx.fem.assemble_vector(self._divergence).array
            result.append(np.abs(cells).max())
        return np.array(result)

    def currents(self, solution: CEMSolution, consistent: bool) -> np.ndarray:
        result = np.empty_like(solution.voltages)
        for i, voltages in enumerate(solution.voltages):
            self._set(voltages)
            if consistent:
                weak = dolfinx.fem.assemble_vector(self._weak_flux).array
                result[i] = [weak[dofs].sum() for dofs in self._electrode_dofs]
            else:
                result[i] = [
                    self._comm.allreduce(dolfinx.fem.assemble_scalar(f), op=MPI.SUM)
                    for f in self._electrode_fluxes
                ]
        return result

    def boundary_flux(self, voltages: np.ndarray, consistent: bool) -> np.ndarray:
        self._set(voltages)
        if consistent:
            weak = dolfinx.fem.assemble_vector(self._weak_flux).array
            return weak[self._boundary] / self._lumped
        rhs = dolfinx.fem.assemble_vector(self._gradient_flux).array[self._boundary]
        return self._mass.solve(rhs)

    def gap_flux(self, solution: CEMSolution, consistent: bool) -> float:
        return max(
            np.abs(self.boundary_flux(v, consistent)[~self._on_electrode]).max()
            for v in solution.voltages
        )

    def _contact(self, flux: np.ndarray, index: int) -> np.ndarray:
        nodes = self._electrode_local[index - 1]
        return (
            self._u.x.array[self._electrode_dofs[index - 1]]
            + self._forward.impedances[index - 1] * flux[nodes]
        )

    def contact_profile(
        self, voltages: np.ndarray, index: int, consistent: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        flux = self.boundary_flux(voltages, consistent)
        return self._points[self._electrode_local[index - 1]], self._contact(
            flux, index
        )

    def contact_voltages(self, solution: CEMSolution, consistent: bool) -> np.ndarray:
        count = len(self._electrode_dofs)
        result = np.empty_like(solution.voltages)
        for i, voltages in enumerate(solution.voltages):
            flux = self.boundary_flux(voltages, consistent)
            result[i] = [self._contact(flux, j).mean() for j in range(1, count + 1)]
        return result
