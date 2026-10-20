from __future__ import annotations

from dataclasses import dataclass

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from petsc4py import PETSc

from cem3d.config import CEMConfig, electrode_tag
from cem3d.geometry.mesh import MeshData, boundary_measure, facet_area

_PREFIX = "cem3d_"
_AMG = {
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_strong_threshold": 0.5,
    "pc_hypre_boomeramg_coarsen_type": "HMIS",
    "pc_hypre_boomeramg_interp_type": "ext+i",
}


@dataclass
class CEMSolution:
    currents: np.ndarray
    voltages: np.ndarray


class CEMForward:
    def __init__(
        self,
        data: MeshData,
        gamma: dolfinx.fem.Function | dolfinx.fem.Constant,
        config: CEMConfig | None = None,
    ) -> None:
        self._data = data
        self._config = config or CEMConfig()
        self._V = dolfinx.fem.functionspace(
            data.mesh, ("Lagrange", self._config.degree)
        )
        self._gamma = gamma
        self._count = len(data.layout)
        self._z = np.full(self._count, self._config.impedance)
        self._areas = np.array(
            [facet_area(data, electrode_tag(e.index)) for e in data.layout]
        )
        loads, self._basis = self._solve_basis()
        self._schur = self._schur_complement(loads)

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        return self._V

    @property
    def schur(self) -> np.ndarray:
        return self._schur

    @property
    def areas(self) -> np.ndarray:
        return self._areas

    @property
    def impedances(self) -> np.ndarray:
        return self._z

    def solve(self, currents: np.ndarray) -> CEMSolution:
        currents = np.atleast_2d(np.asarray(currents, dtype=float))
        if currents.shape[1] != self._count:
            raise ValueError(f"each current pattern needs {self._count} values")
        if np.any(np.abs(currents.sum(axis=1)) > 1e-12 * np.abs(currents).max()):
            raise ValueError("each current pattern must sum to zero")
        n = self._count
        K = np.zeros((n + 1, n + 1))
        K[:n, :n] = self._schur
        K[:n, n] = K[n, :n] = 1.0
        rhs = np.vstack([currents.T, np.zeros((1, len(currents)))])
        voltages = np.linalg.solve(K, rhs)[:n].T
        return CEMSolution(currents, voltages)

    def potential(self, voltages: np.ndarray) -> dolfinx.fem.Function:
        u = dolfinx.fem.Function(self._V)
        u.x.array[:] = sum(
            U_j * w.x.array for U_j, w in zip(voltages, self._basis, strict=True)
        )
        return u

    def _solve_basis(self) -> tuple[list[PETSc.Vec], list[dolfinx.fem.Function]]:
        u, v = ufl.TrialFunction(self._V), ufl.TestFunction(self._V)
        ds = boundary_measure(self._data)
        tags = [electrode_tag(e.index) for e in self._data.layout]
        a = ufl.inner(self._gamma * ufl.grad(u), ufl.grad(v)) * ufl.dx
        a += sum(
            (1 / z) * u * v * ds(tag) for z, tag in zip(self._z, tags, strict=True)
        )
        A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
        A.assemble()

        ksp = PETSc.KSP().create(self._data.mesh.comm)
        ksp.setOptionsPrefix(_PREFIX)
        options = PETSc.Options()
        for key, value in _AMG.items():
            options[f"{_PREFIX}{key}"] = value
        ksp.setOperators(A)
        ksp.setType("cg")
        ksp.getPC().setType("hypre")
        ksp.setTolerances(rtol=self._config.rtol, max_it=self._config.max_it)
        ksp.setFromOptions()

        loads, basis = [], []
        try:
            for z, tag in zip(self._z, tags, strict=True):
                c = dolfinx.fem.petsc.assemble_vector(
                    dolfinx.fem.form((1 / z) * v * ds(tag))
                )
                c.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
                w = dolfinx.fem.Function(self._V)
                ksp.solve(c, w.x.petsc_vec)
                if ksp.getConvergedReason() <= 0:
                    raise RuntimeError(
                        f"CG did not converge ({ksp.getConvergedReason()})"
                    )
                w.x.scatter_forward()
                loads.append(c)
                basis.append(w)
        finally:
            ksp.destroy()
            A.destroy()
            for key in _AMG:
                options.delValue(f"{_PREFIX}{key}")
        return loads, basis

    def _schur_complement(self, loads: list[PETSc.Vec]) -> np.ndarray:
        n = self._count
        S = np.diag(self._areas / self._z)
        for k in range(n):
            for j in range(n):
                S[k, j] -= loads[k].dot(self._basis[j].x.petsc_vec)
        for c in loads:
            c.destroy()
        return (S + S.T) / 2
