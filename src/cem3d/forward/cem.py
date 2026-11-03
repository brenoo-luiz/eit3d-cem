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
        self._ksp = self._build_solver()
        self._loads = self._electrode_loads()
        self._basis = [self._solve_interior(c) for c in self._loads]
        self._schur = self._schur_complement()

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        return self._V

    @property
    def conductivity(self) -> dolfinx.fem.Function | dolfinx.fem.Constant:
        return self._gamma

    @property
    def schur(self) -> np.ndarray:
        return self._schur

    @property
    def areas(self) -> np.ndarray:
        return self._areas

    @property
    def impedances(self) -> np.ndarray:
        return self._z

    def with_conductivity(
        self, gamma: dolfinx.fem.Function | dolfinx.fem.Constant
    ) -> CEMForward:
        return CEMForward(self._data, gamma, self._config)

    def solve(self, currents: np.ndarray) -> CEMSolution:
        currents = np.atleast_2d(np.asarray(currents, dtype=float))
        if currents.shape[1] != self._count:
            raise ValueError(f"each current pattern needs {self._count} values")
        if np.any(np.abs(currents.sum(axis=1)) > 1e-12 * np.abs(currents).max()):
            raise ValueError("each current pattern must sum to zero")
        return CEMSolution(currents, self.solve_schur(currents))

    def solve_source(
        self, source: PETSc.Vec
    ) -> tuple[dolfinx.fem.Function, np.ndarray]:
        omega = self._solve_interior(source)
        coupling = np.array([[c.dot(omega.x.petsc_vec) for c in self._loads]])
        voltages = self.solve_schur(coupling)[0]
        omega.x.array[:] += self.potential(voltages).x.array
        return omega, voltages

    def solve_schur(self, rhs: np.ndarray) -> np.ndarray:
        n = self._count
        K = np.zeros((n + 1, n + 1))
        K[:n, :n] = self._schur
        K[:n, n] = K[n, :n] = 1.0
        rhs = np.vstack([rhs.T, np.zeros((1, len(rhs)))])
        return np.linalg.solve(K, rhs)[:n].T

    def potential(self, voltages: np.ndarray) -> dolfinx.fem.Function:
        u = dolfinx.fem.Function(self._V)
        u.x.array[:] = sum(
            U_j * w.x.array for U_j, w in zip(voltages, self._basis, strict=True)
        )
        return u

    def _tags(self) -> list[int]:
        return [electrode_tag(e.index) for e in self._data.layout]

    def _build_solver(self) -> PETSc.KSP:
        u, v = ufl.TrialFunction(self._V), ufl.TestFunction(self._V)
        ds = boundary_measure(self._data)
        a = ufl.inner(self._gamma * ufl.grad(u), ufl.grad(v)) * ufl.dx
        a += sum(
            (1 / z) * u * v * ds(tag)
            for z, tag in zip(self._z, self._tags(), strict=True)
        )
        A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
        A.assemble()

        ksp = PETSc.KSP().create(self._data.mesh.comm)
        ksp.setOptionsPrefix(_PREFIX)
        options = PETSc.Options()
        for key, value in _AMG.items():
            options[f"{_PREFIX}{key}"] = value
        try:
            ksp.setOperators(A)
            ksp.setType("cg")
            ksp.getPC().setType("hypre")
            ksp.setTolerances(rtol=self._config.rtol, max_it=self._config.max_it)
            ksp.setFromOptions()
            ksp.setUp()
        finally:
            for key in _AMG:
                options.delValue(f"{_PREFIX}{key}")
        return ksp

    def _electrode_loads(self) -> list[PETSc.Vec]:
        v = ufl.TestFunction(self._V)
        ds = boundary_measure(self._data)
        loads = []
        for z, tag in zip(self._z, self._tags(), strict=True):
            c = dolfinx.fem.petsc.assemble_vector(
                dolfinx.fem.form((1 / z) * v * ds(tag))
            )
            c.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
            loads.append(c)
        return loads

    def _solve_interior(self, load: PETSc.Vec) -> dolfinx.fem.Function:
        w = dolfinx.fem.Function(self._V)
        self._ksp.solve(load, w.x.petsc_vec)
        if self._ksp.getConvergedReason() <= 0:
            raise RuntimeError(
                f"CG did not converge ({self._ksp.getConvergedReason()})"
            )
        w.x.scatter_forward()
        return w

    def _schur_complement(self) -> np.ndarray:
        n = self._count
        S = np.diag(self._areas / self._z)
        for k in range(n):
            for j in range(n):
                S[k, j] -= self._loads[k].dot(self._basis[j].x.petsc_vec)
        return (S + S.T) / 2
