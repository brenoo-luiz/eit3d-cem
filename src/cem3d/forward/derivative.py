from __future__ import annotations

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from petsc4py import PETSc

from cem3d.forward.cem import CEMForward


class CEMDerivative:
    def __init__(self, forward: CEMForward, currents: np.ndarray) -> None:
        gamma = forward.conductivity
        if not isinstance(gamma, dolfinx.fem.Function):
            raise TypeError("the conductivity must be a DG0 function")
        self._forward = forward
        self._gamma = gamma
        solution = forward.solve(currents)
        self._currents = solution.currents
        self._voltages = solution.voltages
        V = forward.function_space
        Q = dolfinx.fem.functionspace(V.mesh, ("DG", 0))
        self._u = dolfinx.fem.Function(V)
        self._eta = dolfinx.fem.Function(Q)
        grad_u = ufl.grad(self._u)
        self._weighted_energy = dolfinx.fem.form(
            self._eta * ufl.inner(grad_u, ufl.grad(ufl.TestFunction(V))) * ufl.dx
        )

    @property
    def voltages(self) -> np.ndarray:
        return self._voltages

    def directional(self, eta: dolfinx.fem.Function) -> np.ndarray:
        self._eta.x.array[:] = eta.x.array
        rows = []
        for voltages in self._voltages:
            self._u.x.array[:] = self._forward.potential(voltages).x.array
            source = dolfinx.fem.petsc.assemble_vector(self._weighted_energy)
            source.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
            source.scale(-1.0)
            rows.append(self._forward.solve_source(source)[1])
            source.destroy()
        return np.array(rows)

    def difference_quotient(self, eta: dolfinx.fem.Function, step: float) -> np.ndarray:
        perturbed = dolfinx.fem.Function(self._gamma.function_space)
        perturbed.x.array[:] = self._gamma.x.array + step * eta.x.array
        shifted = self._forward.with_conductivity(perturbed)
        return (shifted.solve(self._currents).voltages - self._voltages) / step
