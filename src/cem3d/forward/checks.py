from __future__ import annotations

import dolfinx
import numpy as np
from mpi4py import MPI

from cem3d.config import electrode_tag
from cem3d.forward.cem import CEMForward, CEMSolution
from cem3d.geometry.mesh import MeshData, boundary_measure


def injected_currents(
    forward: CEMForward, data: MeshData, solution: CEMSolution
) -> np.ndarray:
    # Current through e_j: (|e_j| U_j - int u) / z_j.
    u = dolfinx.fem.Function(forward.function_space)
    ds = boundary_measure(data)
    forms = [dolfinx.fem.form(u * ds(electrode_tag(e.index))) for e in data.layout]
    comm = data.mesh.comm
    result = np.empty_like(solution.voltages)
    for i, voltages in enumerate(solution.voltages):
        u.x.array[:] = forward.potential(voltages).x.array
        means = np.array(
            [comm.allreduce(dolfinx.fem.assemble_scalar(f), op=MPI.SUM) for f in forms]
        )
        result[i] = (forward.areas * voltages - means) / forward.impedances
    return result


def antisymmetry_error(voltages: np.ndarray) -> float:
    # With gamma = 1 and current on (e1, e2): U_k = -U_m when k + m = 3 (mod L).
    n = voltages.size
    mirror = (1 - np.arange(n)) % n
    return float(np.abs(voltages + voltages[mirror]).max() / np.abs(voltages).max())
