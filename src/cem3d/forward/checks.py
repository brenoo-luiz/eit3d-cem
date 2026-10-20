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
    n = voltages.size
    mirror = (1 - np.arange(n)) % n
    return float(np.abs(voltages + voltages[mirror]).max() / np.abs(voltages).max())


def circulant_error(voltages: np.ndarray) -> float:
    # Rotational symmetry: row i is row 1 shifted by i - 1 electrodes.
    shifted = np.array([np.roll(voltages[0], i) for i in range(len(voltages))])
    return float(np.abs(voltages - shifted).max() / np.abs(voltages).max())


def extremes_off_electrodes(voltages: np.ndarray, currents: np.ndarray) -> int:
    # Rows whose highest and lowest U are not at the electrodes injecting +I and -I.
    highest = voltages.argmax(axis=1) != currents.argmax(axis=1)
    lowest = voltages.argmin(axis=1) != currents.argmin(axis=1)
    return int(np.count_nonzero(highest | lowest))
