from __future__ import annotations

import dolfinx
import dolfinx.geometry
import numpy as np


def horizontal_slice(
    functions: list[dolfinx.fem.Function], radius: float, z: float, n: int = 241
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    mesh = functions[0].function_space.mesh
    x = np.linspace(-radius, radius, n)
    X, Y = np.meshgrid(x, x)
    inside = X**2 + Y**2 <= (0.999 * radius) ** 2
    points = np.column_stack([X[inside], Y[inside], np.full(inside.sum(), z)])
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, points)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, points)
    cells = np.array(
        [colliding.links(i)[0] for i in range(len(points))], dtype=np.int32
    )
    values = []
    for f in functions:
        V = np.full(X.shape, np.nan)
        V[inside] = f.eval(points, cells).ravel()
        values.append(V)
    return X, Y, values
