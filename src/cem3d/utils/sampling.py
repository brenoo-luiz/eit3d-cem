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
    # The faceted boundary lies inside the circle, so some points hit no cell.
    links = [colliding.links(i) for i in range(len(points))]
    found = np.array([len(cell) > 0 for cell in links])
    cells = np.array([cell[0] for cell in links if len(cell)], dtype=np.int32)
    sampled = inside.copy()
    sampled[inside] = found
    values = []
    for f in functions:
        V = np.full(X.shape, np.nan)
        V[sampled] = f.eval(points[found], cells).ravel()
        values.append(V)
    return X, Y, values
