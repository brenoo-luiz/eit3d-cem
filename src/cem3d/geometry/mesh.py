from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import dolfinx
import dolfinx.io
import gmsh
import numpy as np
import ufl
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

from cem3d.config import (
    BOTTOM_TAG,
    CACHE_DIR,
    GAP_TAG,
    TOP_TAG,
    VOLUME_TAG,
    SetupConfig,
    electrode_tag,
)
from cem3d.geometry.electrodes import ElectrodeLayout

_SCHEMA = 1


@dataclass
class MeshData:
    mesh: dolfinx.mesh.Mesh
    facet_tags: dolfinx.mesh.MeshTags
    layout: ElectrodeLayout


def cache_path(config: SetupConfig, cache_dir: Path = CACHE_DIR) -> Path:
    payload = {"v": _SCHEMA, **dataclasses.asdict(config)}
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    return cache_dir / f"mesh_{key}.xdmf"


class CylinderWithElectrodes:
    """Cylinder mesh whose lateral surface is split into tagged electrodes and gap."""

    def __init__(
        self,
        config: SetupConfig,
        comm: MPI.Comm = MPI.COMM_WORLD,
        cache_dir: Path = CACHE_DIR,
        force: bool = False,
    ) -> None:
        self._config = config
        self._comm = comm
        self._file = cache_path(config, cache_dir)
        self._force = force
        self.layout = ElectrodeLayout(config.electrodes)

    @property
    def file(self) -> Path:
        return self._file

    def get(self) -> MeshData:
        if self._file.exists() and not self._force:
            mesh, tags = self._load()
        else:
            mesh, tags = self._generate()
            self._save(mesh, tags)
        return MeshData(mesh, tags, self.layout)

    def _generate(self) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags]:
        if gmsh.isInitialized():
            gmsh.finalize()
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            self._build_geometry()
            self._tag_entities()
            self._set_sizes()
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.optimize("Netgen")
            data = gmshio.model_to_mesh(gmsh.model, self._comm, 0, gdim=3)
        finally:
            gmsh.finalize()
        return data.mesh, data.facet_tags

    def _build_geometry(self) -> None:
        occ = gmsh.model.occ
        m = self._config.mesh
        body = occ.addCylinder(0, 0, -m.height / 2, 0, 0, m.height, m.radius)
        patches = []
        for e in self.layout:
            wedge = occ.addCylinder(
                0, 0, e.z_min, 0, 0, e.height, m.radius, angle=e.width
            )
            occ.rotate([(3, wedge)], 0, 0, 0, 0, 0, 1, e.theta_start)
            patches.append((3, wedge))
        occ.fragment([(3, body)], patches)
        occ.synchronize()

    def _tag_entities(self) -> None:
        m = self._config.mesh
        tol = 1e-6 * m.height
        groups = {TOP_TAG: [], BOTTOM_TAG: [], GAP_TAG: []}
        electrodes = {e.index: [] for e in self.layout}
        for _, s in gmsh.model.getBoundary(
            gmsh.model.getEntities(3), oriented=False, combined=True
        ):
            x, y, z = gmsh.model.occ.getCenterOfMass(2, s)
            if z > m.height / 2 - tol:
                groups[TOP_TAG].append(s)
            elif z < -m.height / 2 + tol:
                groups[BOTTOM_TAG].append(s)
            else:
                index = self.layout.locate(float(np.arctan2(y, x)), z)
                (electrodes[index] if index else groups[GAP_TAG]).append(s)
        for tag, surfaces in groups.items():
            gmsh.model.addPhysicalGroup(2, surfaces, tag=tag)
        for index, surfaces in electrodes.items():
            if not surfaces:
                raise RuntimeError(
                    f"electrode {index} not found on the lateral surface"
                )
            gmsh.model.addPhysicalGroup(2, surfaces, tag=electrode_tag(index))
        gmsh.model.addPhysicalGroup(
            3, [v for _, v in gmsh.model.getEntities(3)], tag=VOLUME_TAG
        )

    def _set_sizes(self) -> None:
        m = self._config.mesh
        edges = self._electrode_edges()
        field = gmsh.model.mesh.field
        distance = field.add("Distance")
        field.setNumbers(distance, "CurvesList", edges)
        field.setNumber(distance, "Sampling", 200)
        threshold = field.add("Threshold")
        field.setNumber(threshold, "InField", distance)
        field.setNumber(threshold, "SizeMin", m.size_min)
        field.setNumber(threshold, "SizeMax", m.size_max)
        field.setNumber(threshold, "DistMin", m.size_min)
        field.setNumber(threshold, "DistMax", 2 * m.size_max)
        field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    def _electrode_edges(self) -> list[int]:
        tags = [electrode_tag(e.index) for e in self.layout]
        edges = set()
        for tag in tags:
            for s in gmsh.model.getEntitiesForPhysicalGroup(2, tag):
                edges.update(
                    abs(c) for _, c in gmsh.model.getBoundary([(2, s)], oriented=False)
                )
        return sorted(edges)

    def _save(self, mesh: dolfinx.mesh.Mesh, tags: dolfinx.mesh.MeshTags) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        mesh.topology.create_connectivity(2, 3)
        with dolfinx.io.XDMFFile(self._comm, str(self._file), "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_meshtags(tags, mesh.geometry)

    def _load(self) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags]:
        with dolfinx.io.XDMFFile(self._comm, str(self._file), "r") as xdmf:
            mesh = xdmf.read_mesh(name="mesh")
            mesh.topology.create_connectivity(2, 3)
            tags = xdmf.read_meshtags(mesh, name="facet_tags")
        return mesh, tags


def boundary_measure(data: MeshData) -> ufl.Measure:
    return ufl.Measure("ds", domain=data.mesh, subdomain_data=data.facet_tags)


def facet_area(data: MeshData, tag: int) -> float:
    ds = boundary_measure(data)
    form = dolfinx.fem.form(dolfinx.fem.Constant(data.mesh, 1.0) * ds(tag))
    return data.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


def facet_centroid(data: MeshData, tag: int) -> np.ndarray:
    ds = boundary_measure(data)
    x = ufl.SpatialCoordinate(data.mesh)
    area = facet_area(data, tag)
    comm = data.mesh.comm
    return np.array(
        [
            comm.allreduce(
                dolfinx.fem.assemble_scalar(dolfinx.fem.form(x[i] * ds(tag))),
                op=MPI.SUM,
            )
            / area
            for i in range(3)
        ]
    )
