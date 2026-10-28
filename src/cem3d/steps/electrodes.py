from __future__ import annotations

from pathlib import Path

import numpy as np

from cem3d.config import (
    BOTTOM_TAG,
    CACHE_DIR,
    GAP_TAG,
    OUTPUTS_DIR,
    TOP_TAG,
    SetupConfig,
    electrode_tag,
)
from cem3d.geometry.electrodes import TWO_PI, ElectrodeLayout, ElectrodeRow
from cem3d.geometry.mesh import (
    CylinderWithElectrodes,
    cache_path,
    facet_area,
    facet_centroid,
)
from cem3d.plots.electrodes import render_layout
from cem3d.utils import report

TOL_AREA = 1e-3
TOL_ANGLE = 1e-3


def _angle_gap(a: float, b: float) -> float:
    return abs((a - b + np.pi) % TWO_PI - np.pi)


def _describe(config: SetupConfig, layout: ElectrodeLayout) -> None:
    m, e = config.mesh, config.electrodes
    report.info(
        "Cilindro", f"raio {m.radius:g}, altura {m.height:g} (H/R = {m.aspect:g})"
    )
    report.info(
        "Eletrodos",
        f"{len(e.rows)} linha(s), {e.count} eletrodos, altura/largura {e.aspect:.4g}",
    )
    report.info("Folga vertical entre linhas", f"{layout.vertical_gap:.4g}")
    report.info("Referência de θ", "geratriz θ = 0 (eixo x), linha 1 no topo")
    for row in layout.rows:
        r = row.config
        indices = [el.index for el in row]
        first, last = indices[0], indices[-1]
        sense = "horário" if r.clockwise else "anti-horário"
        report.info(
            f"Linha {row.number} (e{first} a e{last})",
            f"L = {r.count}, cobertura {r.coverage:.0%}, θ = {r.theta:g}°, {sense}",
        )
        report.info(
            "",
            f"largura {r.width(m.radius):.4g}, altura {row.z_max - row.z_min:.4g}, "
            f"z de {row.z_min:.4g} a {row.z_max:.4g}",
        )


def _first_start(row: ElectrodeRow, angle: float) -> float:
    return float(np.degrees((row.sense * angle - row.width / 2) % TWO_PI))


def run(
    config: SetupConfig | None = None,
    force: bool = False,
    cache_dir: Path = CACHE_DIR,
    outputs: Path = OUTPUTS_DIR,
) -> bool:
    config = config or SetupConfig()
    m = config.mesh

    mesher = CylinderWithElectrodes(config, cache_dir=cache_dir, force=force)

    report.title("GEOMETRIA  (cilindro com eletrodos)")
    _describe(config, mesher.layout)

    report.section("Malha")
    if force or not cache_path(config, cache_dir).exists():
        print("  Gerando a malha (só na primeira vez)...")
    data = mesher.get()
    report.mesh(data)

    layout = data.layout
    areas = np.array([facet_area(data, electrode_tag(el.index)) for el in layout])
    exact = np.array([el.area(m.radius) for el in layout])
    area_err = float(np.max(np.abs(areas - exact) / exact))

    centroids = np.array(
        [facet_centroid(data, electrode_tag(el.index)) for el in layout]
    )
    angles = np.arctan2(centroids[:, 1], centroids[:, 0])
    angle_err = max(
        _angle_gap(a, el.theta_center) for a, el in zip(angles, layout, strict=True)
    )
    height_err = float(np.max(np.abs(centroids[:, 2] - [el.z_center for el in layout])))

    boundary = (
        sum(facet_area(data, t) for t in (TOP_TAG, BOTTOM_TAG, GAP_TAG)) + areas.sum()
    )
    boundary_exact = TWO_PI * m.radius * m.height + 2 * np.pi * m.radius**2
    boundary_err = abs(boundary - boundary_exact) / boundary_exact
    tol_boundary = (m.size_max / m.radius) ** 2 / 6

    report.section("Eletrodos na malha")
    checks = [
        report.check(
            "Erro relativo da área (maior)", area_err, area_err < TOL_AREA, ideal="~0"
        ),
        report.check(
            "Erro da posição angular (maior, rad)",
            angle_err,
            angle_err < TOL_ANGLE,
            ideal="~0",
        ),
        report.check(
            "Erro da posição vertical (maior)",
            height_err,
            height_err < TOL_AREA,
            ideal="~0",
        ),
    ]
    start = 0
    for row in layout.rows:
        stop = start + len(row)
        row_angles = angles[start:stop]
        steps = (np.roll(row_angles, -1) - row_angles) * row.sense % TWO_PI
        order_err = max(_angle_gap(s, row.sector) for s in steps)
        first = _first_start(row, row_angles[0])
        theta = row.config.theta % 360
        coverage = areas[start:stop].sum() / (
            TWO_PI * m.radius * (row.z_max - row.z_min)
        )
        checks += [
            report.check(
                f"Linha {row.number}: ordem e espaçamento (rad)",
                order_err,
                order_err < TOL_ANGLE,
                ideal="~0",
            ),
            report.check(
                f"Linha {row.number}: início do 1º eletrodo (°)",
                first,
                _angle_gap(np.radians(first), np.radians(theta)) < TOL_ANGLE,
                ideal=f"{theta:.2f}",
                fmt=".2f",
            ),
            report.check(
                f"Linha {row.number}: cobertura",
                coverage,
                abs(coverage - row.config.coverage) < TOL_AREA,
                ideal=f"{row.config.coverage:g}",
                fmt=".4f",
            ),
        ]
        start = stop
    checks.append(
        report.check(
            "Erro relativo da área da borda",
            boundary_err,
            boundary_err < tol_boundary,
            ideal="~0",
        )
    )
    ok = all(checks)

    path = render_layout(data, m.radius, m.height, outputs / "step1_electrodes.png")
    report.result(
        ok,
        "eletrodos posicionados e marcados corretamente",
        "há verificações incorretas",
    )
    report.files([path])
    return ok
