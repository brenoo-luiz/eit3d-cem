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
from cem3d.geometry.electrodes import TWO_PI
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


def run(
    config: SetupConfig | None = None,
    force: bool = False,
    cache_dir: Path = CACHE_DIR,
    outputs: Path = OUTPUTS_DIR,
) -> bool:
    config = config or SetupConfig()
    m, e = config.mesh, config.electrodes

    report.title("GEOMETRIA  (cilindro com eletrodos)")
    report.info("Cilindro", f"raio {m.radius:g}, altura {m.height:g}")
    report.info(
        "Eletrodos", f"L = {e.count}, cobertura {e.coverage:.0%}, altura {e.height:g}"
    )
    report.info(
        "Rotação",
        f"{e.rotation:g} rad, sentido {'horário' if e.clockwise else 'anti-horário'}",
    )

    report.section("Malha")
    if force or not cache_path(config, cache_dir).exists():
        print("  Gerando a malha (só na primeira vez)...")
    data = CylinderWithElectrodes(config, cache_dir=cache_dir, force=force).get()
    report.mesh(data)

    layout = data.layout
    areas = np.array([facet_area(data, electrode_tag(el.index)) for el in layout])
    exact = np.array([el.area(m.radius) for el in layout])
    area_err = float(np.max(np.abs(areas - exact) / exact))

    angles = np.array(
        [
            np.arctan2(c[1], c[0])
            for c in (facet_centroid(data, electrode_tag(el.index)) for el in layout)
        ]
    )
    angle_err = max(
        _angle_gap(a, el.theta_center) for a, el in zip(angles, layout, strict=True)
    )

    sense = -1 if e.clockwise else 1
    steps = (np.roll(angles, -1) - angles) * sense % TWO_PI
    order_err = max(_angle_gap(s, layout.sector) for s in steps)

    coverage = areas.sum() / (TWO_PI * m.radius * e.height)
    boundary = (
        sum(facet_area(data, t) for t in (TOP_TAG, BOTTOM_TAG, GAP_TAG)) + areas.sum()
    )
    boundary_exact = TWO_PI * m.radius * m.height + 2 * np.pi * m.radius**2
    boundary_err = abs(boundary - boundary_exact) / boundary_exact
    tol_boundary = (m.size_max / m.radius) ** 2 / 6

    report.section("Eletrodos na malha")
    ok = all(
        [
            report.check(
                "Erro relativo da área (maior)",
                area_err,
                area_err < TOL_AREA,
                ideal="~0",
            ),
            report.check(
                "Erro da posição angular (maior, rad)",
                angle_err,
                angle_err < TOL_ANGLE,
                ideal="~0",
            ),
            report.check(
                "Erro da ordem e espaçamento (rad)",
                order_err,
                order_err < TOL_ANGLE,
                ideal="~0",
            ),
            report.check(
                "Cobertura na faixa dos eletrodos",
                coverage,
                abs(coverage - e.coverage) < TOL_AREA,
                ideal=f"{e.coverage:g}",
                fmt=".4f",
            ),
            report.check(
                "Erro relativo da área da borda",
                boundary_err,
                boundary_err < tol_boundary,
                ideal="~0",
            ),
        ]
    )
    report.info("Área de cada eletrodo", f"{exact[0]:.5f}")
    report.info("Centro de e1", f"{np.degrees(layout[1].theta_center) % 360:.1f}°")

    path = render_layout(data, m.radius, m.height, outputs / "step1_electrodes.png")
    report.result(
        ok,
        "eletrodos posicionados e marcados corretamente",
        "há verificações incorretas",
    )
    report.files([path])
    return ok
