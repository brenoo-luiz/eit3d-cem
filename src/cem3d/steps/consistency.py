from __future__ import annotations

from pathlib import Path

import numpy as np

from cem3d.config import (
    CACHE_DIR,
    OUTPUTS_DIR,
    CEMConfig,
    InclusionConfig,
    SetupConfig,
)
from cem3d.fields import sphere_conductivity
from cem3d.forward.cem import CEMForward
from cem3d.forward.consistency import ConsistencyTests
from cem3d.forward.currents import adjacent_by_row, driving_electrodes
from cem3d.geometry.mesh import CylinderWithElectrodes, cache_path
from cem3d.plots.consistency import render_boundary, render_contact, render_matrix
from cem3d.utils import report

TOL_ZERO = 1e-8


def run(
    config: SetupConfig | None = None,
    cem: CEMConfig | None = None,
    inclusion: InclusionConfig | None = None,
    force: bool = False,
    cache_dir: Path = CACHE_DIR,
    outputs: Path = OUTPUTS_DIR,
) -> bool:
    config = config or SetupConfig()
    cem = cem or CEMConfig()
    inclusion = inclusion or InclusionConfig()
    count = config.electrodes.count

    report.title("TESTES DE CONSISTÊNCIA  (seção 3.2.1 do livro)")
    report.info("Impedância de contato", f"z = {cem.impedance:g} em todos os eletrodos")
    report.info("Correntes", f"adjacentes em cada linha, {count} experimentos")
    x, y, z = inclusion.center
    report.info(
        "Condutividade",
        f"esfera em ({x:g}, {y:g}, {z:g}), raio {inclusion.radius:g}, "
        f"γ = {inclusion.inside:g} (fundo {inclusion.outside:g})",
    )
    report.info("Fluxo γ ∂u/∂η", "pelo gradiente (livro) e pela forma fraca")

    report.section("Malha")
    if force or not cache_path(config, cache_dir).exists():
        print("  Gerando a malha (só na primeira vez)...")
    data = CylinderWithElectrodes(config, cache_dir=cache_dir, force=force).get()
    report.mesh(data)

    gamma = sphere_conductivity(data.mesh, inclusion)
    forward = CEMForward(data, gamma, cem)
    currents = adjacent_by_row([row.count for row in config.electrodes.rows])
    solution = forward.solve(currents)
    tests = ConsistencyTests(forward, data, gamma)

    divergence = float(tests.divergence(solution).max())
    current_book = currents - tests.currents(solution, consistent=False)
    current_weak = currents - tests.currents(solution, consistent=True)
    gap_book = tests.gap_flux(solution, consistent=False)
    gap_weak = tests.gap_flux(solution, consistent=True)
    contact_book = solution.voltages - tests.contact_voltages(solution, False)
    contact_weak = solution.voltages - tests.contact_voltages(solution, True)
    scale = np.abs(solution.voltages).max()
    tol_contact = config.mesh.size_min / 2

    report.section("Teste 2: I - I*  (fluxo pelo gradiente)")
    report.matrix(current_book, row="I", column="e", fmt=".5f")
    report.section("Teste 4: U - U*  (fluxo pela forma fraca)")
    report.matrix(contact_weak, row="I", column="e", fmt=".1e")

    report.section("Verificações")
    weak = np.abs(current_weak).max()
    contact = np.abs(contact_weak).max() / scale
    ok = all(
        [
            report.check(
                "1. div(γ∇u) em Ω (maior)",
                divergence,
                divergence < TOL_ZERO,
                ideal="~0",
            ),
            report.check("2. I - I* (maior)", weak, weak < TOL_ZERO, ideal="~0"),
            report.check(
                "3. γ ∂u/∂η nos gaps (maior)", gap_weak, gap_weak < TOL_ZERO, ideal="~0"
            ),
            report.check(
                "4. U - U* (maior, relativo)",
                contact,
                contact < tol_contact,
                ideal="~0",
            ),
        ]
    )
    report.section("Mesmos testes com o fluxo pelo gradiente")
    report.info("2. I - I* (maior)", f"{np.abs(current_book).max():.1e}")
    report.info("3. γ ∂u/∂η nos gaps (maior)", f"{gap_book:.1e}")
    report.info(
        "4. U - U* (maior, relativo)", f"{np.abs(contact_book).max() / scale:.1e}"
    )

    band = config.mesh.size_min
    level = data.layout.rows[0].z_center
    levels = [data.layout.row_of(j).z_center for j in driving_electrodes(currents)]
    points, profile = tests.contact_profile(solution.voltages[0], 1, consistent=True)
    fluxes = [tests.boundary_flux(v, consistent=True) for v in solution.voltages]
    book_fluxes = [tests.boundary_flux(v, consistent=False) for v in solution.voltages]
    potentials = [
        tests.boundary_potential(v, consistent=True) for v in solution.voltages
    ]
    paths = [
        render_matrix([(solution.voltages, r"$U$")], outputs / "step3_voltages.png"),
        render_matrix(
            [
                (current_book, r"$I - I^\star$ (gradiente)"),
                (current_weak, r"$I - I^\star$ (forma fraca)"),
            ],
            outputs / "step3_currents.png",
        ),
        render_boundary(
            tests.boundary_points,
            fluxes,
            tests.on_electrode,
            band,
            r"$\gamma\,\partial u / \partial \eta$",
            outputs / "step3_flux.png",
            levels=levels,
        ),
        render_boundary(
            tests.boundary_points,
            book_fluxes,
            tests.on_electrode,
            band,
            r"$\gamma\,\partial u / \partial \eta$ (gradiente)",
            outputs / "step3_flux_gradient.png",
            levels=levels,
        ),
        render_boundary(
            tests.boundary_points,
            potentials,
            tests.on_electrode,
            band,
            r"$u + z\gamma\,\partial u / \partial \eta$",
            outputs / "step3_boundary.png",
            levels=levels,
        ),
        render_contact(
            points,
            profile,
            solution.voltages[0, 0],
            contact_weak,
            band,
            outputs / "step3_contact.png",
            z=level,
        ),
    ]
    report.result(
        ok,
        "as quatro equações do CEM são satisfeitas",
        "há verificações incorretas",
    )
    report.files(paths)
    return ok
