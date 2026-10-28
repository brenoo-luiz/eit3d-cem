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
from cem3d.fields import constant_conductivity, sphere_conductivity
from cem3d.forward.cem import CEMForward
from cem3d.forward.checks import (
    antisymmetry_error,
    circulant_error,
    extremes_off_electrodes,
    injected_currents,
)
from cem3d.forward.currents import adjacent_by_row
from cem3d.geometry.mesh import CylinderWithElectrodes, cache_path
from cem3d.plots.forward import render_potential
from cem3d.utils import report

TOL_ZERO = 1e-8
TOL_SYMMETRY = 1e-2


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

    report.title("PROBLEMA DIRETO  (modelo completo de eletrodos)")
    report.info("Impedância de contato", f"z = {cem.impedance:g} em todos os eletrodos")
    report.info(
        "Correntes",
        f"adjacentes em cada linha, {count} experimentos, I = (1, -1, 0, ..., 0)",
    )
    report.info("Medidas", f"U em todos os {count} eletrodos, {count * count} valores")
    x, y, z = inclusion.center
    report.info(
        "Inclusão",
        f"esfera em ({x:g}, {y:g}, {z:g}), raio {inclusion.radius:g}, "
        f"γ = {inclusion.inside:g} (fundo {inclusion.outside:g})",
    )

    report.section("Malha")
    if force or not cache_path(config, cache_dir).exists():
        print("  Gerando a malha (só na primeira vez)...")
    data = CylinderWithElectrodes(config, cache_dir=cache_dir, force=force).get()
    report.mesh(data)

    currents = adjacent_by_row([row.count for row in config.electrodes.rows])
    homogeneous = CEMForward(data, constant_conductivity(data.mesh, 1.0), cem)
    with_inclusion = CEMForward(data, sphere_conductivity(data.mesh, inclusion), cem)
    sol_h = homogeneous.solve(currents)
    sol_i = with_inclusion.solve(currents)

    current_err = float(
        np.abs(injected_currents(with_inclusion, data, sol_i) - currents).max()
    )
    sum_err = float(np.abs(sol_i.voltages.sum(axis=1)).max())
    power = currents @ sol_i.voltages.T
    reciprocity_err = float(np.abs(power - power.T).max() / np.abs(power).max())
    off_electrodes = extremes_off_electrodes(sol_i.voltages, currents)
    single_row = len(data.layout.rows) == 1

    report.section(
        "Tensões U nos eletrodos  (linha i: experimento, coluna j: eletrodo)"
    )
    report.matrix(sol_i.voltages, row="I", column="e")

    report.section("Verificações")
    checks = [
        report.check(
            "Corrente nos eletrodos (maior erro)",
            current_err,
            current_err < TOL_ZERO,
            ideal="~0",
        ),
        report.check(
            "Soma das tensões (maior)", sum_err, sum_err < TOL_ZERO, ideal="~0"
        ),
        report.check(
            "Reciprocidade (erro relativo)",
            reciprocity_err,
            reciprocity_err < TOL_ZERO,
            ideal="~0",
        ),
    ]
    if single_row:
        symmetry_err = antisymmetry_error(sol_h.voltages[0])
        circulant_err = circulant_error(sol_h.voltages)
        checks += [
            report.check(
                "Antissimetria com γ ≡ 1 (relativo)",
                symmetry_err,
                symmetry_err < TOL_SYMMETRY,
                ideal="~0",
            ),
            report.check(
                "Circulante com γ ≡ 1 (relativo)",
                circulant_err,
                circulant_err < TOL_SYMMETRY,
                ideal="~0",
            ),
        ]
    checks.append(
        report.check(
            "Máx. e mín. fora dos eletrodos de I",
            off_electrodes,
            off_electrodes == 0,
            fmt="d",
        )
    )
    ok = all(checks)
    if not single_row:
        report.info("Simetrias com γ ≡ 1", "valem só para uma linha de eletrodos")
    difference = sol_i.voltages - sol_h.voltages
    report.info(
        "U(1) nos eletrodos 1 e 2",
        f"{sol_i.voltages[0, 0]:.4f}, {sol_i.voltages[0, 1]:.4f}",
    )
    report.info("Maior efeito da inclusão", f"{np.abs(difference).max():.2e}")

    paths = []
    for row in data.layout.rows:
        experiment = row.first - 1
        potential = with_inclusion.potential(sol_i.voltages[experiment])
        effect = homogeneous.potential(sol_h.voltages[experiment])
        effect.x.array[:] = potential.x.array - effect.x.array
        name = "step2_potential" if single_row else f"step2_potential_row{row.number}"
        paths.append(
            render_potential(
                potential,
                effect,
                data.layout,
                inclusion,
                config.mesh.radius,
                outputs / f"{name}.png",
                z=row.z_center,
            )
        )
    report.result(
        ok,
        "problema direto do CEM resolvido corretamente",
        "há verificações incorretas",
    )
    report.files(paths)
    return ok
