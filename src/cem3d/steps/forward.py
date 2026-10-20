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
from cem3d.forward.checks import antisymmetry_error, injected_currents
from cem3d.forward.currents import adjacent_pattern
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
        "Correntes", f"adjacentes, {count} experimentos, I = (1, -1, 0, ..., 0)"
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

    currents = adjacent_pattern(count)
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
    symmetry_err = antisymmetry_error(sol_h.voltages[0])

    report.section("Verificações")
    ok = all(
        [
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
            report.check(
                "Antissimetria com γ ≡ 1 (relativo)",
                symmetry_err,
                symmetry_err < TOL_SYMMETRY,
                ideal="~0",
            ),
        ]
    )
    difference = sol_i.voltages - sol_h.voltages
    report.info(
        "U(1) nos eletrodos 1 e 2",
        f"{sol_i.voltages[0, 0]:.4f}, {sol_i.voltages[0, 1]:.4f}",
    )
    report.info("Maior efeito da inclusão", f"{np.abs(difference).max():.2e}")

    potential = with_inclusion.potential(sol_i.voltages[0])
    effect = homogeneous.potential(sol_h.voltages[0])
    effect.x.array[:] = potential.x.array - effect.x.array
    paths = [
        render_potential(
            potential,
            effect,
            data.layout,
            inclusion,
            config.mesh.radius,
            outputs / "step2_potential.png",
        )
    ]
    report.result(
        ok,
        "problema direto do CEM resolvido corretamente",
        "há verificações incorretas",
    )
    report.files(paths)
    return ok
