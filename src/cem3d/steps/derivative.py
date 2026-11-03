from __future__ import annotations

from pathlib import Path

import numpy as np

from cem3d.config import (
    CACHE_DIR,
    OUTPUTS_DIR,
    CEMConfig,
    DirectionConfig,
    InclusionConfig,
    SetupConfig,
)
from cem3d.fields import sphere_conductivity, sphere_direction
from cem3d.forward.cem import CEMForward
from cem3d.forward.currents import adjacent_by_row
from cem3d.forward.derivative import CEMDerivative
from cem3d.geometry.mesh import CylinderWithElectrodes, cache_path
from cem3d.plots.derivative import render_convergence
from cem3d.utils import report

REFINEMENTS = 50
HALVING = 0.5
ORDER_FIRST = 5
ORDER_LAST = 15
TOL_ZERO = 1e-8
TOL_ORDER = 0.1
TOL_SMALLEST = 1e-5


def _sphere_list(direction: DirectionConfig) -> str:
    return " e ".join(f"({x:g}, {y:g}, {z:g})" for x, y, z in direction.centers)


def _convergence_rows(steps: np.ndarray, errors: np.ndarray) -> list[list[str]]:
    rows = []
    previous = None
    for k, (t, a) in enumerate(zip(steps, errors, strict=True), start=1):
        ratio = "" if previous is None else f"{previous / a:.3f}"
        rows.append([str(k), f"{t:.2e}", f"{a:.2e}", ratio])
        previous = a
    return rows


def _descent_order(steps: np.ndarray, errors: np.ndarray) -> float:
    window = slice(ORDER_FIRST - 1, ORDER_LAST)
    return float(np.polyfit(np.log(steps[window]), np.log(errors[window]), 1)[0])


def run(
    config: SetupConfig | None = None,
    cem: CEMConfig | None = None,
    inclusion: InclusionConfig | None = None,
    direction: DirectionConfig | None = None,
    force: bool = False,
    cache_dir: Path = CACHE_DIR,
    outputs: Path = OUTPUTS_DIR,
) -> bool:
    config = config or SetupConfig()
    cem = cem or CEMConfig()
    inclusion = inclusion or InclusionConfig()
    direction = direction or DirectionConfig()
    currents = adjacent_by_row([row.count for row in config.electrodes.rows])
    currents = currents[:1]

    report.title("DERIVADA DO OPERADOR DIRETO  (equação 2.10 e seção 3.2.2 do livro)")
    report.info("Impedância de contato", f"z = {cem.impedance:g} em todos os eletrodos")
    x, y, z = inclusion.center
    report.info(
        "Condutividade γ",
        f"esfera em ({x:g}, {y:g}, {z:g}), raio {inclusion.radius:g}, "
        f"γ = {inclusion.inside:g} (fundo {inclusion.outside:g})",
    )
    report.info(
        "Direção η",
        f"{direction.outside:g} no cilindro, {direction.inside:g} nas esferas "
        f"de raio {direction.radius:g} em {_sphere_list(direction)}",
    )
    report.info("Teste a_k", "corrente I(1) = (1, -1, 0, ..., 0)")
    report.info("Passos", f"t_k = {HALVING:g}^k, k = 1, ..., {REFINEMENTS}")

    report.section("Malha")
    if force or not cache_path(config, cache_dir).exists():
        print("  Gerando a malha (só na primeira vez)...")
    data = CylinderWithElectrodes(config, cache_dir=cache_dir, force=force).get()
    report.mesh(data)

    gamma = sphere_conductivity(data.mesh, inclusion)
    eta = sphere_direction(data.mesh, direction)
    derivative = CEMDerivative(CEMForward(data, gamma, cem), currents)
    variational = derivative.directional(eta)
    single = variational[0]

    steps = HALVING ** np.arange(1, REFINEMENTS + 1)
    errors = np.array(
        [
            np.linalg.norm(derivative.difference_quotient(eta, t)[0] - single)
            / np.linalg.norm(single)
            for t in steps
        ]
    )

    report.section("Uma corrente: W = F'_I(γ)η pela equação (2.10), I = I(1)")
    report.matrix(single[None], row="I", column="e", fmt=".2e")
    report.section(
        "Teste de consistência  a_k = |(F_I(γ + t_k η) - F_I(γ))/t_k - W| / |W|"
    )
    report.table(["k", "t_k", "a_k", "a_(k-1)/a_k"], _convergence_rows(steps, errors))

    total = float(np.abs(single.sum()))
    order = _descent_order(steps, errors)
    best = int(np.argmin(errors))
    report.section("Verificações")
    ok = all(
        [
            report.check("Soma de F'_I(γ)η", total, total < TOL_ZERO, ideal="~0"),
            report.check(
                f"Ordem na descida (k = {ORDER_FIRST}..{ORDER_LAST})",
                order,
                abs(order - 1) < TOL_ORDER,
                ideal="1",
                fmt=".3f",
            ),
            report.check(
                "Menor a_k",
                errors[best],
                errors[best] < TOL_SMALLEST,
                ideal="~0",
            ),
        ]
    )
    report.info("Menor a_k em", f"k = {best + 1}, t_k = {steps[best]:.2e}")

    paths = [render_convergence(steps, errors, outputs / "step4_convergence.png")]
    report.result(
        ok,
        "a derivada passa no teste de consistência",
        "há verificações incorretas",
    )
    report.files(paths)
    return ok
