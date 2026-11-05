from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import dolfinx
import numpy as np

from cem3d.config import (
    CACHE_DIR,
    OUTPUTS_DIR,
    CEMConfig,
    DirectionConfig,
    InclusionConfig,
    SetupConfig,
)
from cem3d.fields import (
    cell_indicator,
    constant_conductivity,
    midpoint_direction,
    sphere_conductivity,
    sphere_direction,
)
from cem3d.forward.cem import CEMForward
from cem3d.forward.currents import get_protocol
from cem3d.forward.derivative import CEMDerivative
from cem3d.geometry.electrodes import ElectrodeLayout
from cem3d.geometry.mesh import CylinderWithElectrodes, cache_path
from cem3d.plots.consistency import render_matrix
from cem3d.plots.derivative import render_convergence
from cem3d.utils import report

EXPERIMENTS = 8
REFINEMENTS = 50
HALVING = 0.5
STEPS = HALVING ** np.arange(1, REFINEMENTS + 1)
ORDER_FIRST = 5
ORDER_LAST = 15
EXTRA_EXPONENTS = np.arange(4, 31)
EXTRA_ORDER_FIRST = 8
EXTRA_ORDER_LAST = 16
COLUMNS = 5
RANDOM_SEED = 0
SPHERES = "esferas"
TOL_ZERO = 1e-8
TOL_ORDER = 0.1
TOL_SMALLEST = 1e-5
TOL_COLUMN = 1e-6
PARALLEL_ALIGNMENT = 0.99


@dataclass(frozen=True)
class DirectionSummary:
    alignment: float
    order: float
    smallest: float


def _sphere_list(direction: DirectionConfig) -> str:
    return " e ".join(f"({x:g}, {y:g}, {z:g})" for x, y, z in direction.centers)


def _report_setup(
    cem: CEMConfig,
    inclusion: InclusionConfig,
    direction: DirectionConfig,
    experiments: int,
    scope: str,
) -> None:
    report.title("DERIVADA DO OPERADOR DIRETO")
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
    report.info(
        "Teste a_k",
        f"corrente I(1) = (1, -1, 0, ..., 0) e as {experiments} correntes juntas",
    )
    report.info("Passos", f"t_k = {HALVING:g}^k, k = 1, ..., {REFINEMENTS}")
    report.info("Correntes", f"{experiments} adjacentes, dentro de {scope}")


def _directions(
    mesh: dolfinx.mesh.Mesh, direction: DirectionConfig
) -> dict[str, dolfinx.fem.Function]:
    rng = np.random.default_rng(RANDOM_SEED)
    return {
        SPHERES: sphere_direction(mesh, direction),
        "constante": constant_conductivity(mesh, 1.0),
        "linear em z": midpoint_direction(mesh, lambda x: x[:, 2]),
        "aleatória": midpoint_direction(mesh, lambda x: rng.standard_normal(len(x))),
    }


def _convergence_errors(
    derivative: CEMDerivative,
    eta: dolfinx.fem.Function,
    variational: np.ndarray,
    steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = (
        np.array([derivative.difference_quotient(eta, t) for t in steps]) - variational
    )
    return (
        np.linalg.norm(residuals, axis=(1, 2)) / np.linalg.norm(variational),
        np.linalg.norm(residuals, axis=2) / np.linalg.norm(variational, axis=1),
    )


def _ratios(errors: np.ndarray) -> list[str]:
    return ["", *(f"{a / b:.3f}" for a, b in pairwise(errors))]


def _log_slope(steps: np.ndarray, errors: np.ndarray) -> np.ndarray:
    return np.polyfit(np.log(steps), np.log(errors), 1)[0]


def _descent_order(errors: np.ndarray) -> np.ndarray:
    window = slice(ORDER_FIRST - 1, ORDER_LAST)
    return _log_slope(STEPS[window], errors[window])


def _alignment(variation: np.ndarray, voltages: np.ndarray) -> float:
    return float(
        -np.sum(variation * voltages)
        / (np.linalg.norm(variation) * np.linalg.norm(voltages))
    )


def _alignment_note(alignment: float) -> str:
    note = f"cos(W, -U) = {alignment:.4f}"
    if alignment > PARALLEL_ALIGNMENT:
        return f"{note}: η quase paralelo a γ, então a_k ≈ t/(1 + t)"
    return note


def _summarize_direction(
    derivative: CEMDerivative,
    eta: dolfinx.fem.Function,
    variation: np.ndarray,
    exponents: np.ndarray,
) -> DirectionSummary:
    steps = HALVING**exponents
    errors = _convergence_errors(derivative, eta, variation, steps)[0]
    window = (exponents >= EXTRA_ORDER_FIRST) & (exponents <= EXTRA_ORDER_LAST)
    return DirectionSummary(
        alignment=_alignment(variation, derivative.voltages),
        order=float(_log_slope(steps[window], errors[window])),
        smallest=float(errors.min()),
    )


def _report_convergence(overall: np.ndarray, per_current: np.ndarray) -> bool:
    experiments = per_current.shape[1]
    single = per_current[:, 0]
    report.section(
        "Teste de consistência  a_k = |(F(γ + t_k η) - F(γ))/t_k - F'(γ)η| / |F'(γ)η|"
    )
    report.table(
        ["k", "t_k", "a_k I(1)", "razão", f"a_k {experiments} correntes", "razão"],
        zip(
            map(str, range(1, REFINEMENTS + 1)),
            (f"{t:.2e}" for t in STEPS),
            (f"{a:.2e}" for a in single),
            _ratios(single),
            (f"{a:.2e}" for a in overall),
            _ratios(overall),
            strict=True,
        ),
    )

    orders = _descent_order(per_current)
    smallest = per_current.min(axis=0)
    report.section("a_k de cada corrente")
    report.table(
        ["I", "ordem", "menor a_k", "em k"],
        [
            [str(i), f"{order:.3f}", f"{a:.2e}", str(k + 1)]
            for i, (order, a, k) in enumerate(
                zip(orders, smallest, per_current.argmin(axis=0), strict=True),
                start=1,
            )
        ],
    )

    order = float(_descent_order(overall))
    worst_order = float(orders[np.argmax(np.abs(orders - 1))])
    window = f"k = {ORDER_FIRST}..{ORDER_LAST}"
    best = int(np.argmin(overall))
    report.section("Verificações do teste a_k")
    ok = all(
        [
            report.check(
                f"Ordem {experiments} correntes ({window})",
                order,
                abs(order - 1) < TOL_ORDER,
                ideal="1",
                fmt=".3f",
            ),
            report.check(
                "Ordem mais distante de 1",
                worst_order,
                abs(worst_order - 1) < TOL_ORDER,
                ideal="1",
                fmt=".3f",
            ),
            report.check(
                f"Menor a_k {experiments} correntes",
                overall[best],
                overall[best] < TOL_SMALLEST,
                ideal="~0",
            ),
            report.check(
                "Maior dos menores a_k",
                smallest.max(),
                smallest.max() < TOL_SMALLEST,
                ideal="~0",
            ),
        ]
    )
    report.info("Menor a_k em", f"k = {best + 1}, t_k = {STEPS[best]:.2e}")
    return ok


def _column_error(column: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(column - expected) / np.linalg.norm(expected))


def _report_columns(
    derivative: CEMDerivative, jacobian: np.ndarray, mesh: dolfinx.mesh.Mesh
) -> bool:
    cells = np.linspace(0, jacobian.shape[1] - 1, COLUMNS, dtype=int)
    errors = [
        _column_error(
            jacobian[:, cell],
            derivative.directional(cell_indicator(mesh, int(cell))).ravel(),
        )
        for cell in cells
    ]
    report.section("Colunas de J contra a equação (2.10) com η = χ_{T_k}")
    report.table(
        ["T_k", "erro relativo"],
        [[str(c), f"{e:.2e}"] for c, e in zip(cells, errors, strict=True)],
    )
    report.section("Verificação das colunas")
    return report.check(
        "Maior erro relativo", max(errors), max(errors) < TOL_COLUMN, ideal="~0"
    )


def _report_directions(
    jacobian: np.ndarray,
    directions: dict[str, dolfinx.fem.Function],
    variations: dict[str, np.ndarray],
) -> bool:
    report.section("J η̄ contra F'(γ)η pela equação (2.10), erro relativo")
    checks = []
    for name, eta in directions.items():
        error = _column_error(jacobian @ eta.x.array, variations[name].ravel())
        checks.append(report.check(f"η {name}", error, error < TOL_ZERO, ideal="~0"))
    return all(checks)


def _report_other_directions(
    derivative: CEMDerivative,
    directions: dict[str, dolfinx.fem.Function],
    variations: dict[str, np.ndarray],
) -> bool:
    summaries = {
        name: _summarize_direction(derivative, eta, variations[name], EXTRA_EXPONENTS)
        for name, eta in directions.items()
        if name != SPHERES
    }
    report.section(
        f"a_k com as outras direções (k = {EXTRA_EXPONENTS[0]}..{EXTRA_EXPONENTS[-1]})"
    )
    report.table(
        ["η", "cos(W, -U)", "ordem", "menor a_k"],
        [
            [name, f"{s.alignment:.4f}", f"{s.order:.3f}", f"{s.smallest:.2e}"]
            for name, s in summaries.items()
        ],
    )
    report.section(f"Ordem (k = {EXTRA_ORDER_FIRST}..{EXTRA_ORDER_LAST})")
    checks = [
        report.check(
            f"η {name}", s.order, abs(s.order - 1) < TOL_ORDER, ideal="1", fmt=".3f"
        )
        for name, s in summaries.items()
    ]
    return all(checks)


def run(
    config: SetupConfig | None = None,
    cem: CEMConfig | None = None,
    inclusion: InclusionConfig | None = None,
    direction: DirectionConfig | None = None,
    force: bool = False,
    cache_dir: Path = CACHE_DIR,
    outputs: Path = OUTPUTS_DIR,
    all_directions: bool = False,
) -> bool:
    config = config or SetupConfig()
    cem = cem or CEMConfig()
    inclusion = inclusion or InclusionConfig()
    direction = direction or DirectionConfig()
    protocol = get_protocol(config.measurement.protocol)
    currents = protocol.currents(ElectrodeLayout(config))
    currents = currents[: config.measurement.experiments or EXPERIMENTS]

    _report_setup(cem, inclusion, direction, len(currents), protocol.scope)
    report.section("Malha")
    if force or not cache_path(config, cache_dir).exists():
        print("  Gerando a malha (só na primeira vez)...")
    data = CylinderWithElectrodes(config, cache_dir=cache_dir, force=force).get()
    report.mesh(data)

    gamma = sphere_conductivity(data.mesh, inclusion)
    started = perf_counter()
    derivative = CEMDerivative(CEMForward(data, gamma, cem), currents)
    preparation = perf_counter() - started

    directions = _directions(data.mesh, direction)
    variations = {name: derivative.directional(eta) for name, eta in directions.items()}
    eta, variational = directions[SPHERES], variations[SPHERES]

    report.section("Uma corrente: W = F'_I(γ)η pela equação (2.10), I = I(1)")
    report.matrix(variational[:1], row="I", column="e", fmt=".2e")
    total = float(np.abs(variational.sum(axis=1)).max())
    report.section("Soma de cada linha de F'(γ)η")
    checks = [report.check("Maior soma", total, total < TOL_ZERO, ideal="~0")]

    overall, per_current = _convergence_errors(derivative, eta, variational, STEPS)
    checks.append(_report_convergence(overall, per_current))
    report.info(
        "η das esferas",
        _alignment_note(_alignment(variational, derivative.voltages)),
    )

    started = perf_counter()
    jacobian = derivative.jacobian()
    elapsed = perf_counter() - started

    report.section(f"Jacobiana com {len(currents)} correntes")
    report.info("J", f"{jacobian.shape[0]} x {jacobian.shape[1]}  ((L·l) x M)")
    report.info("Tempo só de J", f"{elapsed:.2f} s")
    report.info("Tempo antes de J (w_j, S e U)", f"{preparation:.2f} s")
    report.section("F'(γ)η pela equação (2.10)  (linha i: corrente I(i))")
    report.matrix(variational, row="I", column="e", fmt=".2e")
    checks.append(_report_columns(derivative, jacobian, data.mesh))
    checks.append(_report_directions(jacobian, directions, variations))
    if all_directions:
        checks.append(_report_other_directions(derivative, directions, variations))
    ok = all(checks)

    by_jacobian = (jacobian @ eta.x.array).reshape(variational.shape)
    paths = [
        render_convergence(STEPS, per_current[:, 0], outputs / "step4_convergence.png"),
        render_convergence(
            STEPS,
            overall,
            outputs / "step4_convergence_all.png",
            per_current=per_current,
        ),
        render_matrix(
            [
                (variational, r"$F'(\gamma)\eta$"),
                (by_jacobian, r"$J\bar{\eta}$"),
                (by_jacobian - variational, r"$J\bar{\eta} - F'(\gamma)\eta$"),
            ],
            outputs / "step4_derivative.png",
        ),
    ]
    report.result(
        ok,
        "a derivada e a Jacobiana passam nos testes de consistência",
        "há verificações incorretas",
    )
    report.files(paths)
    return ok
