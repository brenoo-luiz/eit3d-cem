from __future__ import annotations

from pathlib import Path
from typing import Iterable

TOL_ZERO = 1e-10
LABEL = 36

_columns_pending = False


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def title(text: str) -> None:
    print(f"\n{text}\n")


def section(text: str) -> None:
    global _columns_pending
    _columns_pending = True
    print(f"\n{text}")


def info(label: str, value: str) -> None:
    print(f"  {label:<{LABEL}}{value}")


def check(label: str, value: float, ok: bool, ideal: str = "0", fmt: str = ".1e") -> bool:
    global _columns_pending
    if _columns_pending:
        print(f"  {'':<{LABEL}}{'valor':>10}{'ideal':>9}   situação")
        _columns_pending = False
    status = "correto" if ok else "incorreto"
    print(f"  {label:<{LABEL}}{value:>10{fmt}}{ideal:>9}   {status}")
    return ok


def check_zero(label: str, value: float, tol: float = TOL_ZERO) -> bool:
    return check(label, value, abs(value) < tol)


def mesh(data) -> None:
    cells = data.mesh.topology.index_map(3).size_global
    vertices = data.mesh.topology.index_map(0).size_global
    info("Malha", f"{_thousands(cells)} elementos, {_thousands(vertices)} vértices")


def result(ok: bool, text_ok: str, text_fail: str) -> None:
    print(f"\nResultado: {text_ok if ok else text_fail}")


def files(paths: Iterable[Path]) -> None:
    print("\nArquivos gerados:")
    for p in paths:
        print(f"  {Path(p)}")
