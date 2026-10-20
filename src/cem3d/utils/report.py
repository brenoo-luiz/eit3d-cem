from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

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


def check(
    label: str, value: float, ok: bool, ideal: str = "0", fmt: str = ".1e"
) -> bool:
    global _columns_pending
    if _columns_pending:
        print(f"  {'':<{LABEL}}{'valor':>10}{'ideal':>9}   situação")
        _columns_pending = False
    status = "correto" if ok else "incorreto"
    print(f"  {label:<{LABEL}}{value:>10{fmt}}{ideal:>9}   {status}")
    return ok


def matrix(values: np.ndarray, row: str, column: str, fmt: str = ".4f") -> None:
    width = max(len(f"{v:{fmt}}") for v in values.ravel()) + 2
    head = "".join(f"{f'{column}{j}':>{width}}" for j in range(1, values.shape[1] + 1))
    print(f"  {'':<6}{head}")
    for i, line in enumerate(values, start=1):
        cells = "".join(f"{v:>{width}{fmt}}" for v in line)
        print(f"  {f'{row}{i}':<6}{cells}")


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
