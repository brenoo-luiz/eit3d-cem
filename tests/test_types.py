import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "cem3d"


@pytest.fixture(scope="module")
def type_errors() -> list[dict]:
    pytest.importorskip("pyright")
    command = [sys.executable, "-m", "pyright", "--outputjson"]
    command += ["--pythonpath", sys.executable, "src"]
    # pyright exits non-zero when it finds errors, which the tests report themselves
    result = subprocess.run(
        command, check=False, cwd=ROOT, capture_output=True, text=True
    )
    report = json.loads(result.stdout)
    return [d for d in report["generalDiagnostics"] if d["severity"] == "error"]


def describe(error: dict) -> str:
    module = Path(error["file"]).relative_to(PACKAGE)
    return f"{module}:{error['range']['start']['line'] + 1}: {error['message']}"


def errors_in(type_errors: list[dict], module: str, needle: str = "") -> list[str]:
    path = str(PACKAGE / module)
    return [
        describe(d) for d in type_errors if d["file"] == path and needle in d["message"]
    ]


def test_package_has_no_type_errors(type_errors):
    assert [describe(d) for d in type_errors] == []


@pytest.mark.parametrize("module", ["config.py", "geometry/electrodes.py"])
def test_margins_are_never_none(type_errors, module):
    assert errors_in(type_errors, module) == []


def test_mesh_facet_tags_are_never_none(type_errors):
    assert errors_in(type_errors, "geometry/mesh.py", "MeshTags | None") == []


def test_row_colors_come_from_a_color_sequence(type_errors):
    assert errors_in(type_errors, "plots/consistency.py", '"colors"') == []


@pytest.mark.parametrize("module", ["plots/consistency.py", "plots/forward.py"])
def test_style_is_accepted_by_rc_context(type_errors, module):
    assert errors_in(type_errors, module, 'parameter "rc"') == []


def test_axes_rectangles_are_tuples(type_errors):
    assert errors_in(type_errors, "plots/electrodes.py", '"add_axes"') == []
