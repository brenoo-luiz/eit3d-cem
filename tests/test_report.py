import numpy as np

from cem3d.utils import report


def test_check_prints_value_ideal_and_status(capsys):
    report.section("Verificações")
    assert report.check("Erro", 1e-5, True, ideal="~0") is True
    assert report.check("Erro", 0.5, False) is False
    out = capsys.readouterr().out
    assert "valor" in out and "ideal" in out and "situação" in out
    assert "correto" in out and "incorreto" in out
    assert "1.0e-05" in out


def test_columns_header_is_printed_once_per_section(capsys):
    report.section("A")
    report.check("x", 1.0, True)
    report.check("y", 1.0, True)
    assert capsys.readouterr().out.count("situação") == 1


def test_matrix_prints_labels_and_values(capsys):
    report.matrix(np.array([[0.5, -0.5], [0.25, -0.25]]), row="I", column="e")
    lines = capsys.readouterr().out.splitlines()
    assert "e1" in lines[0] and "e2" in lines[0]
    assert lines[1].split() == ["I1", "0.5000", "-0.5000"]
    assert lines[2].split() == ["I2", "0.2500", "-0.2500"]


def test_thousands_separator():
    assert report._thousands(116399) == "116.399"


def test_mesh_line(mesh_data, capsys):
    report.mesh(mesh_data)
    out = capsys.readouterr().out
    assert "elementos" in out and "vértices" in out


def test_saved_to_writes_terminal_and_file(tmp_path, capsys):
    path = tmp_path / "out" / "step.txt"
    with report.saved_to(path):
        report.title("TÍTULO")
        report.files([tmp_path / "fig.png"])
    out = capsys.readouterr().out
    assert path.read_text(encoding="utf-8") == out
    assert "TÍTULO" in out and str(path) in out


def test_files_without_saved_report(tmp_path, capsys):
    report.files([tmp_path / "fig.png"])
    assert ".txt" not in capsys.readouterr().out
