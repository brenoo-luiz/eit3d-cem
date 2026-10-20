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


def test_check_zero():
    assert report.check_zero("x", 1e-12)
    assert not report.check_zero("x", 1e-3)


def test_thousands_separator():
    assert report._thousands(116399) == "116.399"


def test_mesh_line(mesh_data, capsys):
    report.mesh(mesh_data)
    out = capsys.readouterr().out
    assert "elementos" in out and "vértices" in out
