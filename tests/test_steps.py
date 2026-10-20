import pytest

from cem3d.cli import STEPS, build_parser
from cem3d.steps import electrodes, forward


def test_parser_accepts_known_steps():
    args = build_parser().parse_args(["electrodes", "--force"])
    assert args.step == "electrodes" and args.force


def test_parser_rejects_unknown_step():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nothing"])


def test_every_step_is_registered():
    assert {"electrodes", "forward"} <= set(STEPS)


def test_electrodes_step_passes_all_checks(mesh_config, tmp_path, capsys):
    pytest.importorskip("dolfinx")
    ok = electrodes.run(
        mesh_config, cache_dir=tmp_path / "cache", outputs=tmp_path / "out"
    )
    assert ok
    assert (tmp_path / "out" / "step1_electrodes.png").exists()
    assert "incorreto" not in capsys.readouterr().out


def test_forward_step_passes_all_checks(mesh_config, tmp_path, capsys):
    pytest.importorskip("dolfinx")
    ok = forward.run(
        mesh_config, cache_dir=tmp_path / "cache", outputs=tmp_path / "out"
    )
    assert ok
    assert (tmp_path / "out" / "step2_voltages.png").exists()
    assert (tmp_path / "out" / "step2_difference.png").exists()
    assert "incorreto" not in capsys.readouterr().out
