import pytest

from cem3d import cli
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


def test_step_outputs_are_unique():
    outputs = [step.output for step in STEPS.values()]
    assert len(outputs) == len(set(outputs))


def test_main_saves_the_report(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setitem(STEPS, "electrodes", cli.Step("step1", lambda force: True))
    assert cli.main(["electrodes"]) == 0
    assert (tmp_path / "step1.txt").exists()


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
    assert (tmp_path / "out" / "step2_potential.png").exists()
    assert "incorreto" not in capsys.readouterr().out
