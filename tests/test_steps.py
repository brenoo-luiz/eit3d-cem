import dataclasses
from pathlib import Path

import pytest

from cem3d import cli
from cem3d.cli import STEPS, build_parser, load_setup
from cem3d.config import ROOT_DIR, SetupConfig
from cem3d.steps import consistency, electrodes, forward

CONFIGS = sorted((ROOT_DIR / "configs").glob("*.toml"))


def test_parser_accepts_known_steps():
    args = build_parser().parse_args(["electrodes", "--force"])
    assert args.step == "electrodes" and args.force


def test_parser_rejects_unknown_step():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nothing"])


def test_every_step_is_registered():
    assert {"electrodes", "forward", "consistency"} <= set(STEPS)


def test_step_outputs_are_unique():
    outputs = [step.output for step in STEPS.values()]
    assert len(outputs) == len(set(outputs))


def test_main_saves_the_report(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setitem(STEPS, "electrodes", cli.Step("step1", lambda **_: True))
    assert cli.main(["electrodes"]) == 0
    assert (tmp_path / "step1.txt").exists()


def test_main_runs_the_setup_file_into_its_own_folder(monkeypatch, tmp_path):
    seen = {}

    def run(config, force, outputs):
        seen.update(config=config, outputs=outputs)
        return True

    monkeypatch.setattr(cli, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setitem(STEPS, "electrodes", cli.Step("step1", run))
    assert cli.main(["electrodes", "--config", str(CONFIGS[0])]) == 0
    assert seen["config"] == load_setup(CONFIGS[0])
    assert seen["outputs"] == tmp_path / CONFIGS[0].stem
    assert (tmp_path / CONFIGS[0].stem / "step1.txt").exists()


def test_empty_setup_file_gives_the_default(tmp_path):
    path = tmp_path / "empty.toml"
    path.write_text("")
    assert load_setup(path) == SetupConfig()


def test_setup_file_sets_every_row(tmp_path):
    path = tmp_path / "rows.toml"
    path.write_text(
        "[mesh]\naspect = 3.0\n"
        "[electrodes]\ncoverage = 0.4\naspect = 2.0\n"
        "top_margin = 0.3\nbottom_margin = 0.5\n"
        "[[electrodes.rows]]\ncount = 8\ntheta = 30.0\nclockwise = false\n"
        "[[electrodes.rows]]\ncount = 12\n"
    )
    config = load_setup(path)
    assert config.mesh.height == 3.0
    e = config.electrodes
    assert (e.coverage, e.aspect, e.top_margin, e.bottom_margin) == (0.4, 2, 0.3, 0.5)
    first, second = e.rows
    assert (first.count, first.theta, first.clockwise) == (8, 30.0, False)
    assert (second.count, second.clockwise) == (12, True)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: Path(p).stem)
def test_example_setups_are_valid(path):
    assert len(load_setup(path).electrodes.rows) > 1


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


def test_consistency_step_passes_all_checks(mesh_config, tmp_path, capsys):
    pytest.importorskip("dolfinx")
    ok = consistency.run(
        mesh_config, cache_dir=tmp_path / "cache", outputs=tmp_path / "out"
    )
    assert ok
    names = ("voltages", "currents", "flux", "flux_gradient", "boundary", "contact")
    for name in names:
        assert (tmp_path / "out" / f"step3_{name}.png").exists()
    assert "incorreto" not in capsys.readouterr().out


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda p: Path(p).stem)
def rows_case(request, tmp_path_factory):
    config = load_setup(request.param)
    coarse = dataclasses.replace(config.mesh, size_max=0.25, size_min=0.08)
    return dataclasses.replace(config, mesh=coarse), tmp_path_factory.mktemp("cache")


@pytest.mark.parametrize("step", [electrodes, forward, consistency])
def test_steps_pass_with_several_rows(step, rows_case, tmp_path, capsys):
    pytest.importorskip("dolfinx")
    config, cache = rows_case
    ok = step.run(config, cache_dir=cache, outputs=tmp_path / "out")
    assert ok
    assert "incorreto" not in capsys.readouterr().out
    if step is forward:
        for row in range(1, len(config.electrodes.rows) + 1):
            assert (tmp_path / "out" / f"step2_potential_row{row}.png").exists()
