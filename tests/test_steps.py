import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from cem3d import cli
from cem3d.cli import STEPS, build_parser, load_setup
from cem3d.config import ROOT_DIR, MeasurementConfig, RowConfig, SetupConfig
from cem3d.forward.derivative import CEMDerivative
from cem3d.geometry.electrodes import ElectrodeLayout
from cem3d.steps import consistency, derivative, electrodes, forward
from cem3d.steps.electrodes import _first_start

CONFIGS = sorted((ROOT_DIR / "configs").glob("*.toml"))


def test_parser_accepts_known_steps():
    args = build_parser().parse_args(["electrodes", "--force"])
    assert args.step == "electrodes" and args.force


def test_parser_rejects_unknown_step():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nothing"])


def test_every_step_is_registered():
    assert {"electrodes", "forward", "consistency", "derivative"} <= set(STEPS)


def test_step_outputs_are_unique():
    outputs = [step.output for step in STEPS.values()]
    assert len(outputs) == len(set(outputs))


def test_parser_leaves_the_other_directions_off_by_default():
    assert not build_parser().parse_args(["derivative"]).all_directions


def test_main_passes_the_option_only_when_it_is_on(monkeypatch, tmp_path):
    seen = []

    def run(**options):
        seen.append(options.get("all_directions", False))
        return True

    monkeypatch.setattr(cli, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setitem(STEPS, "derivative", cli.Step("step4", run))
    assert cli.main(["derivative"]) == 0
    assert cli.main(["derivative", "--all-directions"]) == 0
    assert seen == [False, True]


@pytest.mark.parametrize("step", ["electrodes", "forward", "consistency"])
def test_main_rejects_the_option_in_the_other_steps(step, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "OUTPUTS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        cli.main([step, "--all-directions"])


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


def test_setup_file_sets_the_measurement(tmp_path):
    path = tmp_path / "measurement.toml"
    path.write_text('[measurement]\nprotocol = "adjacent_chain"\nexperiments = 4\n')
    assert load_setup(path).measurement == MeasurementConfig("adjacent_chain", 4)


@pytest.mark.parametrize(
    "text",
    [
        "experiments = 16\n",
        "[measuremnt]\nexperiments = 4\n",
        '[measurement]\nprotocol = "nothing"\n',
        "[measurement]\nprotocol = 3\n",
        "[measurement]\nexperiments = 0\n",
        "[measurement]\nexperiments = 17\n",
        '[measurement]\nexperiments = "16"\n',
        "[measurement]\nexperiments = 16.0\n",
        "[measurement]\ncurrents = 16\n",
    ],
    ids=[
        "old top-level key",
        "misspelled section",
        "unknown protocol",
        "protocol not a name",
        "zero currents",
        "more currents than electrodes",
        "currents as text",
        "currents as float",
        "unknown key",
    ],
)
def test_invalid_measurement_in_the_setup_file_is_refused(text, tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text(text)
    with pytest.raises((ValueError, TypeError)):
        load_setup(path)


def test_two_rows_of_eight_setup_has_equal_rows_and_all_the_currents():
    config = load_setup(ROOT_DIR / "configs" / "two_rows_8.toml")
    first, second = config.electrodes.rows
    assert first == second
    assert first.count == 8
    assert config.measurement == MeasurementConfig("adjacent_by_row", 16)
    layout = ElectrodeLayout(config)
    assert layout.row_gap == pytest.approx(0.4)
    assert layout[1].area(1.0) == pytest.approx(math.pi / 16)
    assert layout[1].z_center == pytest.approx(-layout[9].z_center)


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


def test_first_start_at_zero_reads_zero_not_360():
    config = dataclasses.replace(
        SetupConfig(),
        electrodes=dataclasses.replace(
            SetupConfig().electrodes, rows=(RowConfig(theta=0.0),)
        ),
    )
    row = ElectrodeLayout(config).rows[0]
    centroid = next(iter(row)).theta_center + 1e-5
    assert _first_start(row, centroid) == 0.0


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


def test_derivative_step_passes_all_checks(mesh_config, tmp_path, capsys):
    pytest.importorskip("dolfinx")
    ok = derivative.run(
        mesh_config, cache_dir=tmp_path / "cache", outputs=tmp_path / "out"
    )
    assert ok
    for name in ("convergence", "convergence_all", "derivative"):
        assert (tmp_path / "out" / f"step4_{name}.png").exists()
    out = capsys.readouterr().out
    assert "incorreto" not in out
    assert "I(1) = (1, -1, 0, ..., 0)" in out and "8 correntes" in out
    assert "Tempo só de J" in out
    assert "T_k" in out
    for name in ("esferas", "constante", "linear em z", "aleatória"):
        assert f"η {name}" in out


def test_derivative_step_uses_all_the_currents_of_the_two_rows(tmp_path, capsys):
    pytest.importorskip("dolfinx")
    config = load_setup(ROOT_DIR / "configs" / "two_rows_8.toml")
    coarse = dataclasses.replace(config.mesh, size_max=0.25, size_min=0.08)
    ok = derivative.run(
        dataclasses.replace(config, mesh=coarse),
        cache_dir=tmp_path / "cache",
        outputs=tmp_path / "out",
    )
    assert ok
    out = capsys.readouterr().out
    assert "Jacobiana com 16 correntes" in out
    assert "256 x" in out


@pytest.mark.parametrize("all_directions", [False, True])
def test_other_directions_run_only_with_the_option(
    all_directions, mesh_config, monkeypatch, tmp_path, capsys
):
    pytest.importorskip("dolfinx")
    steps = []

    def quotient(self, eta, step):
        steps.append(step)
        return np.ones_like(self.voltages)

    monkeypatch.setattr(CEMDerivative, "difference_quotient", quotient)
    derivative.run(
        mesh_config,
        cache_dir=tmp_path / "cache",
        outputs=tmp_path / "out",
        all_directions=all_directions,
    )
    out = capsys.readouterr().out
    extras = 3 * [0.5**k for k in range(4, 31)]
    assert len(steps) == 50 + (len(extras) if all_directions else 0)
    assert ("a_k com as outras direções" in out) == all_directions
    assert ("cos(W, -U) = " in out) and ("a_k ≈ t/(1 + t)" in out)
    if all_directions:
        assert steps[50:] == extras


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda p: Path(p).stem)
def rows_case(request, tmp_path_factory):
    config = load_setup(request.param)
    coarse = dataclasses.replace(config.mesh, size_max=0.25, size_min=0.08)
    return dataclasses.replace(config, mesh=coarse), tmp_path_factory.mktemp("cache")


@pytest.mark.parametrize("step", [forward, consistency, derivative])
def test_steps_are_ready_for_the_chain_of_currents(step, tmp_path, capsys):
    pytest.importorskip("dolfinx")
    config = load_setup(ROOT_DIR / "configs" / "two_rows_8.toml")
    chain = dataclasses.replace(
        config,
        mesh=dataclasses.replace(config.mesh, size_max=0.25, size_min=0.08),
        measurement=MeasurementConfig("adjacent_chain", 16),
    )
    assert step.run(chain, cache_dir=tmp_path / "cache", outputs=tmp_path / "out")
    out = capsys.readouterr().out
    assert "incorreto" not in out
    assert "uma cadeia" in out


@pytest.mark.parametrize("step", [electrodes, forward, consistency, derivative])
def test_steps_pass_with_several_rows(step, rows_case, tmp_path, capsys):
    pytest.importorskip("dolfinx")
    config, cache = rows_case
    ok = step.run(config, cache_dir=cache, outputs=tmp_path / "out")
    assert ok
    assert "incorreto" not in capsys.readouterr().out
    if step is forward:
        for row in range(1, len(config.electrodes.rows) + 1):
            assert (tmp_path / "out" / f"step2_potential_row{row}.png").exists()
