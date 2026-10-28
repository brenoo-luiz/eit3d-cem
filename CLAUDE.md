# CLAUDE.md

Guidance for Claude Code in this repository. Talk to the user in Portuguese; code, comments and commit messages are in English.

The user's personal standards in `~/.claude/CLAUDE.md` also apply. Topic guides live in `~/.claude/padroes/` (style, OOP, design patterns, project structure, data structures and algorithms, AI, testing, UML, git, figures, references): read the relevant one before a task on that topic. This file takes precedence on conflicts.

## Project

3D Electrical Impedance Tomography (EIT) with the Complete Electrode Model (CEM), an undergraduate research project (IC, UFSC) supervised by Prof. Fábio Margotti. It is the 3D version of the 2D code in HafemannE/FEIT_CBM34, and the consistency tests follow section 3.2.1 of the supervisor's book. Stack: gmsh (OCC) for the mesh, FEniCSx (dolfinx 0.11 + PETSc) for the solver.

Setup fixed by the supervisor (do not change without asking):

- Cylinder of radius 1 and height/radius 2; one row of L = 16 electrodes, 50% coverage, height/width 16/π (height 1), centered in z.
- First electrode at θ = 1 rad (57.2958°), numbered clockwise.
- Weak form: ∫γ∇u·∇v + Σ (1/z_j) ∫_{e_j} (u − U_j)(v − V_j) = Σ I_j V_j, with z_j = 1e-3, ΣU = 0, ΣI = 0.
- Adjacent current patterns I = (1, −1, 0, …), all 16 of them, measuring U on every electrode.

Electrode parameters asked by the supervisor (email of Oct 2026), with the choices made where the email was open:

- Cylinder height/radius (`MeshConfig.aspect`) and one electrode height/width (`ElectrodeConfig.aspect`, width = arc length).
- Rows (`ElectrodeConfig.rows`, top to bottom), each with count, coverage of the cross section, sense and θ (`RowConfig`).
- θ in degrees from the generatrix θ = 0 (x axis) to the leading edge of the first electrode, measured in the row sense; the electrode extends in that sense, so a clockwise row mirrors an anticlockwise one. This differs from the 2D code (FEIT_CBM34), where e1 always spans [rotate, rotate + w].
- Electrodes are identical within a row; rows with different widths have different heights. Vertical gaps between rows and to the top and bottom are all equal.
- Electrodes are numbered row after row; currents are adjacent within each row (provisional until the measurement protocols).

Next tasks, in order: measurement protocols (how currents are injected and voltages collected), then the derivative of the forward operator (book section 3.2.2), then the inverse problem.

## Environment and commands

```bash
conda env create -f environment.yml
conda activate eit3d-env
pip install -e .
git config core.hooksPath .githooks

python main.py electrodes [--force]
python main.py forward [--force]
python main.py consistency [--force]
python main.py electrodes --config configs/three_rows.toml   # outputs/three_rows/

pytest -q
pytest tests/test_cem.py::test_name
```

- `--force` regenerates the mesh instead of loading it from `cache/`. A step exits non-zero if any check fails.
- `--config` reads a TOML setup (`[mesh]`, `[electrodes]`, `[[electrodes.rows]]`, see `configs/`); missing keys keep the defaults.
- The pre-commit hook runs `pytest -q` and blocks the commit if a test fails.
- Tests that need FEniCSx use `pytest.importorskip("dolfinx")`. Reuse the session fixtures `mesh_config` and `mesh_data` from `tests/conftest.py` (coarse mesh, `size_max=0.25`, `size_min=0.08`) instead of meshing inside a test.

## Architecture

Pipeline: `config` → `geometry` (layout + mesh) → `fields` (γ) → `forward` (CEM solve and checks) → `plots`, run by `steps`.

- `cli.py`: `STEPS` maps a name to `Step(output, run)`; `main` saves everything printed to `outputs/<output>.txt` (or `outputs/<config name>/` with `--config`); `load_setup` reads the TOML setup. A new step is `steps/<name>.py` with `run(config=None, ..., force=False, cache_dir=CACHE_DIR, outputs=OUTPUTS_DIR) -> bool` plus an entry in `STEPS` (`stepN`).
- `config.py`: frozen dataclasses validated in `__post_init__`, and the gmsh tags: top 1, bottom 2, gap 3, volume 10, electrode j = `electrode_tag(j) = 100 + j`.
- `geometry/electrodes.py`: `ElectrodeLayout` is the only source of electrode positions: it stacks `ElectrodeRow`s with equal vertical gaps, indexes electrodes 1-based across rows, and gives `locate(theta, z)` and the `seam` angle.
- `geometry/mesh.py`: OCC cylinder fragmented by wedges, seam rotated to the gap center farthest from every row (staggered rows may share no gap, then the seam cuts an electrode), surfaces tagged by center of mass, refinement only on the true border of each electrode. Meshes are cached by a hash of the config; bump `_SCHEMA` whenever mesh generation changes.
- `forward/cem.py`: `CEMForward` solves A w_j = c_j once per electrode (CG + BoomerAMG) and builds the Schur complement; `solve(currents)` is a dense solve with ΣU = 0 through a Lagrange multiplier; `potential(U) = Σ U_j w_j`.
- `forward/checks.py`: checks of step 2 (currents, ΣU, reciprocity, antisymmetry and circulant structure for γ ≡ 1, which apply only to a single row, extremes on the driving electrodes).
- `forward/consistency.py`: `ConsistencyTests`, the four tests of book section 3.2.1, with the flux from the gradient (as in the book) and the consistent flux from the weak form.
- `utils/report.py`: report helpers (`title`, `section`, `info`, `check`, `matrix`, `result`, `files`). `check` prints `correto`/`incorreto`; step tests assert `incorreto` never appears.
- `plots/paper.py`: shared academic figure style (`STYLE`, `PALETTE`, `COLUMN`, `frame`, `save`).

## Code standards

- Object-oriented design and patterns where they help; keep complexity low and the structure professional.
- No alignment with extra spaces (`x = 1`, never `x    = 1`); ruff rules E, F, W, B, C4, UP, I with 88 columns; `from __future__ import annotations`, builtin generics, `strict=True` on `zip`.
- Comments are short and only where they add something; no long docstrings.
- No unused code: when replacing something, remove the old version and whatever became unused (the user checks with vulture).
- Every feature and every bug fix comes with a test, so that fixed errors cannot come back. Check that a new test fails on the old code.
- Step reports are minimal and professional: no emojis or symbols such as ✓/✗, only value, ideal and situation (`correto`/`incorreto`), no tolerance text. Every step saves its report to `outputs/stepN.txt`.
- New figures use `plots/paper.py`: serif LaTeX font, thin frame, light grid, legend in a row above, stacked panels with shared axes. Do not restyle existing figures unless asked.
- `cache/` and `outputs/` are generated and git-ignored.

## Workflow

- Never run `git commit` or `git push`. The user reviews the diff and commits from his own terminal.
- One logical change per commit. When a change is done, suggest the commit message: Conventional Commits without a scope (`feat: ...`, never `feat(step7): ...`), short and in English.
- Before suggesting a commit, run `ruff check src tests`, `ruff format src tests` and `pytest -q`, and report the result.
- Always edit whole files consistently; do not leave partial changes for the user to finish.
- When something differs from what the supervisor asked or from the book, say so explicitly instead of silently choosing.
