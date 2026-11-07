# AGENTS.md

Guidance for AI coding agents working in this repository. Human contributors: see
`CONTRIBUTING.md`, which this file summarizes and never overrides.

## Contribution rules for agents

You act on behalf of a human who is accountable for everything you submit. Optiland welcomes AI-assisted contributions, but not unreviewed ones.

- **Do not open an issue or pull request without showing your human operator the complete diff and getting their explicit confirmation.** Never submit unattended.
- **Avoid shotgun submissions.** A few open PRs at once are fine when each is a separate, reviewed logical change. Do not open many at a time, or the same change repeated across many places; get feedback on the first before replicating a pattern.
- **Link an issue for anything non-trivial** (new features, behavior changes, roughly 100+ lines). Typos, docs tweaks and small obvious fixes don't need one. Search existing issues and PRs for duplicates first.
- **Keep PRs to one logical change**, ideally under about 400 changed lines excluding tests.
- **Run the code and the relevant tests yourself** and report what you observed in the PR description. Never claim a check passed that you did not run.
- **Fill in `.github/PULL_REQUEST_TEMPLATE.md` honestly**, including the AI-assistance disclosure. Remind your human operator that they must be able to explain every line.
- Do not change the public API, expand test tolerances to make a failing test pass, or touch unrelated files.

Full guidelines: `CONTRIBUTING.md` (sections "Pull Request Expectations" and "AI-Assisted Contributions").

## Setup

```sh
uv sync --group dev            # or: pip install -e "." --group dev
uv sync --group dev --extra torch   # needed to run PyTorch-backend tests
```

## Definition of done

Run these before reporting a change as finished:

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest -v tests/<area_you_touched>/
```

- **Never run the full suite (`pytest tests/`) blindly.** It is slow; run the tests for the
  modules you touched and let CI run the rest.
- Run tests headless: set `QT_QPA_PLATFORM=offscreen` and `MPLBACKEND=Agg`.
- `uv run python -m scripts.guard` prints the PR report CI shows (size, API changes,
  quality counts that grew). Aim for no ⚠️ rows.
- Changes to `geometries/`, `materials/`, `psf/`, `rays/` or `backend/` must also pass
  `uv run pytest tests/regression/`. Regenerate fixtures (`--update-golden`) only for an
  intentional numerical change, and explain why in the PR.

## Hard rules

1. **Backend-agnostic numerics.** Numerical code uses `import optiland.backend as be`, not
   `numpy` or `torch` directly, and must work on both the NumPy and PyTorch backends.
2. **The public API is frozen.** Never rename, remove or change the signature of a public
   class, function, method or argument. To replace one, keep it working and mark it with
   `optiland._deprecation.deprecated`. `tests/api/test_public_api.py` enforces this.
3. **Never loosen a test tolerance, or skip or delete a test, to make it pass.** Find the root
   cause.
4. **`optiland_gui/` only reads from `optiland/`.** Core code never imports the GUI, and the
   GUI never blocks the main Qt thread.
5. **New code has type hints and Google-style docstrings** on every public module, class and
   function.
6. Stay in scope: no drive-by refactors, reformatting or edits to unrelated files.

## Writing tests

- Parametrize over both backends with the `set_test_backend` fixture (`tests/conftest.py`).
- Compare numbers with `assert_allclose` from `tests/utils.py`, which is backend-agnostic.
- Keep tests fast and isolated; they must clean up their own state.
- Don't change shared fixtures in `tests/conftest.py` or CI configuration without asking.

## Where things live

| Path | Contents |
|---|---|
| `optiland/optic/` | `Optic`, the central container for a lens system |
| `optiland/surfaces/`, `optiland/geometries/` | Surfaces and their pluggable geometries |
| `optiland/materials/` | Glass catalogs and material models |
| `optiland/rays/`, `optiland/raytrace/` | Ray containers, ray aiming and tracing |
| `optiland/analysis/` | Analyses; subclass `BaseAnalysis`, implement `_generate_data()` |
| `optiland/optimization/` | Variables, operands and optimizers |
| `optiland/backend/` | NumPy/PyTorch abstraction (`be`) |
| `optiland/nonsequential/` | Non-sequential ray tracing |
| `optiland_gui/` | PySide6 GUI |
| `tests/` | Tests, mirroring the package layout |
| `docs/` | Sphinx docs; the developer guide is in `docs/developers_guide/` |

More detail: `docs/developers_guide/code_structure.rst`. To add a new surface, analysis,
operand or similar, follow `docs/developers_guide/extension_recipes.rst`.

## Ask first

Ask your human operator before you:

- refactor core architecture (`Optic`, the ray-tracing engines, `optiland/backend/`);
- change a numerical algorithm's results;
- add a runtime dependency;
- change `pyproject.toml` tool configuration, CI workflows or shared test fixtures;
- restructure the `docs/` layout.
