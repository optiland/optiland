# Contributing to Optiland

> For a deep dive into Optiland's architecture and step-by-step extension recipes, see the [Developer's Guide](https://optiland.readthedocs.io/en/latest/developers_guide/introduction.html) on Read the Docs.

Thank you for your interest in contributing to **Optiland**! Contributions are welcome in many forms, including but not limited to:

- Bug reports and feature requests.
- Code contributions and pull requests.
- Improvements to documentation and examples.

## How to Contribute

1. **Start with an issue.** Before beginning work, check whether there's already an open issue for the feature or bug you want to work on. If not, [open one](https://github.com/optiland/optiland/issues). This helps others know what's in progress and avoids duplicating effort.
2. **Let others know you're working on it.** If you'd like to work on an issue, leave a comment to say so. You can also ask to be assigned — this is optional, but helps us track who’s working on what.
3. **Fork** the repository on GitHub.
4. **Clone** your forked repository locally.
5. **Create** a new branch for your feature or bugfix.
6. **Commit** your changes with clear commit messages.
7. **Push** your changes to your fork.
8. **Open** a pull request with a detailed description of your changes.


## Development Setup

1. Clone your fork and create a virtual environment (`.venv`) in the repository root.
2. Install the project with its dev dependency group using [`uv`](https://docs.astral.sh/uv/):

   ```sh
   uv sync --group dev
   ```

   This installs Optiland in editable mode plus `pytest`, `mypy`, and `vulture`. If you
   don't use `uv`, `pip install -e "." --group dev` (or installing the packages listed under
   `[dependency-groups].dev` in `pyproject.toml` manually) works too.
3. Run the test suite scoped to what you're changing — **never run the full suite blindly**:

   ```sh
   .venv/Scripts/python.exe -m pytest -v tests/<area_you_touched>/
   ```

   The full suite (`pytest tests/`) is slow enough that it's rarely the right first check; let
   CI run it in full and iterate locally on the scoped subset.

## Quality Gates

Every pull request runs the following in CI. Understanding what each one does (and doesn't) do
helps you land a first PR without surprises:

- **Ruff (lint + format):** blocking. `ruff check optiland/` and `ruff format optiland/` must
  pass. This includes docstring checks (`D1xx`) on public classes/functions — but only on public,
  non-underscore-prefixed names, so a new contributor's PR is never blocked by pre-existing
  undocumented internals elsewhere in the file.
- **mypy:** blocking, but only for modules explicitly listed under
  `[[tool.mypy.overrides]]` in `pyproject.toml`. Files outside that allowlist are not
  type-checked in CI, so touching an unrelated file never introduces a new type-checking
  obligation. If you refactor a file onto the allowlist, add it to the override list in the
  same PR.
- **Golden-value regression snapshots** (`tests/regression/`): required for any change touching
  `geometries/`, `materials/`, `psf/`, `rays/`, or `backend/`. This suite traces a fixed set of
  representative optical systems at both backends and compares ray intercepts, OPD, spot
  centroids, and PSF metrics against committed fixtures in `tests/regression/fixtures/`. Run it
  locally with `pytest tests/regression/`. If you made an intentional numerical change, regenerate
  the fixtures deliberately with `pytest tests/regression/ --update-golden` and explain why in
  the PR description — never regenerate to silence an unexplained diff.
- **`vulture` (dead-code scan):** dev-only, **not** part of CI. See "Dead-Code Audits" below.

## Task Workflow and Coordination

We use a lightweight workflow to help contributors collaborate smoothly and avoid duplicated effort:

- **Each task should have an issue** on GitHub. If you're working on something new, check for an existing issue or [open a new one](https://github.com/optiland/optiland/issues). This keeps the project transparent and easier to coordinate.
- **Leave a comment on the issue** if you plan to work on it. Optionally, you can be assigned the issue to make your involvement visible.
- **Progress is tracked using GitHub Projects (kanban).** You can view the board [here](https://github.com/users/HarrisonKramer/projects/1). Issues move between columns like “To Do,” “In Progress,” and “In Review.” If you can’t move cards directly, that’s okay — maintainers will update them based on issue comments.
- **Milestones help us plan releases.** Larger features and grouped improvements may be linked to a milestone. If you're contributing to one of these, try to finish within the milestone timeframe — but there's no hard deadline.

If you’ve started something but run into delays or need to step away, just leave a quick note in the issue so others can jump in if needed.

## Guidelines

- **Coding Style:** Follow the project's style guidelines. We use automated tools like [`Ruff`](https://docs.astral.sh/ruff/) to enforce code formatting and linting.
- **Testing:** Write tests for new features and bug fixes. Ensure all tests pass before submitting a pull request.
- **Documentation:** Update documentation and examples as necessary.
- **Commit Messages:** Use clear and descriptive commit messages.

## Code Style Guidelines

Please adhere to the following guidelines when contributing.

### General Style Rules

- Follow [`PEP 8`](https://peps.python.org/pep-0008/) for code style.
- Use meaningful variable names that clearly describe their purpose.

### Formatting and Linting

We use [`Ruff`](https://docs.astral.sh/ruff/) for both linting and formatting. Formatting and linting are **automatically enforced** in pull requests through a GitHub Action and must pass before merging.

To ensure compliance before committing, install [`pre-commit`](https://pre-commit.com/) and set up the hook:

```sh
pip install pre-commit
pre-commit install
```

This will manually install the pre-commit hooks from the ``.pre-commit-config.yaml`` file in your local Optiland repository. The pre-commit hooks will automatically run Ruff checks on staged files before committing.

To manually run Ruff checks before committing, use:

```sh
pre-commit run --all-files
```

Ruff can be used to automatically apply fixes for formatting and linting issues where possible. To do this, first install Ruff:

```sh
pip install ruff
```

Then, you can run Ruff to automatically fix issues in your code:

```sh
ruff format .
```

#### Key Formatting Rules:

- Keep line length to a maximum of 88 characters.
- Use spaces instead of tabs for indentation.
- Organize imports as follows:
    1. Standard library imports
    2. Third-party library imports
    3. Local module imports

Example:

```python
import os
import numpy as np
from optiland.analysis import SpotDiagram
```

#### Docstrings and Comments

Write docstrings for all public functions, classes, and modules using the [Google docstring style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

Use inline comments sparingly and only when necessary to explain complex logic.

## Testing

- Write tests for new features or bug fixes in the tests/ directory.
- Use pytest for running tests:

```sh
pytest
```

Note: Coverage reporting is automatically handled by the CI pipeline when you submit a pull request.

## Writing a Plugin

Third-party packages can register a new surface geometry, material catalog,
or analysis without editing Optiland's source, via Python entry points. A
plugin package declares one of the `optiland.surfaces`, `optiland.materials`,
or `optiland.analyses` groups in its own `pyproject.toml`, pointing at a
zero-argument callable that registers itself against the relevant Optiland
registry. Optiland discovers and calls it lazily, on first use of the
corresponding factory (see `optiland/plugins.py`).

Example: a trivial custom surface geometry shipped in a separate package
`optiland-my-surface`:

```python
# my_surface_plugin/register.py
from dataclasses import dataclass

from optiland.surfaces.factories.geometry_factory import GeometryFactory


@dataclass
class MyConfig:
    surface_type = "my_surface"
    radius: float = float("inf")


def _create_my_surface(cs, config):
    from optiland.geometries import StandardGeometry

    return StandardGeometry(cs, radius=config.radius)  # trivial passthrough


def register() -> None:
    GeometryFactory.register("my_surface", _create_my_surface, MyConfig)
```

```toml
# my_surface_plugin's pyproject.toml
[project.entry-points."optiland.surfaces"]
my_surface = "my_surface_plugin.register:register"
```

Once `optiland-my-surface` is installed alongside Optiland,
`optic.surfaces.add(surface_type="my_surface", ...)` works with no changes
to Optiland itself.

The same mechanism works for material catalogs (`optiland.materials`) and analyses
(`optiland.analyses`) — see the [Plugin Packages](https://optiland.readthedocs.io/en/latest/developers_guide/plugin_packages.html)
guide for full worked examples of all three.

## Dead-Code Audits

If you suspect a file or function is unused, don't remove it on a hunch --
dynamic dispatch (factories, plugin registries) makes static analysis prone
to false positives. Use [`vulture`](https://github.com/jendrikseipp/vulture)
as a starting point, then verify manually:

```sh
pip install vulture
vulture optiland/ --min-confidence 80
```

For each hit, cross-reference `git log -1 --format=%ad -- <file>` --
recently-touched code flagged by vulture is more likely a false positive
(e.g. a factory-registered class vulture can't see being called) than
genuinely dead code. Prioritize old, unreferenced hits. Include this
evidence (last-touched date, confirmed absence of live references) in the
PR description of any dead-code removal. `vulture` is a dev-only tool and
is not part of the CI gate.

## Reporting Issues

If you encounter any bugs or issues, please report them on our GitHub issue tracker. Include detailed steps to reproduce the issue, along with any relevant logs or error messages.

Thank you for contributing to Optiland!
