# Contributing to Optiland

> For a deep dive into Optiland's architecture and step-by-step extension recipes, see the [Developer's Guide](https://www.optiland.org/docs/developers_guide/introduction.html).

Thank you for your interest in contributing to **Optiland**! Contributions are welcome in many forms, including but not limited to:

- Bug reports and feature requests.
- Code contributions and pull requests.
- Improvements to documentation and examples.

## How to Contribute

1. **Start with an issue for anything non-trivial.** For new features, behavior changes, or anything larger than roughly 100 lines, check whether there's already an open issue. If not, [open one](https://github.com/optiland/optiland/issues) so scope can be agreed before you write code. This also helps others know what's in progress and avoids duplicating effort. Typo fixes, documentation tweaks, and small obvious bug fixes can go straight to a pull request.
2. **Ask to be assigned before you start.** Comment on the issue to say you'd like to work on it, and wait until a maintainer assigns it to you before opening a pull request. Pull requests for issues that are unassigned, or assigned to someone else, may be closed. This keeps two people from solving the same issue and gives us a chance to point you at the right approach early.
3. **Fork** the repository on GitHub.
4. **Clone** your forked repository locally.
5. **Create** a new branch for your feature or bugfix.
6. **Commit** your changes with clear commit messages.
7. **Push** your changes to your fork.
8. **Open** a pull request with a detailed description of your changes.

### Finding Something to Work On

Issues labeled [`good first issue`](https://github.com/optiland/optiland/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) are well scoped and a good way to learn the codebase. Issues labeled [`help wanted`](https://github.com/optiland/optiland/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) are larger or need more domain knowledge. Each one states its scope, acceptance criteria, and what is out of scope; please stay within them. If something in an issue is unclear, ask in the issue before writing code.

To find modules that need more tests, browse the coverage report on [Codecov](https://app.codecov.io/gh/optiland/optiland).

## Pull Request Expectations

Maintainer review time is our scarcest resource, so please make your pull request easy to review:

- **One logical change per PR, kept small.** Aim for under roughly 400 changed lines, not counting tests. Large changes are easier to review, and to revert, as a series of independently mergeable PRs.
- **Agree scope first for larger work.** For new features or anything sizeable, describe your plan in an issue or discussion before opening PRs. If you have a large body of work, a tracking issue with a proposed split is the best way to get it reviewed.
- **Avoid opening many PRs at once.** A few open PRs at a time is plenty. When many are open we review them oldest first, and may ask you to hold off on new ones until earlier ones land.
- **Fill in the pull request template.** In particular, say in your own words what changed, why, and how you verified it.

## AI-Assisted Contributions

AI tools are welcome here. The standard is the same as for any contribution, and a human is accountable for every line:

- **Read and understand every line** before you submit. You should be able to explain and defend any change during review.
- **Run it yourself.** Passing CI is not verification. Tell us what you ran and what you observed.
- **Write the description and answer review comments yourself,** in your own words. Please don't paste unedited model output.
- **Disclose substantial AI assistance** using the checkbox in the pull request template. Disclosure is never held against you.
- **Agents act for a person.** If an autonomous agent opens a PR, the human operator must have reviewed the whole diff first. Unattended bulk submissions will be closed.
- **Give your agent the project context.** [`agents.md`](agents.md) describes the conventions coding agents should follow in this repository. Point your tool at it, together with the issue you are working on.

PRs that appear unreviewed (for example unrelated edits, invented APIs, tests that assert nothing, or a boilerplate description) may be closed without a detailed review. You're welcome to resubmit once these are addressed.

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

The [weekly dependency compatibility guide](docs/development/weekly-compatibility.md)
explains the Friday matrix, how to update version combinations in one file, and
how to reproduce or investigate dependency-specific failures.

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
- **Leave a comment on the issue** if you plan to work on it, and wait to be assigned before opening a pull request.
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

## Building the Documentation

The documentation lives in `docs/` as reStructuredText pages, Jupyter notebooks
(committed **with** their outputs; builds never execute them) and API pages
generated from docstrings. It is published automatically at
<https://www.optiland.org/docs/> by the `Docs` workflow on every push to
`master`, so a documentation change needs nothing beyond a normal pull request.

Build it locally with the same pinned environment the hosted build uses:

```sh
# one-time: create/update the environment (needs micromamba or conda)
micromamba create -f docs/build-environment.yml
micromamba activate sphinx-build

# normal local build -> docs/_build/html/index.html
make -C docs html

# the exact build + validation CI runs
python scripts/docs/build_docs.py
python scripts/docs/validate_build.py docs/_build/html \
    --warnings docs/_build/warnings.log \
    --allowlist scripts/docs/warnings-allowlist.txt
```

Notes:

- `pip install -r docs/requirements.txt` mirrors the Python pins for pip/uv
  users; `pandoc` must then be installed separately.
- The "Try Optiland in your browser" page is built by `jupyterlite-xeus` with
  `micromamba`. On hosts where that is not possible (for example Windows), pass
  `--skip-jupyterlite` to `build_docs.py`; CI always builds it.
- CI treats every Sphinx warning as an error unless it is listed, with a
  reason, in `scripts/docs/warnings-allowlist.txt`. Fix the warning at the
  source rather than extending the allowlist.
- Never commit generated HTML. Never hardcode the version: it is derived from
  the package metadata.

## Testing

- Write tests for new features or bug fixes in the tests/ directory.
- Use pytest for running tests:

```sh
pytest
```

Note: Coverage reporting is automatically handled by the CI pipeline when you submit a pull request. Results are published on [Codecov](https://app.codecov.io/gh/optiland/optiland).

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
(`optiland.analyses`) — see the [Plugin Packages](https://www.optiland.org/docs/developers_guide/plugin_packages.html)
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
