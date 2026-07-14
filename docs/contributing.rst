.. _contributing:

Contributing to Optiland
========================

Thank you for your interest in contributing to **Optiland**! Contributions are welcome in many forms, including but not limited to:

- Bug reports and feature requests.
- Code contributions and pull requests.
- Improvements to documentation and examples.

How to Contribute
-----------------

1. **Start with an issue.** Before beginning work, check whether there's already an open issue for the feature or bug you want to work on. If not, please open one first. This helps others know what's in progress and avoid duplicating effort.
2. **Let others know you're working on it.** If you'd like to work on an issue, leave a comment to say so. You can also request to be assigned, but this is optional — the goal is just to keep communication open.
3. **Fork** the repository on GitHub.
4. **Clone** your forked repository locally.
5. **Create** a new branch for your feature or bugfix.
6. **Commit** your changes with clear commit messages.
7. **Push** your changes to your fork.
8. **Open** a pull request with a detailed description of your changes.


Development Setup
------------------

1. Clone your fork and create a virtual environment (``.venv``) in the repository root.
2. Install the project with its dev dependency group using `uv <https://docs.astral.sh/uv/>`_::

       uv sync --group dev

   This installs Optiland in editable mode plus ``pytest``, ``ruff``, ``mypy``, and ``vulture``.
   Without ``uv``, ``pip install -e ".[dev]"`` (or installing the packages listed under
   ``[dependency-groups].dev`` in ``pyproject.toml`` manually) works too.
3. Run the test suite scoped to what you're changing — **never run the full suite blindly**::

       pytest -v tests/<area_you_touched>/

   The full suite (``pytest tests/``) is slow enough that it's rarely the right first check; let
   CI run it in full and iterate locally on the scoped subset.

Quality Gates
-------------

Every pull request runs the following in CI:

- **Ruff (lint + format):** blocking, including docstring checks on public classes/functions
  only — pre-existing undocumented internals elsewhere in a touched file never block a PR.
- **mypy:** blocking, but only for modules listed under ``[[tool.mypy.overrides]]`` in
  ``pyproject.toml``. Untouched, unlisted modules are never checked. If you refactor a file onto
  the allowlist, add it to the override list in the same PR.
- **Golden-value regression snapshots** (``tests/regression/``): required for changes touching
  ``geometries/``, ``materials/``, ``psf/``, ``rays/``, or ``backend/``. Run locally with
  ``pytest tests/regression/``; regenerate deliberately with ``--update-golden`` only after an
  intentional numerical change, and explain why in the PR description.
- **vulture (dead-code scan):** dev-only, not part of CI — see "Dead-Code Audits" in
  ``CONTRIBUTING.md``.

For the full plugin-package mechanism (shipping a new surface geometry, material catalog, or
analysis as an installable package rather than editing Optiland's source) see
:doc:`developers_guide/plugin_packages`.

Task Workflow and Coordination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use a lightweight workflow to help contributors collaborate smoothly and avoid duplicated effort:

- **Each task should have an issue** on GitHub. If you're working on something new, check for an existing issue or `open a new one <https://github.com/optiland/optiland/issues>`_. This keeps the project transparent and easier to coordinate.
- **Leave a comment on the issue** if you plan to work on it. Optionally, you can be assigned the issue to make your involvement visible.
- **Progress is tracked using GitHub Projects (kanban).** Issues are moved across columns such as "To Do", "In Progress", and "In Review". You can view the board here: `Optiland GitHub Project <https://github.com/users/HarrisonKramer/projects/1>`_. If you don’t have permission to move cards, maintainers will update them based on issue comments.
- **Milestones help plan releases.** Larger features and grouped improvements may be linked to a milestone. If you're contributing to a milestone, try to complete the work in time — but there's no strict deadline.

If you start working on something but can't finish it, just leave a note in the issue. That way, someone else can take over if needed.

Guidelines
----------

- **Coding Style:** Follow the project's style guidelines. We use automated tools (e.g., `Ruff <https://docs.astral.sh/ruff/>`_) to enforce code formatting and linting.
- **Testing:** Write tests for new features and bug fixes. Ensure all tests pass before submitting a pull request.
- **Documentation:** Update documentation and examples as necessary.
- **Commit Messages:** Use clear and descriptive commit messages.

Code Style Guidelines
---------------------

Please adhere to the following guidelines when contributing.

General Style Rules
~~~~~~~~~~~~~~~~~~~

- Follow `PEP 8 <https://peps.python.org/pep-0008/>`_ for code style.
- Use meaningful variable names that clearly describe their purpose.

Formatting and Linting
~~~~~~~~~~~~~~~~~~~~~~

We use `Ruff <https://docs.astral.sh/ruff/>`_ for both linting and formatting. Formatting and linting are **automatically enforced** in pull requests through a GitHub Action and must pass before merging.

To ensure compliance before committing, install `pre-commit <https://pre-commit.com/>`_ and set up the hook::

    pip install pre-commit
    pre-commit install

This will manually install the pre-commit hooks from the ``.pre-commit-config.yaml`` file in your local Optiland repository.
The pre-commit hooks will automatically run Ruff checks on staged files before committing.

To manually run Ruff checks before committing, use::

    pre-commit run --all-files

Ruff can be used to automatically apply fixes for formatting and linting issues where possible. To do this, first install Ruff::

    pip install ruff

Then, you can run Ruff to automatically fix issues in your code::

    ruff format .

Key Formatting Rules:

- Keep line length to a maximum of **88 characters**.
- Use spaces instead of tabs for indentation.
- Organize imports as follows:

  1. Standard library imports
  2. Third-party library imports
  3. Local module imports

  Example::

      import os
      import numpy as np
      from optiland.analysis import SpotDiagram

Docstrings and Comments
~~~~~~~~~~~~~~~~~~~~~~~~

- Write docstrings for all public functions, classes, and modules using the `Google docstring style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html/>`_.
- Use inline comments sparingly and only when necessary to explain complex logic.

Testing
-------

- Write tests for new features or bug fixes in the ``tests/`` directory.
- Use **pytest** for running tests::

      pytest

- Run tests with coverage before submitting a PR::

      pytest --cov=optiland --cov-report=xml

Reporting Issues
----------------

If you encounter any bugs or issues, please report them on our GitHub issue tracker. Include detailed steps to reproduce the issue, along with any relevant logs or error messages.

Thank you for contributing to Optiland!
