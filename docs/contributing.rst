.. _contributing:

Contributing to Optiland
========================

Thank you for your interest in contributing to **Optiland**! Contributions are welcome in many forms, including but not limited to:

- Bug reports and feature requests.
- Code contributions and pull requests.
- Improvements to documentation and examples.

How to Contribute
-----------------

1. **Start with an issue for anything non-trivial.** For new features, behavior changes, or anything larger than roughly 100 lines, check whether there's already an open issue. If not, please open one first so scope can be agreed before you write code. This also helps others know what's in progress and avoids duplicating effort. Typo fixes, documentation tweaks, and small obvious bug fixes can go straight to a pull request.
2. **Ask to be assigned before you start.** Comment on the issue to say you'd like to work on it, and wait until a maintainer assigns it to you before opening a pull request. Pull requests for issues that are unassigned, or assigned to someone else, may be closed. This keeps two people from solving the same issue and gives us a chance to point you at the right approach early.
3. **Fork** the repository on GitHub.
4. **Clone** your forked repository locally.
5. **Create** a new branch for your feature or bugfix.
6. **Commit** your changes with clear commit messages.
7. **Push** your changes to your fork.
8. **Open** a pull request with a detailed description of your changes.

Finding Something to Work On
----------------------------

Issues labeled `good first issue <https://github.com/optiland/optiland/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_ are well scoped and a good way to learn the codebase. Issues labeled `help wanted <https://github.com/optiland/optiland/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22>`_ are larger or need more domain knowledge. Each one states its scope, acceptance criteria, and what is out of scope; please stay within them. If something in an issue is unclear, ask in the issue before writing code.

To find modules that need more tests, browse the coverage report on `Codecov <https://app.codecov.io/gh/optiland/optiland>`_.


Pull Request Expectations
-------------------------

Maintainer review time is our scarcest resource, so please make your pull request easy to review:

- **One logical change per PR, kept small.** Aim for under roughly 400 changed lines, not counting tests. Large changes are easier to review, and to revert, as a series of independently mergeable PRs.
- **Agree scope first for larger work.** For new features or anything sizeable, describe your plan in an issue or discussion before opening PRs. If you have a large body of work, a tracking issue with a proposed split is the best way to get it reviewed.
- **Avoid opening many PRs at once.** A few open PRs at a time is plenty. When many are open we review them oldest first, and may ask you to hold off on new ones until earlier ones land.
- **Fill in the pull request template.** In particular, say in your own words what changed, why, and how you verified it.

AI-Assisted Contributions
-------------------------

AI tools are welcome here. The standard is the same as for any contribution, and a human is accountable for every line:

- **Read and understand every line** before you submit. You should be able to explain and defend any change during review.
- **Run it yourself.** Passing CI is not verification. Tell us what you ran and what you observed.
- **Write the description and answer review comments yourself,** in your own words. Please don't paste unedited model output.
- **Disclose substantial AI assistance** using the checkbox in the pull request template. Disclosure is never held against you.
- **Agents act for a person.** If an autonomous agent opens a PR, the human operator must have reviewed the whole diff first. Unattended bulk submissions will be closed.

- **Give your agent the project context.** ``agents.md`` in the repository root describes the conventions coding agents should follow. Point your tool at it, together with the issue you are working on.

PRs that appear unreviewed (for example unrelated edits, invented APIs, tests that assert nothing, or a boilerplate description) may be closed without a detailed review. You're welcome to resubmit once these are addressed.


Development Setup
------------------

1. Clone your fork and create a virtual environment (``.venv``) in the repository root.
2. Install the project with its dev dependency group using `uv <https://docs.astral.sh/uv/>`_::

       uv sync --group dev

   This installs Optiland in editable mode plus ``pytest``, ``ruff``, ``mypy``, and ``vulture``.
   Without ``uv``, ``pip install -e "." --group dev`` (requires pip 25.1+; or install the
   packages listed under ``[dependency-groups].dev`` in ``pyproject.toml`` manually) works too.
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
- **Leave a comment on the issue** if you plan to work on it, and wait to be assigned before opening a pull request.
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

Building the Documentation
--------------------------

The documentation lives in ``docs/`` as reStructuredText pages, Jupyter
notebooks (committed **with** their outputs; builds never execute them) and
API pages generated from docstrings. It is published automatically at
https://www.optiland.org/docs/ by the ``Docs`` GitHub Actions workflow on
every push to ``master``, so a documentation change needs nothing beyond a
normal pull request.

Build it locally with the same pinned environment the hosted build uses::

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

- ``pip install -r docs/requirements.txt`` mirrors the Python pins for pip/uv
  users; ``pandoc`` must then be installed separately.
- The :doc:`try_it` page is built by ``jupyterlite-xeus`` with ``micromamba``.
  On hosts where that is not possible (for example Windows), pass
  ``--skip-jupyterlite`` to ``build_docs.py``; CI always builds it.
- CI treats every Sphinx warning as an error unless it is listed, with a
  reason, in ``scripts/docs/warnings-allowlist.txt``. Fix the warning at the
  source rather than extending the allowlist.
- Never commit generated HTML. Never hardcode the version: it is derived from
  the package metadata.

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

.. toctree::
   :hidden:
   :maxdepth: 1

   authors
   license
   references
