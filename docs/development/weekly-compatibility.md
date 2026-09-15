# Weekly dependency compatibility

Ordinary CI varies Python 3.11–3.14 while installing the same `uv.lock` versions.
Issue [#746](https://github.com/optiland/optiland/issues/746) showed why that is
insufficient: an array-to-scalar conversion allowed with a warning by the locked
NumPy 2.3.5 fails on NumPy 2.4+. More supported dependency combinations need to
be exercised regularly, before users discover incompatibilities. The scalar
test fix is separate in [#848](https://github.com/optiland/optiland/pull/848).

The weekly workflow runs on **Fridays at 03:23 UTC**, leaving contributors the
weekend to investigate failures. It also supports manual runs. Scheduled runs
are restricted to `optiland/optiland`; forks can qualify changes manually.
GitHub may delay scheduled runs and disables public-repository schedules after
60 days without activity; re-enable a disabled workflow from the Actions page.

## Why these eight combinations?

The initial matrix samples the supported Python minors, NumPy 2.3/2.4/2.5,
coupled Numba/llvmlite families, two Torch versions, and genuine Torch absence:

- A Python 3.11 control follows the committed lock, with the matching CPU Torch
  build. It distinguishes dependency changes from failures in the source itself.
- Four reviewed current stacks cover Python 3.11–3.14. Python 3.11 remains on
  NumPy 2.4 and a compatible SciPy because the newer families require 3.12+.
- A NumPy 2.4 row retains the scalar-conversion boundary with preceding Numba.
- A previous-Torch row and a Torch-absent row share the primary non-Torch pins
  of the current Python 3.14 row.

Every row is expected to work within project and dependency requirements.
None is allowed to fail silently. This is a practical sample, not an exhaustive
Cartesian product or a guarantee about every version allowed by project metadata.
None exactly matches the original report's Python 3.14.7 / NumPy 2.5.1 pair:
the NumPy 2.4+ rows target the same failure mechanism with reviewed newer patches.
Focused Windows reproduction of the initial eight stacks passed all 165 cases
with #848 applied; that does not establish full-suite Ubuntu compatibility.

## Change combinations in one place

Edit [`.github/compatibility-matrix.toml`](../../.github/compatibility-matrix.toml).
It is the only maintained version table. Print its current contents with:

```sh
python scripts/ci/compatibility_matrix.py show
python scripts/ci/compatibility_matrix.py validate
python -m unittest discover -s tests/ci -v
```

Change a pin by editing its quoted exact stable version. Keep Numba and llvmlite
compatible. Add another constrained package, such as pandas, to `packages`.
Add or remove one `[[rows]]` block to change the combinations. Keep stable IDs
when upgrading versions so job history and rerun commands remain useful. There
are no row-specific branches or duplicated workflow choices to update.

Each row has `id`, `purpose`, `python`, `source`, and `torch` fields. A `pinned`
source requires a `packages` table with NumPy, Numba, llvmlite and SciPy pins,
plus Torch for `torch = "cpu"`. A `project-lock` source reads pins from `uv.lock`
and must not supply a package table. `torch = "absent"` omits the extra and
verifies that Torch is not installed or importable. Package names use lowercase
and hyphens; versions are exact stable releases, not ranges or `latest`.

Review pins **monthly and after supported Python/NumPy/Numba/Torch releases**.
Check upstream compatibility requirements and Linux x64 wheels, then qualify
the affected rows before merging. Primary pins do not automatically advance;
they will miss a future numerical-library regression until refreshed. Other
dependencies resolve to compatible stable versions declared by `pyproject.toml`.
The lock control instead constrains those versions from the project lock.

In Actions, choose **Weekly dependency compatibility → Run workflow**, select
the branch, and optionally enter comma-separated IDs, for example
`current-py314,without-torch-py314`. Empty runs all rows; unknown IDs fail.
Version changes live in the branch's manifest, not free-form dispatch inputs.
GitHub requires the workflow to be registered on the default branch before
manual dispatch is available. For a new workflow, qualify with a temporary
fork-only branch trigger or register a dispatch-only workflow in the fork first.

## Resolution, execution and reports

Each Ubuntu job uses a fresh environment. uv compiles the checkout's declared
requirements and extras under the selected constraints, then installs the exact
resolved requirements with hashes. CPU Torch comes from the official CPU index
through `--torch-backend cpu`; CUDA packages in the control lock are constraints,
not requirements to install. GUI and other ordinary CI extras remain included.
Compiled third-party packages require wheels. The checkout is installed editable
without another dependency resolution; neither `uv.lock` nor application code
is modified. Tests invoke that environment's Python directly, never `uv run`.

For local resolution, use the selected Python minor and the workflow's uv version:

```sh
uv venv .compat-venv --python 3.14
.compat-venv/bin/python scripts/ci/compatibility_matrix.py resolve --row current-py314
uv pip sync --python .compat-venv/bin/python --require-hashes --only-binary :all: \
  --torch-backend cpu compatibility-reports/current-py314/requirements.txt
uv pip install --python .compat-venv/bin/python --no-deps -e .
uv pip check --python .compat-venv/bin/python
.compat-venv/bin/python scripts/ci/compatibility_matrix.py verify --row current-py314
```

On Windows, use `.compat-venv/Scripts/python.exe` instead of the `bin/python`
path. Local Windows execution is supplementary; qualify the workflow on GitHub
Ubuntu without needing a Linux VM or WSL installation. For an old failure, use
the recorded source SHA, exact Python patch and archived `requirements.txt`
instead of resolving again. Wheel availability and the runner image can change.

All rows run the full pytest tree with normal Numba JIT, CPU execution, Qt
offscreen, Xvfb and the existing coverage configuration. Torch-present rows
exercise both backends; the absent row still runs the full core and GUI suite.
No broad warning filters, skip rules or continue-on-error settings are added.

Reports retain requested and installed versions, source SHA, Python/uv/runner
versions, full resolution with hashes, logs, stage outcomes, JUnit and coverage
XML. The summary fails on missing reports and failing stages. Compare matching
primary pins using the recorded non-Torch inventories; differing transitive
versions prevent a strictly isolated Torch comparison. Installation errors and
unexpected optional-dependency skips need investigation, not suppression.
Hard timeouts or runner loss may prevent uploads; inspect the Actions logs too.

## Workload and maintenance checks

The manifest controls three concurrent jobs, a 75-minute job timeout, and
30-day artifact retention. The eight full suites are estimated at 4–8 aggregate
runner-hours weekly; measure actual usage before expanding the matrix. Cache
downloads, not environments. Application imports and runtime have no CI-helper
dependency, and ordinary PRs do not run these eight extra test environments.

PR changes to the matrix/helper/workflow run lightweight validation and helper
tests. Normal CI also collects the helper tests. A common resolver or workflow
change needs complete matrix qualification; a pin edit needs affected-row
qualification. Keep weekly coverage as diagnostic artifacts rather than mixing
changing dependency environments into ordinary PR Codecov results.

Merge #848 before enabling this workflow on a default branch that still has the
old scalar-conversion tests. This workflow detects the regression; it does not
implement the scalar fix or automatically open issues/PRs when tests fail.
