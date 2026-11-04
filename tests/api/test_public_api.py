"""The public API is frozen: nothing in the committed snapshot may break.

New public symbols and new optional parameters are allowed; they are listed in
the Guard report on pull requests. If a break is truly intended and approved by
the maintainers, regenerate the snapshot with
``python -m scripts.guard.api_snapshot --update``.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts.guard import api_snapshot
from scripts.guard.api_snapshot import ROOT, compare, is_compatible, read_snapshot


@pytest.fixture(scope="module")
def live_api() -> tuple[list[str], list[str]]:
    # A subprocess keeps the import of every module from leaking global state
    # (backend selection, registries, warning filters) into other tests.
    result = subprocess.run(
        [sys.executable, "-m", "scripts.guard.api_snapshot", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return data["lines"], data["failed"]


def test_public_api_is_backward_compatible(live_api):
    lines, failed = live_api
    removed, changed, _ = compare(read_snapshot(), lines, failed)
    problems = [f"removed: {ln}" for ln in removed]
    problems += [f"changed: {old}\n     now: {new}" for old, new in changed]
    assert not problems, (
        "Public API broke. Keep the old name/signature working and mark it with "
        "optiland._deprecation.deprecated instead.\n" + "\n".join(problems)
    )


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("m:f(a, b=1)", "m:f(a, b=1)", True),
        ("m:f(a, b=1)", "m:f(a, b=1, c=2)", True),
        ("m:f(a, b=1)", "m:f(a, b=1, *, c=2)", True),
        ("m:f(a)", "m:f(a=0)", True),
        ("m:f(a, b=1)", "m:f(a, b=1, c)", False),
        ("m:f(a, b=1)", "m:f(a, b=2)", False),
        ("m:f(a, b=1)", "m:f(b=1, a=0)", False),
        ("m:f(a, b=1)", "m:f(a)", False),
        ("m:f(a, *, b=1)", "m:f(a, *, c=1)", False),
        ("m:f(a, **kw)", "m:f(a)", False),
        ("m:f(a, *args)", "m:f(a)", False),
        ("m:f(a, *, b)", "m:f(a, *, b, c)", False),
        ("m:f(?)", "m:f(a)", False),
    ],
)
def test_is_compatible(old, new, expected):
    assert is_compatible(old, new) is expected


def test_compare_skips_modules_that_cannot_be_imported():
    snapshot = ["a:f(x)", "b:g(y)"]
    removed, changed, added = compare(snapshot, ["a:f(x)", "a:h()"], ["b"])
    assert (removed, changed, added) == ([], [], ["a:h()"])


def test_snapshot_lines_are_sorted_and_unique():
    lines = read_snapshot(api_snapshot.SNAPSHOT_PATH)
    assert lines == sorted(set(lines))
