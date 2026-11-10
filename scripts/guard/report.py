"""Guard report: a one-screen summary of what a pull request changes in kind.

Nothing here fails CI. The report is written as Markdown (to the GitHub job
summary in CI) so a reviewer can see at a glance where to look.

Usage (from the repository root)::

    python -m scripts.guard [--base origin/master]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib

from scripts.guard import api_snapshot, ratchet
from scripts.guard.files import ROOT, git

SIZE_BUDGET = 400
# Files excluded from the size count: tests, lock files and generated data.
_SIZE_EXCLUDE = ("tests/", "uv.lock")
_FIXTURE_DIRS = ("tests/golden/fixtures/", "tests/regression/fixtures/")
_JUSTIFICATIONS = "tests/golden/JUSTIFICATIONS.md"
_TOLERANCE = re.compile(r"\b(rtol|atol|abs|rel|decimal)\s*=\s*([0-9.eE+-]+)")
_SKIP_MARK = re.compile(r"pytest\.(mark\.)?(skip|skipif|xfail)\b")
_TEST_DEF = re.compile(r"^\s*def (test_\w+)")
_DOC_PATH = re.compile(r"`((?:optiland|optiland_gui|tests|scripts|docs)/[\w./-]+)`")


def _changed_files(base: str) -> list[tuple[str, str]]:
    out = git("diff", "--name-status", "--no-renames", base)
    return [tuple(line.split("\t", 1)) for line in out.splitlines()]  # type: ignore[misc]


def _size(base: str) -> int:
    total = 0
    for line in git("diff", "--numstat", base).splitlines():
        added, deleted, path = line.split("\t", 2)
        if added == "-" or path.startswith(_SIZE_EXCLUDE):
            continue
        total += int(added) + int(deleted)
    return total


def _subpackages(paths: list[str]) -> list[str]:
    found = set()
    for path in paths:
        parts = path.split("/")
        if parts[0] == "optiland" and len(parts) > 2:
            found.add(parts[1])
    return sorted(found)


def _dependencies(base: str) -> list[str]:
    def deps(text: str) -> set[str]:
        project = tomllib.loads(text).get("project", {})
        found = {f"dependency: {d}" for d in project.get("dependencies", [])}
        for extra, items in project.get("optional-dependencies", {}).items():
            found |= {f"[{extra}]: {d}" for d in items}
        return found

    old = deps(git("show", f"{base}:pyproject.toml"))
    new = deps((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return [f"+ {d}" for d in sorted(new - old)] + [f"- {d}" for d in sorted(old - new)]


def _test_integrity(base: str) -> list[str]:
    """Flag loosened tolerances, new skips and deleted tests under ``tests/``."""
    notes: list[str] = []
    diff = git("diff", "-U0", base, "--", "tests/")
    path = ""
    removed_tol: dict[str, list[float]] = {}
    removed_tests: set[str] = set()
    added_tests: set[str] = set()

    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else path
            continue
        if line.startswith(("--- ", "diff ")):
            continue
        if line.startswith("@@"):
            removed_tol.clear()
            continue
        sign, body = line[:1], line[1:]
        if sign == "-":
            for name, value in _TOLERANCE.findall(body):
                removed_tol.setdefault(name, []).append(_to_float(value))
            if match := _TEST_DEF.match(body):
                removed_tests.add(match.group(1))
        elif sign == "+":
            for name, value in _TOLERANCE.findall(body):
                old = removed_tol.get(name)
                new = _to_float(value)
                if old and _looser(name, max(old), new):
                    notes.append(f"loosened `{name}` in `{path}`: {body.strip()}")
            if _SKIP_MARK.search(body):
                notes.append(f"new skip/xfail in `{path}`: {body.strip()}")
            if match := _TEST_DEF.match(body):
                added_tests.add(match.group(1))
    for name in sorted(removed_tests - added_tests):
        notes.append(f"test removed: `{name}`")
    return notes


def _to_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _looser(name: str, old: float, new: float) -> bool:
    # ``decimal`` counts digits, so fewer digits is looser.
    return new < old if name == "decimal" else new > old


def _fixtures(paths: list[str]) -> list[str]:
    changed = [p for p in paths if p.startswith(_FIXTURE_DIRS)]
    if changed and _JUSTIFICATIONS not in paths:
        return [
            f"{len(changed)} fixture file(s) changed without a {_JUSTIFICATIONS} entry"
        ]
    return [f"{len(changed)} fixture file(s) changed, justified"] if changed else []


def _doc_refs(paths: list[str]) -> list[str]:
    broken = []
    for path in paths:
        if not path.endswith((".md", ".rst")) or not (ROOT / path).is_file():
            continue
        text = (ROOT / path).read_text(encoding="utf-8", errors="replace")
        for ref in sorted(set(_DOC_PATH.findall(text))):
            target = ref.split(":", 1)[0].rstrip(".")
            if "*" in target or "<" in target:
                continue
            if not (ROOT / target).exists():
                broken.append(f"`{path}` mentions missing `{target}`")
    return broken


def _api() -> tuple[list[str], int]:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.guard.api_snapshot", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    removed, changed, added = api_snapshot.compare(
        api_snapshot.read_snapshot(), data["lines"], data["failed"]
    )
    return added, len(removed) + len(changed)


def build(base: str) -> str:
    """Return the Markdown report for the changes since ``base``."""
    changes = _changed_files(base)
    paths = [p for _, p in changes]
    size = _size(base)
    packages = _subpackages(paths)
    added_api, broken_api = _api()
    regressions, improved = ratchet.run(base)
    deps = _dependencies(base)
    integrity = _test_integrity(base)
    fixtures = _fixtures(paths)
    doc_refs = _doc_refs(paths)

    def row(name: str, ok: bool, text: str) -> str:
        return f"| {name} | {'✅' if ok else '⚠️'} | {text} |"

    rows = [
        row(
            "Size",
            size <= SIZE_BUDGET,
            f"{size} changed lines excluding tests (guide: ≤{SIZE_BUDGET})",
        ),
        row(
            "Spread",
            len(packages) <= 2,
            f"{len(packages)} subpackage(s): {', '.join(packages) or 'none'}",
        ),
        row(
            "Public API",
            broken_api == 0,
            f"{len(added_api)} added, {broken_api} broken (breaks fail the API test)",
        ),
        row("Dependencies", not deps, f"{len(deps)} changed"),
        row(
            "Quality ratchet",
            not regressions,
            f"{len(regressions)} count(s) grew, {improved} improvement(s)",
        ),
        row("Test integrity", not integrity, f"{len(integrity)} flag(s)"),
        row(
            "Fixtures",
            not any("without" in f for f in fixtures),
            fixtures[0] if fixtures else "untouched",
        ),
        row("Doc references", not doc_refs, f"{len(doc_refs)} broken path(s)"),
    ]
    out = [
        "## Guard report",
        "",
        "Informational only; nothing here blocks the PR. ⚠️ marks where to look.",
        "",
        "| Check | | Result |",
        "|---|---|---|",
        *rows,
    ]

    def details(title: str, items: list[str]) -> None:
        if items:
            shown = items[:50] + (
                [f"… and {len(items) - 50} more"] if len(items) > 50 else []
            )
            out.extend(["", f"<details><summary>{title}</summary>", ""])
            out.extend(f"- {item}" for item in shown)
            out.extend(["", "</details>"])

    details("New public API", [f"`{a}`" for a in added_api])
    details("Dependency changes", [f"`{d}`" for d in deps])
    details(
        "Quality ratchet regressions",
        [
            f"`{p}`: {ratchet.METRIC_NAMES[m]} ({m}) {b} → {n}"
            for p, m, b, n in regressions
        ],
    )
    details("Test integrity", integrity)
    details("Broken doc references", doc_refs)
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/master", help="ref to compare to")
    args = parser.parse_args(argv)
    merge_base = git("merge-base", args.base, "HEAD").strip()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(build(merge_base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
