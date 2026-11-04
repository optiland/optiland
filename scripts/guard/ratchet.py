"""Quality ratchet: a pull request should not add code-quality debt.

Per-file debt counts (complexity, missing types or docstrings, suppressions, …)
are measured for the pull request and for its merge base, and every count that
grew is reported. Measuring both sides on the same runner means there is no
baseline file to maintain and nothing for contributors to update. The ratchet
never fails CI; regressions are listed in the Guard summary for review.

Usage (from the repository root)::

    python -m scripts.guard.ratchet [--base origin/master]
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

from scripts.guard.files import ROOT, git, tracked_python_files

# Ruff rules counted by the ratchet but not enforced by the main lint config.
RUFF_METRICS: dict[str, str] = {
    "C901": "complex function",
    "PLR0912": "too many branches",
    "PLR0913": "too many arguments",
    "PLR0915": "too many statements",
    "ANN001": "untyped argument",
    "ANN201": "missing return type (public)",
    "ANN202": "missing return type (private)",
    "ANN204": "missing return type (special method)",
    "ANN401": "Any used as a type",
    "D102": "undocumented public method",
    "D107": "undocumented __init__",
    "T201": "print() in library code",
    "ERA001": "commented-out code",
    "PLC0415": "import inside a function",
    "PGH004": "blanket noqa",
}
CUSTOM_METRICS: dict[str, str] = {
    "suppression": "noqa / type: ignore / pragma: no cover",
    "array-import": "numpy/torch imported outside optiland/backend",
    "long-file": "lines beyond 800",
}
METRIC_NAMES = {**RUFF_METRICS, **CUSTOM_METRICS}

_SUPPRESSION = re.compile(r"#\s*(noqa\b|type:\s*ignore\b|pragma:\s*no cover\b)")
_ARRAY_MODULES = ("numpy", "torch")
_MAX_LINES = 800

Counts = dict[str, dict[str, int]]
Regression = tuple[str, str, int, int]


def _ruff_counts(root: Path, tracked: set[str]) -> Counts:
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--no-cache",
        "--exit-zero",
        "--output-format",
        "json",
        "--select",
        ",".join(RUFF_METRICS),
        "optiland",
    ]
    out = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True)
    counts: Counts = defaultdict(lambda: defaultdict(int))
    for item in json.loads(out.stdout or "[]"):
        rel = Path(item["filename"]).resolve().relative_to(root.resolve()).as_posix()
        if rel in tracked:
            counts[rel][item["code"]] += 1
    return counts


def _array_imports(tree: ast.AST) -> int:
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names = [node.module or ""]
        else:
            continue
        total += sum(n.split(".")[0] in _ARRAY_MODULES for n in names)
    return total


def measure(root: Path, files: list[str]) -> Counts:
    """Return ``{path: {metric: count}}`` for ``files`` under ``root``."""
    tracked = set(files)
    counts = _ruff_counts(root, tracked)
    for rel in files:
        text = (root / rel).read_text(encoding="utf-8")
        counts[rel]["suppression"] += len(_SUPPRESSION.findall(text))
        if not rel.startswith("optiland/backend/"):
            with contextlib.suppress(SyntaxError):
                counts[rel]["array-import"] += _array_imports(ast.parse(text))
        counts[rel]["long-file"] += max(0, len(text.splitlines()) - _MAX_LINES)
    return {path: dict(metrics) for path, metrics in counts.items()}


def measure_head() -> Counts:
    """Measure the working tree."""
    files = [p.relative_to(ROOT).as_posix() for p in tracked_python_files("optiland/")]
    return measure(ROOT, files)


def measure_ref(ref: str) -> Counts:
    """Measure the tree of git ``ref`` (extracted to a temporary directory)."""
    listing = git("ls-tree", "-r", "--name-only", ref, "--", "optiland/")
    files = [f for f in listing.split() if f.endswith(".py")]
    archive = subprocess.run(
        ["git", "archive", "--format=tar", ref, "optiland", "pyproject.toml"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(tmp, filter="data")
        return measure(Path(tmp), files)


def renames(base: str) -> dict[str, str]:
    """Return ``{new_path: old_path}`` for files renamed since ``base``."""
    out = git("diff", "-M", "--name-status", "--diff-filter=R", base, "--", "optiland/")
    mapping = {}
    for line in out.splitlines():
        _, old, new = line.split("\t")
        mapping[new] = old
    return mapping


def compare(
    base: Counts, head: Counts, renamed: dict[str, str] | None = None
) -> tuple[list[Regression], int]:
    """Return ``(regressions, improved)``.

    ``regressions`` holds ``(path, metric, before, now)`` for every count that
    grew; ``improved`` is the total amount by which counts shrank. A renamed
    file is compared with its old path.
    """
    renamed = renamed or {}
    regressions: list[Regression] = []
    improved = 0
    for path in sorted(head.keys() | base.keys()):
        if path in renamed.values() and path not in head:
            continue  # counted under its new name
        old = base.get(renamed.get(path, path), {})
        new = head.get(path, {})
        for metric in sorted(old.keys() | new.keys()):
            before, now = old.get(metric, 0), new.get(metric, 0)
            if now > before:
                regressions.append((path, metric, before, now))
            else:
                improved += before - now
    return regressions, improved


def run(base: str) -> tuple[list[Regression], int]:
    """Measure ``base`` and the working tree, and compare them."""
    return compare(measure_ref(base), measure_head(), renames(base))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/master", help="ref to compare to")
    args = parser.parse_args(argv)
    merge_base = git("merge-base", args.base, "HEAD").strip()
    regressions, improved = run(merge_base)
    for path, metric, before, now in regressions:
        print(f"{path}: {metric} ({METRIC_NAMES[metric]}) {before} -> {now}")
    print(f"{len(regressions)} regressions, {improved} improvements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
