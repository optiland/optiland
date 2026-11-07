"""Helpers for listing tracked source files and reading their imports."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]

# Gitignored scratch areas that may exist in a local checkout.
_LOCAL_ONLY = ("optiland/jupyter/", "optiland/scripts/")


def git(*args: str) -> str:
    """Run ``git`` in the repository root and return its stdout."""
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def tracked_python_files(prefix: str) -> list[Path]:
    """Return git-tracked ``.py`` files under ``prefix`` (relative to the root).

    Falls back to a filesystem glob when git is unavailable, skipping the
    gitignored scratch directories so local and CI results agree.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", f"{prefix}*.py", f"{prefix}**/*.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        rel = sorted(set(out.split()))
    except (OSError, subprocess.CalledProcessError):
        rel = sorted(
            p.relative_to(ROOT).as_posix() for p in ROOT.glob(f"{prefix}**/*.py")
        )
    return [ROOT / r for r in rel if not r.startswith(_LOCAL_ONLY)]


def module_name(path: Path) -> str:
    """Return the dotted module name for a file under the repository root."""
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_modules(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line, absolute module name)`` for every import in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module_name(path)
    if path.name != "__init__.py":
        package = package.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - node.level + 1]
                name = ".".join([*base, node.module] if node.module else base)
            else:
                name = node.module or ""
            yield node.lineno, name
