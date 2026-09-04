"""Fail when a reference to the retired Read the Docs site is (re)introduced.

The canonical documentation URL is https://www.optiland.org/docs/. A short
allowlist covers the files that legitimately mention the old host: the
validation tooling that rejects such links and the operations runbook that
documents the migration.

Usage::

    python scripts/docs/check_no_rtd_links.py          # scan the git tree
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST = "optiland.readthedocs.io"
ALLOWLIST = {
    "scripts/docs/check_no_rtd_links.py",
    "scripts/docs/validate_build.py",
    "scripts/docs/smoke_test.py",
    "docs/developers_guide/documentation_operations.rst",
    ".github/workflows/docs.yml",
}


def main() -> int:
    result = subprocess.run(
        ["git", "grep", "-n", "-I", "--", HOST],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    offenders = []
    for line in result.stdout.splitlines():
        path = line.split(":", 1)[0].replace("\\", "/")
        if path not in ALLOWLIST:
            offenders.append(line)
    for line in offenders:
        print(f"ERROR: {line}")
    if offenders:
        print(
            f"\n{len(offenders)} reference(s) to {HOST}. Link to "
            "https://www.optiland.org/docs/ instead, or add a documented "
            "allowlist entry in scripts/docs/check_no_rtd_links.py."
        )
        return 1
    print(f"no references to {HOST} outside the allowlist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
