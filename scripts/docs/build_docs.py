"""Build the Sphinx HTML documentation with the settings used by CI.

This is a thin wrapper around ``python -m sphinx``: it standardises the
output locations, records warnings to a log file, keeps doctrees out of the
deployable tree, and exports the environment variables read by
``docs/conf.py``. It does not implement a second renderer.

Usage::

    python scripts/docs/build_docs.py                      # docs/_build/html
    python scripts/docs/build_docs.py --strict             # any warning fails
    python scripts/docs/build_docs.py --skip-jupyterlite   # local iteration

The regular local workflow ``make -C docs html`` keeps working unchanged.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", type=Path, default=REPO_ROOT / "docs")
    ap.add_argument(
        "--output", type=Path, default=REPO_ROOT / "docs" / "_build" / "html"
    )
    ap.add_argument(
        "--doctrees", type=Path, default=REPO_ROOT / "docs" / "_build" / "doctrees"
    )
    ap.add_argument(
        "--warnings-log",
        type=Path,
        default=REPO_ROOT / "docs" / "_build" / "warnings.log",
    )
    ap.add_argument(
        "--canonical-base-url",
        default=os.environ.get(
            "OPTILAND_DOCS_BASE_URL", "https://www.optiland.org/docs/"
        ),
    )
    ap.add_argument("--git-sha", default=os.environ.get("OPTILAND_DOCS_GIT_SHA", ""))
    ap.add_argument("--builder", default="html")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Pass -W: turn every warning into an error",
    )
    ap.add_argument(
        "--skip-jupyterlite",
        action="store_true",
        help="Skip the JupyterLite build (local hosts without an emscripten toolchain)",
    )
    ap.add_argument(
        "--fresh", action="store_true", help="Ignore cached doctrees (sphinx -E)"
    )
    ap.add_argument(
        "--jobs", default=None, help="Sphinx -j value (e.g. 'auto'); serial by default"
    )
    args = ap.parse_args(argv)

    env = dict(os.environ)
    env["OPTILAND_DOCS_BASE_URL"] = args.canonical_base_url
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if args.git_sha:
        env["OPTILAND_DOCS_GIT_SHA"] = args.git_sha
    if args.skip_jupyterlite:
        env["OPTILAND_DOCS_SKIP_JUPYTERLITE"] = "1"

    args.warnings_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        args.builder,
        "--keep-going",
        "-d",
        str(args.doctrees),
        "-w",
        str(args.warnings_log),
    ]
    if args.strict:
        cmd.append("-W")
    if args.fresh:
        cmd.append("-E")
    if args.jobs:
        cmd += ["-j", str(args.jobs)]
    cmd += [str(args.source), str(args.output)]

    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if args.warnings_log.is_file():
        count = sum(
            1
            for line in args.warnings_log.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.strip()
        )
        print(
            f"sphinx exit code {result.returncode}; "
            f"{count} warning line(s) in {args.warnings_log}"
        )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
