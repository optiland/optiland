"""Command-line runner for the optimization benchmarks.

Usage:
    python -m benchmarks.optimization.run --quick
    python -m benchmarks.optimization.run --problems cooke double_gauss \
        --methods scipy_lm --json out.json

Prints a markdown table (paste it into the PR description) and optionally
writes the raw results as JSON.

Kramer Harrison, 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optiland.backend as be

from .harness import DEFAULT_MAXITER, METHODS, run_benchmark, to_markdown
from .problems import PROBLEMS, QUICK_PROBLEMS


def main(argv: list[str] | None = None) -> int:
    """Run the selected benchmarks and report the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--problems", nargs="+", choices=sorted(PROBLEMS))
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS))
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Run only {', '.join(QUICK_PROBLEMS)} (ignored with --problems).",
    )
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--json", type=Path, help="Write raw results to this file.")
    args = parser.parse_args(argv)

    be.set_backend("numpy")
    names = args.problems or (list(QUICK_PROBLEMS) if args.quick else list(PROBLEMS))
    methods = args.methods or list(METHODS)

    results = []
    for name in names:
        for method in methods:
            print(f"running {name} / {method} ...", file=sys.stderr)
            results.append(run_benchmark(name, PROBLEMS[name], method, args.maxiter))

    print(to_markdown(results))
    if args.json:
        args.json.write_text(
            json.dumps([r.to_dict() for r in results], indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
