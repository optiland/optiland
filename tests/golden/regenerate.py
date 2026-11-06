"""Regenerate the golden fixtures in ``tests/golden/fixtures``.

Fixtures change only for an intentional numerical change, and the PR must
justify it. A justification is therefore mandatory and is appended to
``tests/golden/JUSTIFICATIONS.md``.

Usage:
    python -m tests.golden.regenerate --justification "why the numbers move"

Kramer Harrison, 2026
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import optiland.backend as be

from .snapshot import GOLDEN_PROBLEMS, compute_snapshot

GOLDEN_DIR = Path(__file__).parent
FIXTURES_DIR = GOLDEN_DIR / "fixtures"


def main(argv: list[str] | None = None) -> int:
    """Rewrite every fixture and log the justification."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--justification",
        required=True,
        help="Why the golden values are changing (recorded in JUSTIFICATIONS.md).",
    )
    args = parser.parse_args(argv)
    justification = args.justification.strip()
    if not justification:
        parser.error("--justification must not be empty")

    be.set_backend("numpy")
    FIXTURES_DIR.mkdir(exist_ok=True)
    for name in GOLDEN_PROBLEMS:
        path = FIXTURES_DIR / f"{name}.json"
        path.write_text(
            json.dumps(compute_snapshot(name), indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")

    log = GOLDEN_DIR / "JUSTIFICATIONS.md"
    if not log.exists():
        log.write_text("# Golden fixture regeneration log\n\n", encoding="utf-8")
    with log.open("a", encoding="utf-8") as f:
        f.write(f"- {datetime.date.today().isoformat()}: {justification}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
