"""Post-merge procedure for the fused trace kernel (plan 6.5).

Run this after every merge of ``upstream/master`` into the fork's ``metal``
branch.  The kernel in ``optiland/backend/torch_backend/metal/kernels/trace.metal``
*mirrors* Python expressions of the upstream code base ("mirror, never
improve", plan 0.2.1); when upstream changes one of the mirrored functions the
kernel is no longer a faithful mirror, so the merge has to be acknowledged
explicitly before the fused path is trusted again.

``--check`` runs the five steps of plan 6.5 and exits non-zero until the
current ``upstream/master`` has been acknowledged with ``--ack``:

1. ``trace_mirror --check``: every MIRRORED fingerprint and every structural
   identity still holds;
2. ``pytest tests/metal/test_trace_mirror_sources.py
   tests/metal/test_trace_gate.py -k "catalog or routing"``;
3. ``metal_oracle_e2e.py --systems CookeTriplet,HubbleTelescope --fused both
   --rings 19``;
4. the upstream changelog for the mirrored areas
   (``git log <last-acked-sha>..upstream/master -- <mirrored paths>``), the
   fork's pre-existing upstream-file diff, and a check that the set of
   *modified* upstream files under ``optiland/`` is still exactly the frozen
   set of :data:`FORK_MODIFIED_FILES` (plan 0.2.3; the set was measured at I0
   and recorded in ``NOTES/fused-trace-research/status.md``);
5. the acknowledgement itself: ``--ack <sha> --hunk "..."`` appends a dated
   section to ``status.md`` naming the merged sha and one line per
   acknowledged hunk, and is what makes ``--check`` exit 0.

Two utilities used by the definition of done (plan 0.4) live here as well:

* ``--compare-junit BASELINE NEW``: exits 1 when ``NEW`` has a failing or
  erroring test case that ``BASELINE`` does not (test-case identity is
  ``classname`` + ``name``);
* ``--lint-scope committed``: ``ruff check`` and ``ruff format --check`` over
  every ``*.py`` file touched between the parent of commit C1 and ``HEAD``,
  which keeps the lint gate off the 34 ruff errors that pre-exist on the
  fork's HEAD in files this work never touches.

Usage (from anywhere; paths are derived from this file's location)::

    python Optiland-Metal/scripts/metal_post_merge.py --check
    python Optiland-Metal/scripts/metal_post_merge.py --ack <sha> \
        --hunk "geometries/newton_raphson.py: docstring only, mirror intact"
    python Optiland-Metal/scripts/metal_post_merge.py --compare-junit a.xml b.xml
    python Optiland-Metal/scripts/metal_post_merge.py --lint-scope committed
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "Optiland-Metal"
PY = ROOT / ".venv" / "bin" / "python"
STATUS = ROOT / "NOTES" / "fused-trace-research" / "status.md"

#: Upstream paths whose changes can invalidate a mirror (plan 6.5 step 4).
MIRRORED_PATHS: tuple[str, ...] = (
    "optiland/backend/_conic.py",
    "optiland/geometries",
    "optiland/surfaces",
    "optiland/coordinate_system.py",
    "optiland/rays",
    "optiland/interactions",
    "optiland/propagation",
    "optiland/materials/base.py",
    "optiland/utils.py",
)

#: The upstream files the fork modifies, measured at I0 (plan 0.2.3) plus the
#: hook of commit C3.  ``--check`` fails when the live set differs: a new entry
#: means the fork grew a divergence that nobody recorded, a missing entry means
#: a merge reverted one.
FORK_MODIFIED_FILES: tuple[str, ...] = (
    "optiland/backend/torch_backend/capabilities.py",
    "optiland/backend/torch_backend/config.py",
    "optiland/backend/torch_backend/conic.py",
    "optiland/backend/torch_backend/creation.py",
    "optiland/backend/torch_backend/indexing.py",
    "optiland/backend/torch_backend/interpolation.py",
    "optiland/backend/torch_backend/linalg.py",
    "optiland/backend/torch_backend/misc.py",
    "optiland/backend/torch_backend/passthrough.py",
    "optiland/backend/torch_backend/random.py",
    "optiland/backend/torch_backend/reductions.py",
    "optiland/backend/utils.py",
    "optiland/materials/material_utils.py",
    "optiland/surfaces/surface_group.py",
    "optiland/utils.py",
)

#: Subject of commit C1; its parent is the default ``--lint-scope`` base.
C1_SUBJECT = "Add fused-trace layout contract"

#: ``- post-merge ack: upstream <sha>`` lines in ``status.md`` are the record
#: of which upstream state the mirrors were last verified against.
_ACK_RE = re.compile(r"^- post-merge ack: upstream ([0-9a-f]{7,40})\b", re.M)

_FIXED_ENV = {
    "PYTORCH_MPS_FAST_MATH": "0",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    "MPLBACKEND": "Agg",
    "QT_QPA_PLATFORM": "offscreen",
}


def _env() -> dict[str, str]:
    """The fixed environment every Python invocation of this project uses."""
    return dict(os.environ, **_FIXED_ENV)


def _run(cmd: list[str], *, capture: bool = False) -> tuple[int, str]:
    """Run ``cmd`` in the fork root; return ``(returncode, stdout)``."""
    print(f"$ {' '.join(cmd)}", flush=True)
    if capture:
        r = subprocess.run(
            cmd, cwd=REPO, env=_env(), capture_output=True, text=True, check=False
        )
        return r.returncode, r.stdout + r.stderr
    r = subprocess.run(cmd, cwd=REPO, env=_env(), check=False)
    return r.returncode, ""


def _git(*args: str) -> tuple[int, str]:
    """Run ``git`` in the fork root, capturing its output."""
    r = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    return r.returncode, (r.stdout + r.stderr).rstrip("\n")


# ---------------------------------------------------------------------------
# Acknowledgements
# ---------------------------------------------------------------------------


def last_acked_sha() -> str | None:
    """The upstream sha of the most recent ``--ack`` row in ``status.md``."""
    if not STATUS.exists():
        return None
    found = _ACK_RE.findall(STATUS.read_text(encoding="utf-8"))
    return found[-1] if found else None


def upstream_sha() -> str | None:
    """``upstream/master``'s sha, or None when the remote ref is absent."""
    rc, out = _git("rev-parse", "upstream/master")
    return out.strip() if rc == 0 else None


def _same_commit(a: str, b: str) -> bool:
    """True when two (possibly abbreviated) shas name the same commit."""
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def ack(sha: str, hunks: list[str]) -> int:
    """Append the dated acknowledgement row of plan 6.5 step 5 to ``status.md``."""
    rc, full = _git("rev-parse", sha)
    if rc != 0:
        print(f"unknown sha {sha!r}: {full}")
        return 2
    full = full.strip()
    rc, head = _git("rev-parse", "--short", "HEAD")
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "",
        f"## {stamp} post-merge ack",
        "",
        f"- post-merge ack: upstream {full}",
        f"- fork HEAD at acknowledgement: `{head.strip()}`",
        "- acknowledged hunks (one line each, plan 6.5 step 5):",
    ]
    lines += [f"  - {h}" for h in hunks]
    lines.append("")
    with STATUS.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"acknowledged upstream {full} in {STATUS.name}")
    return 0


# ---------------------------------------------------------------------------
# JUnit comparison (plan 0.4 step 2)
# ---------------------------------------------------------------------------


def _junit_cases(path: Path) -> tuple[set[str], set[str]]:
    """``(all test ids, failing test ids)`` of a pytest JUnit XML file."""
    tree = ET.parse(path)
    every: set[str] = set()
    failing: set[str] = set()
    for case in tree.iter("testcase"):
        ident = f"{case.get('classname', '')}::{case.get('name', '')}"
        every.add(ident)
        if case.find("failure") is not None or case.find("error") is not None:
            failing.add(ident)
    return every, failing


def compare_junit(baseline: Path, new: Path) -> int:
    """Exit 1 when ``new`` fails a test case ``baseline`` does not."""
    base_all, base_fail = _junit_cases(baseline)
    new_all, new_fail = _junit_cases(new)
    appeared = sorted(new_fail - base_fail)
    fixed = sorted(base_fail - new_fail)
    added = sorted(new_all - base_all)
    missing = sorted(base_all - new_all)
    print(
        f"baseline {baseline.name}: {len(base_all)} tests, {len(base_fail)} failing\n"
        f"new      {new.name}: {len(new_all)} tests, {len(new_fail)} failing"
    )
    for ident in appeared:
        print(f"  NEW FAILURE  {ident}")
    for ident in fixed:
        print(f"  fixed        {ident}")
    for ident in added:
        print(f"  new test     {ident}")
    for ident in missing:
        print(f"  lost test    {ident}")
    if appeared:
        print(f"FAIL: {len(appeared)} new failing test case(s)")
        return 1
    if missing:
        print(f"FAIL: {len(missing)} test case(s) present in the baseline are gone")
        return 1
    print("OK: no new failure, no lost test case")
    return 0


# ---------------------------------------------------------------------------
# Scoped lint (plan 0.4 step 6)
# ---------------------------------------------------------------------------


def _c1_parent() -> str | None:
    """The parent of commit C1, found by its subject."""
    rc, out = _git("log", "--format=%H%x1f%s", "-n", "200")
    if rc != 0:
        return None
    for line in out.splitlines():
        sha, _, subject = line.partition("\x1f")
        if subject.startswith(C1_SUBJECT):
            rc2, parent = _git("rev-parse", f"{sha}^")
            return parent.strip() if rc2 == 0 else None
    return None


def lint_scope(since: str | None) -> int:
    """``ruff check`` + ``ruff format --check`` over the files C1..HEAD touched."""
    base = since or _c1_parent()
    if base is None:
        print(
            "cannot locate commit C1 by subject "
            f"{C1_SUBJECT!r}; pass --since <rev> explicitly"
        )
        return 2
    rc, out = _git("diff", "--name-only", f"{base}..HEAD", "--", "*.py")
    if rc != 0:
        print(out)
        return 2
    files = [f for f in out.split() if (REPO / f).exists()]
    if not files:
        print(f"no committed .py files between {base[:8]} and HEAD")
        return 0
    print(f"lint scope: {len(files)} file(s) committed since {base[:8]}")
    bad = 0
    rc, _ = _run([str(PY), "-m", "ruff", "check", *files])
    bad += int(rc != 0)
    rc, _ = _run([str(PY), "-m", "ruff", "format", "--check", *files])
    bad += int(rc != 0)
    print("LINT OK" if bad == 0 else "LINT FAIL")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# The five steps
# ---------------------------------------------------------------------------


def step_mirror() -> tuple[bool, str]:
    """Step 1: fingerprints and structural identities."""
    rc, _ = _run(
        [str(PY), "-m", "optiland.backend.torch_backend.metal.trace_mirror", "--check"]
    )
    return rc == 0, "mirror fingerprints and structural identities"


def step_tests() -> tuple[bool, str]:
    """Step 2: the mirror-source and gate catalog/routing tests.

    Plan 6.5 names ``test_trace_mirror_sources.py`` and ``test_trace_gate.py``
    with ``-k "catalog or routing"``, but measured on this tree that pair
    selects only ``test_reference_routing[df64|sf64]``: the catalog tests the
    step is after live in ``tests/metal/test_trace_adapters.py``
    (``test_gate_accepts_catalog`` over every shipped sample, and
    ``test_catalog_invariants``), so that file is part of the command.
    ``test_trace_conformance.py::test_catalog`` is deliberately left out -- it
    is the full conformance sweep, not a post-merge smoke test.
    """
    rc, _ = _run(
        [
            str(PY),
            "-m",
            "pytest",
            "tests/metal/test_trace_mirror_sources.py",
            "tests/metal/test_trace_gate.py",
            "tests/metal/test_trace_adapters.py",
            "-k",
            "catalog or routing",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
        ]
    )
    return rc == 0, ("mirror_sources + gate + adapters, -k 'catalog or routing'")


def step_oracle(rings: int) -> tuple[bool, str]:
    """Step 3: the two-system oracle, fused against the per-op path."""
    rc, _ = _run(
        [
            str(PY),
            "scripts/metal_oracle_e2e.py",
            "--systems",
            "CookeTriplet,HubbleTelescope",
            "--fused",
            "both",
            "--rings",
            str(rings),
        ]
    )
    return rc == 0, "oracle CookeTriplet+HubbleTelescope, fused vs per-op vs NumPy"


def step_upstream_report(since: str | None) -> tuple[bool, str]:
    """Step 4: the upstream changelog and the frozen fork-diff file set."""
    ok = True
    up = upstream_sha()
    if up is None:
        print("upstream/master is not available; run `git fetch upstream` first")
        return False, "upstream changelog and fork-diff check"
    base = since or last_acked_sha()
    if base is None:
        rc, base = _git("merge-base", "HEAD", "upstream/master")
        base = base.strip() if rc == 0 else up
        print(f"no acknowledgement on record; using merge-base {base[:8]}")
    print(f"\n--- upstream changes to mirrored paths since {base[:8]} ---")
    rc, out = _git(
        "log",
        "--oneline",
        f"{base}..upstream/master",
        "--",
        *MIRRORED_PATHS,
    )
    print(out if out.strip() else "(none)")
    ok &= rc == 0
    print("\n--- the fork's diff against upstream/master (optiland/) ---")
    rc, out = _git("diff", "--stat", "upstream/master", "HEAD", "--", "optiland")
    print(out.splitlines()[-1] if out.strip() else "(none)")
    ok &= rc == 0
    rc, out = _git("diff", "--name-status", "upstream/master", "HEAD", "--", "optiland")
    live = {
        line.split("\t", 1)[1]
        for line in out.splitlines()
        if line.startswith("M\t") or line.startswith("M ")
    }
    frozen = set(FORK_MODIFIED_FILES)
    grew, shrank = sorted(live - frozen), sorted(frozen - live)
    for f in grew:
        print(f"  UNRECORDED modified upstream file: {f}")
    for f in shrank:
        print(f"  no longer modified (was recorded at I0): {f}")
    if grew or shrank:
        print(
            "FAIL: the fork's modified-file set differs from the frozen set of "
            f"{len(frozen)} files (plan 0.2.3); record the change in status.md "
            "and update FORK_MODIFIED_FILES"
        )
        ok = False
    else:
        print(f"  modified upstream files: {len(live)}, exactly the frozen set")
    return ok, "upstream changelog and fork-diff check"


def step_ack_state() -> tuple[bool, str]:
    """Step 5: has the current ``upstream/master`` been acknowledged?"""
    up = upstream_sha()
    acked = last_acked_sha()
    if up is None:
        print("upstream/master is not available; cannot check the acknowledgement")
        return False, "acknowledgement of the merged upstream sha"
    if acked is None:
        print(
            f"no acknowledgement on record; after reviewing the hunks above run\n"
            f"    {Path(__file__).name} --ack {up[:12]} --hunk '<what was reviewed>'"
        )
        return False, "acknowledgement of the merged upstream sha"
    if not _same_commit(acked, up):
        print(
            f"last acknowledged upstream sha {acked[:12]} != current "
            f"{up[:12]}; review the hunks above, then run\n"
            f"    {Path(__file__).name} --ack {up[:12]} --hunk '<what was reviewed>'"
        )
        return False, "acknowledgement of the merged upstream sha"
    print(f"upstream {up[:12]} acknowledged in {STATUS.name}")
    return True, "acknowledgement of the merged upstream sha"


def check(skip: set[int], rings: int, since: str | None) -> int:
    """Run the plan 6.5 steps; exit non-zero until every one of them passes."""
    steps: dict[int, Any] = {
        1: step_mirror,
        2: step_tests,
        3: lambda: step_oracle(rings),
        4: lambda: step_upstream_report(since),
        5: step_ack_state,
    }
    results: list[tuple[int, bool | None, str]] = []
    for number, fn in steps.items():
        if number in skip:
            results.append((number, None, f"step {number}: SKIPPED on request"))
            continue
        print(f"\n=== post-merge step {number} ===")
        ok, what = fn()
        results.append((number, ok, what))
    print("\n=== summary ===")
    failed = 0
    for number, ok, what in results:
        mark = "skip" if ok is None else ("ok  " if ok else "FAIL")
        print(f"  {mark}  step {number}: {what}")
        failed += int(ok is False)
    if skip:
        print(f"INCOMPLETE: steps {sorted(skip)} were skipped")
        return 1
    if failed:
        print(f"FAIL: {failed} step(s)")
        return 1
    print("PASS: the merge is acknowledged and the mirrors are verified")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--check", action="store_true", help="run the plan 6.5 steps")
    ap.add_argument("--ack", metavar="SHA", help="acknowledge a merged upstream sha")
    ap.add_argument(
        "--hunk",
        action="append",
        default=[],
        metavar="TEXT",
        help="one line per acknowledged hunk (repeatable; required with --ack)",
    )
    ap.add_argument(
        "--compare-junit",
        nargs=2,
        metavar=("BASELINE", "NEW"),
        help="exit 1 on any test case failing in NEW but not in BASELINE",
    )
    ap.add_argument(
        "--lint-scope",
        choices=("committed",),
        help="ruff check + format --check over the files committed since C1",
    )
    ap.add_argument("--since", help="base revision for --lint-scope / the changelog")
    ap.add_argument("--rings", type=int, default=19, help="oracle rings for step 3")
    ap.add_argument(
        "--skip",
        default="",
        help="comma-separated step numbers to skip (diagnostic; --check then "
        "still exits 1)",
    )
    args = ap.parse_args(argv)

    if args.compare_junit:
        return compare_junit(Path(args.compare_junit[0]), Path(args.compare_junit[1]))
    if args.lint_scope:
        return lint_scope(args.since)
    if args.ack:
        if not args.hunk:
            ap.error("--ack needs at least one --hunk 'what was reviewed'")
        return ack(args.ack, args.hunk)
    if args.check:
        skip = {int(s) for s in args.skip.split(",") if s.strip()}
        return check(skip, args.rings, args.since)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
