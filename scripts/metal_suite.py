"""Run the upstream suite under the ``torch-mps`` parametrization, a file per process.

Each test file runs in a fresh interpreter (Metal state and the fast-math flag
are process-global), with a timeout, so a crash or hang in one file cannot take
the run down. Results are aggregated into JSON plus a Markdown summary grouped
by failure signature, which is what the triage loop consumes.

Fused trace (plan 8.2).  ``--grad off`` switches the fork-local ``torch-mps``
fixture to autograd-off, which is what makes a bundle eligible for the kernel,
and ``--fused 0|1`` pins the hook off or on (never ``require``: a suite traces
ineligible systems by design, plan 1.3).  With either flag the per-test census
of ``tests/conftest.py`` is switched on through ``OPTILAND_TEST_MPS_STATS_FILE``
and aggregated here into four sections:

* **Fused trace counters** -- every ``fused_trace:*`` / ``fused_trace_skip:*``
  total of the run;
* **Census** -- the tests violating either identity of plan 1.3
  (``census == candidates`` and
  ``candidates == traces + feature skips + late_fallback``).  Must be empty;
* **Files where the gate never fired** -- files that produced candidate bundles
  but no fused trace;
* **Grad-off vs grad-on** -- the tests that fail only with autograd off, i.e.
  the ones that need it (computed from the grad-on run passed as
  ``--grad-on``, never hand-listed).

Pass criteria per mode (plan 8.2, checked here and reflected in the exit
status): the failure signatures of the fused run are a subset of the grad-off
per-op control's (``--control``), every census identity holds,
``fused_trace_skip:mirror_drift`` is zero, and ``fused_trace:traces > 0`` in
every :data:`ANALYSIS_HEAVY_FILES` entry.

Usage (from the project root)::

    .venv/bin/python Optiland-Metal/scripts/metal_suite.py \
        --out NOTES/suite-torch-mps.json [--files tests/test_geometries.py ...] \
        [--timeout 900] [--jobs 2] [--mode df64|sf64] \
        [--grad on|off] [--fused auto|0|1] \
        [--control NOTES/suite-torch-mps-perop-gradoff-df64.json] \
        [--grad-on NOTES/suite-torch-mps-perop.json]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "Optiland-Metal"
PY = ROOT / ".venv" / "bin" / "python"

#: Files that must exercise the kernel: each traces a bundle above the
#: host-residency threshold somewhere, so ``fused_trace:traces > 0`` is a
#: criterion for them (plan 8.2).  Measured, never assumed: the traces each one
#: produced in the WP8 measurement run (mode df64, ``--grad off --fused 1``)
#: are in the comment beside it.  :data:`ANALYSIS_HEAVY_DROPPED` carries the
#: entries of the plan's list that the same measurement removed.
ANALYSIS_HEAVY_FILES: tuple[str, ...] = (
    "tests/test_analysis.py",  # 578 candidate bundles, 578 fused
    "tests/test_wavefront.py",  # 52 / 52
    "tests/test_fft_psf.py",  # 27 / 27
    "tests/test_mmdft_psf.py",  # 56 / 56
    "tests/test_optic.py",  # 4 / 4
    "tests/test_ray_aiming.py",  # 15 / 15
)

#: Entries of plan 8.2's list dropped from the criterion, with the measurement
#: that dropped them.  Reported in the Markdown so the drop stays on the record.
ANALYSIS_HEAVY_DROPPED: dict[str, str] = {
    "tests/test_psf.py": (
        "no such file in this fork; the PSF tests that are parametrized over "
        "the backend are tests/test_fft_psf.py (30 torch-mps tests, 27 fused "
        "traces) and tests/test_mmdft_psf.py (50 tests, 56 fused traces), "
        "which take its place in the list above"
    ),
    "tests/test_optimization.py": (
        "45 tests, none parametrized over the backend: `-k torch-mps` "
        "deselects all 45, so the file never runs on mps in this harness and "
        "can produce neither a candidate bundle nor a fused trace"
    ),
    "tests/test_tolerancing.py": (
        "8 torch-mps tests, measured census 0: they never call "
        "SurfaceGroup.trace with a GPU-resident bundle above HOST_THRESHOLD "
        "(integration step I2, re-measured by WP8), so fused_trace:traces > 0 "
        "is unreachable there without adding a test"
    ),
}

#: The six structural refusal reasons of plan 1.3, restated here rather than
#: imported so the aggregation stays an independent check of the driver's
#: counters.  Anything else is a feature reason and belongs in identity 2; an
#: unknown reason is therefore counted as a feature reason and makes the
#: identity fail loudly instead of being silently absorbed.
STRUCTURAL_REASONS: frozenset[str] = frozenset(
    {
        "group_type",
        "rays_type",
        "rays_shape",
        "host_resident",
        "requires_grad",
        "skip",
    }
)

_SIG_RE = re.compile(
    r"^(?:E\s+)?([A-Za-z_.]+(?:Error|Exception|Warning|Failed))[:]?\s*"
    r"(.{0,140})"
)
_TAIL_RE = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed)")


def _stats_name(path: str) -> str:
    """The JSONL file name a test file's census lines go to."""
    return path.replace("/", "_").removesuffix(".py") + ".jsonl"


def read_census(path: Path) -> dict[str, Any]:
    """Aggregate one test file's census JSONL (written by ``tests/conftest.py``).

    Returns the per-file totals, the number of tests that produced a line, the
    number of unparsable lines (a line per test is written in a ``finally``
    block, so a crash mid-write shows up here rather than silently), and one
    record per test that violates an identity of plan 1.3.
    """
    out: dict[str, Any] = {
        "tests": 0,
        "unparsable": 0,
        "census": 0,
        "totals": {},
        "violations": [],
    }
    if not path.exists():
        return out
    totals: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            out["unparsable"] += 1
            continue
        out["tests"] += 1
        census = int(rec.get("census", 0))
        out["census"] += census
        stats = {k: int(v) for k, v in rec.get("stats", {}).items()}
        totals.update({k: v for k, v in stats.items() if k.startswith("fused_trace")})
        violation = check_identities(rec.get("nodeid", "?"), census, stats)
        if violation is not None:
            out["violations"].append(violation)
    out["totals"] = dict(sorted(totals.items()))
    return out


def check_identities(nodeid: str, census: int, stats: dict[str, int]) -> dict | None:
    """The two identities of plan 1.3 for one test, or None when both hold."""
    candidates = stats.get("fused_trace:candidates", 0)
    traces = stats.get("fused_trace:traces", 0)
    late = stats.get("fused_trace:late_fallback", 0)
    feature = sum(
        v
        for k, v in stats.items()
        if k.startswith("fused_trace_skip:")
        and k.removeprefix("fused_trace_skip:") not in STRUCTURAL_REASONS
    )
    broken = []
    if census != candidates:
        broken.append(f"census {census} != fused_trace:candidates {candidates}")
    if candidates != traces + feature + late:
        broken.append(
            f"candidates {candidates} != traces {traces} + feature skips "
            f"{feature} + late_fallback {late}"
        )
    if not broken:
        return None
    return {"nodeid": nodeid, "census": census, "stats": stats, "broken": broken}


def check_hook_off(nodeid: str, stats: dict[str, int]) -> dict | None:
    """With the hook off no ``fused_trace*`` counter may exist at all.

    ``OPTILAND_METAL_FUSED_TRACE=0`` makes the hook return before the gate, so
    not even ``candidates`` is counted; the census still counts the same
    bundles, which is what makes the control and the fused run comparable.
    """
    present = {k: v for k, v in stats.items() if k.startswith("fused_trace")}
    if not present:
        return None
    return {
        "nodeid": nodeid,
        "census": 0,
        "stats": present,
        "broken": [f"hook off, but {sorted(present)} were counted"],
    }


def run_file(
    path: str,
    timeout: int,
    mode: str,
    keyword: str,
    grad: str,
    fused: str,
    stats_dir: Path | None,
) -> dict:
    """Run one test file under torch-mps and summarize its outcome."""
    env = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        # A single file run alone would pick the interactive macosx backend,
        # whose ``plt.show()`` blocks (JonesPupil.view) until the timeout.
        MPLBACKEND="Agg",
        PYTORCH_MPS_FAST_MATH="0",
        OPTILAND_TEST_MPS="1",
        OPTILAND_TEST_MPS_GRAD="1" if grad == "on" else "0",
        OPTILAND_METAL_MODE=mode,
        PYTHONUNBUFFERED="1",
    )
    if fused != "auto":
        env["OPTILAND_METAL_FUSED_TRACE"] = fused
    else:
        env.pop("OPTILAND_METAL_FUSED_TRACE", None)
    stats_file = None
    if stats_dir is not None:
        stats_file = stats_dir / _stats_name(path)
        stats_file.unlink(missing_ok=True)
        env["OPTILAND_TEST_MPS_STATS_FILE"] = str(stats_file)
    else:
        env.pop("OPTILAND_TEST_MPS_STATS_FILE", None)
    cmd = [
        str(PY),
        "-m",
        "pytest",
        path,
        "-q",
        "-p",
        "no:cacheprovider",
        "-k",
        keyword,
        "--tb=line",
        "-o",
        "addopts=",
        "-rfE",
    ]
    t0 = time.time()
    rec = {
        "file": path,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "crash": False,
        "timeout": False,
        "signatures": [],
        "failed_tests": [],
        "seconds": 0.0,
        "rc": None,
    }
    try:
        r = subprocess.run(
            cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout
        )
        rec["rc"] = r.returncode
        out = r.stdout
        for line in out.splitlines()[-3:]:
            for count, kind in _TAIL_RE.findall(line):
                key = {"error": "errors", "errors": "errors"}.get(kind, kind)
                if key in rec:
                    rec[key] = int(count)
        for line in out.splitlines():
            if line.startswith(("FAILED ", "ERROR ")):
                rec["failed_tests"].append(line.split(" - ")[0].strip())
            m = _SIG_RE.match(line.strip())
            if m and (line.startswith("E ") or ": " in line):
                rec["signatures"].append(f"{m.group(1)}: {m.group(2)}".strip())
        rec["crash"] = (
            r.returncode < 0 or r.returncode >= 128 or "Fatal Python error" in r.stderr
        )
        if rec["crash"]:
            rec["signatures"].append(
                "CRASH: " + r.stderr.strip().splitlines()[-1][:160]
                if r.stderr.strip()
                else "CRASH"
            )
    except subprocess.TimeoutExpired:
        rec["timeout"] = True
        rec["signatures"].append("TIMEOUT")
    rec["seconds"] = round(time.time() - t0, 1)
    if stats_file is not None:
        census = read_census(stats_file)
        if fused == "0":
            census["violations"] = [
                v
                for v in (
                    check_hook_off(line.get("nodeid", "?"), line.get("stats", {}))
                    for line in _lines(stats_file)
                )
                if v is not None
            ]
        rec["fused"] = census
    return rec


def _lines(path: Path) -> list[dict]:
    """The parsable JSON lines of a census file."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------
# Provenance and aggregation
# ---------------------------------------------------------------------------


def provenance() -> dict[str, str]:
    """Fork HEAD and the mirror table hash stamped into every run (plan 3.7)."""
    out: dict[str, str] = {}
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        out["commit"] = r.stdout.strip()
    except OSError:  # pragma: no cover - git missing
        out["commit"] = ""
    r = subprocess.run(
        [
            str(PY),
            "-c",
            "from optiland.backend.torch_backend.metal.trace_mirror import "
            "table_hash; print(table_hash())",
        ],
        cwd=REPO,
        env=dict(os.environ, PYTORCH_MPS_FAST_MATH="0"),
        capture_output=True,
        text=True,
        check=False,
    )
    out["mirror_table_hash"] = r.stdout.strip() if r.returncode == 0 else ""
    return out


def signatures_of(summary: dict) -> set[str]:
    """Every failure signature of a run summary."""
    return set(summary.get("signatures", {}))


def failed_tests_of(summary: dict) -> set[str]:
    """Every failing test id of a run summary."""
    return {t for r in summary.get("results", []) for t in r.get("failed_tests", [])}


def fused_summary(results: list[dict], fused: str = "auto") -> dict[str, Any]:
    """The four sections of plan 8.2 over the per-file census records.

    With the hook off every file with a candidate bundle would trivially be a
    file "where the gate never fired", so that section is empty by
    construction there and the census totals are what the control run
    contributes (they must equal the fused run's, which is how the two runs
    are shown to have seen the same work).
    """
    totals: Counter[str] = Counter()
    violations: list[dict] = []
    per_file: dict[str, dict[str, int]] = {}
    tests = unparsable = census_total = 0
    for rec in results:
        census = rec.get("fused")
        if census is None:
            continue
        totals.update(census["totals"])
        violations += [dict(v, file=rec["file"]) for v in census["violations"]]
        tests += census["tests"]
        unparsable += census["unparsable"]
        census_total += census["census"]
        per_file[rec["file"]] = {
            "census": census["census"],
            "candidates": census["totals"].get("fused_trace:candidates", 0),
            "traces": census["totals"].get("fused_trace:traces", 0),
            "surface_steps": census["totals"].get("fused_trace:surface_steps", 0),
        }
    silent = (
        sorted(f for f, v in per_file.items() if v["census"] > 0 and v["traces"] == 0)
        if fused != "0"
        else []
    )
    heavy = {
        f: per_file.get(f, {}).get("traces", None)
        for f in ANALYSIS_HEAVY_FILES
        if f in per_file
    }
    return {
        "totals": dict(sorted(totals.items())),
        "tests_with_census": tests,
        "unparsable_lines": unparsable,
        "census_total": census_total,
        "violations": violations,
        "per_file": per_file,
        "files_gate_never_fired": silent,
        "analysis_heavy": heavy,
        "mirror_drift": totals.get("fused_trace_skip:mirror_drift", 0),
    }


def criteria(
    summary: dict, fused: str, control: dict | None
) -> list[tuple[str, bool, str]]:
    """The plan 8.2 pass criteria as ``(name, ok, detail)`` rows."""
    rows: list[tuple[str, bool, str]] = []
    fu = summary.get("fused_trace")
    if fu is None:
        return rows
    rows.append(
        (
            "census identities",
            not fu["violations"],
            f"{len(fu['violations'])} violating test(s) of "
            f"{fu['tests_with_census']} with a census line",
        )
    )
    rows.append(
        (
            "census lines parsable",
            fu["unparsable_lines"] == 0,
            f"{fu['unparsable_lines']} unparsable line(s)",
        )
    )
    if fused != "0":
        rows.append(
            (
                "no mirror drift",
                fu["mirror_drift"] == 0,
                f"fused_trace_skip:mirror_drift = {fu['mirror_drift']}",
            )
        )
        missing = sorted(f for f, t in fu["analysis_heavy"].items() if not t)
        rows.append(
            (
                "fused_trace:traces > 0 in every ANALYSIS_HEAVY_FILES entry run",
                not missing,
                f"{len(fu['analysis_heavy'])} of {len(ANALYSIS_HEAVY_FILES)} entries "
                f"in this run; without a fused trace: {missing or 'none'}",
            )
        )
    if control is not None:
        mine, theirs = signatures_of(summary), signatures_of(control)
        extra = sorted(mine - theirs)
        my_tests = failed_tests_of(summary)
        their_tests = failed_tests_of(control)
        new_tests = sorted(my_tests - their_tests)
        rows.append(
            (
                "failure signatures subset of the control",
                not extra,
                f"{len(extra)} signature(s) only in this run: {extra[:4] or 'none'}",
            )
        )
        rows.append(
            (
                "no test fails only in this run",
                not new_tests,
                f"{len(new_tests)} test(s): {new_tests[:4] or 'none'}",
            )
        )
    return rows


def write_markdown(
    path: Path,
    summary: dict,
    args: argparse.Namespace,
    control: dict | None,
    grad_on: dict | None,
    rows: list[tuple[str, bool, str]],
) -> None:
    """The Markdown companion of the JSON summary."""
    tot = summary["totals"]
    md = [
        f"# torch-mps suite ({args.mode}, grad {args.grad}, fused {args.fused})",
        "",
        f"Totals: {tot}",
        "",
        f"Provenance: commit `{summary['provenance'].get('commit', '')}`, "
        f"mirror table `{summary['provenance'].get('mirror_table_hash', '')[:16]}`",
        "",
        "## Failure signatures (files)",
        "",
    ]
    for s, fs in summary["signatures"].items():
        names = ", ".join(Path(f).name for f in fs[:8])
        more = " ..." if len(fs) > 8 else ""
        md.append(f"- `{s}` ({len(fs)} files): {names}{more}")
    md += ["", "## Files with failures", ""]
    for r in summary["results"]:
        if r["failed"] or r["errors"] or r["crash"] or r["timeout"]:
            md.append(
                f"- {r['file']}: failed={r['failed']} errors={r['errors']} "
                f"crash={r['crash']} timeout={r['timeout']}"
            )
    fu = summary.get("fused_trace")
    if fu is not None:
        md += ["", "## Fused trace counters", ""]
        if fu["totals"]:
            for k, v in fu["totals"].items():
                md.append(f"- `{k}`: {v}")
        else:
            md.append("- (none: the hook was off for this run)")
        md.append(
            f"- independent census of candidate bundles: {fu['census_total']} "
            f"over {fu['tests_with_census']} test(s)"
        )
        md += ["", "## Census", ""]
        if fu["violations"]:
            for v in fu["violations"]:
                md.append(f"- `{v['nodeid']}`: {'; '.join(v['broken'])}")
        else:
            md.append(
                "- no test violates either identity of plan 1.3 "
                f"({fu['tests_with_census']} tests, {fu['unparsable_lines']} "
                "unparsable census lines)"
            )
        md += ["", "## Files where the gate never fired", ""]
        if args.fused == "0":
            md.append(
                "- not applicable: the hook was off for this run, so no file "
                f"fused anything ({fu['census_total']} candidate bundles were "
                "counted by the census, which the fused run must match)"
            )
        elif fu["files_gate_never_fired"]:
            for f in fu["files_gate_never_fired"]:
                per = fu["per_file"][f]
                md.append(
                    f"- {f}: census {per['census']} candidate bundle(s), "
                    f"fused_trace:traces 0"
                )
        else:
            md.append("- every file with a candidate bundle fused at least one")
        md += ["", "### ANALYSIS_HEAVY_FILES", ""]
        for f in ANALYSIS_HEAVY_FILES:
            traces = fu["analysis_heavy"].get(f)
            state = "not in this run" if traces is None else f"traces {traces}"
            md.append(f"- {f}: {state}")
        for f, why in ANALYSIS_HEAVY_DROPPED.items():
            md.append(f"- ~~{f}~~ dropped from the criterion: {why}")
    md += ["", "## Grad-off vs grad-on", ""]
    if grad_on is None:
        md.append(
            "- not computed: pass `--grad-on <the grad-on run's JSON>` to list "
            "the tests that need autograd"
        )
    else:
        need_grad = sorted(failed_tests_of(summary) - failed_tests_of(grad_on))
        md.append(
            f"- {len(need_grad)} test(s) fail here but not in the grad-on run, "
            "i.e. they need autograd:"
        )
        md += [f"  - `{t}`" for t in need_grad]
        gone = sorted(failed_tests_of(grad_on) - failed_tests_of(summary))
        md.append(f"- {len(gone)} test(s) fail only in the grad-on run: {gone[:8]}")
    if rows:
        md += ["", "## Pass criteria (plan 8.2)", ""]
        for name, ok, detail in rows:
            md.append(f"- {'PASS' if ok else 'FAIL'} — {name}: {detail}")
        if control is None:
            md.append(
                "- (the subset criterion needs `--control <the grad-off per-op "
                "run's JSON>`)"
            )
    path.write_text("\n".join(md) + "\n")


def _load(path: str | None) -> dict | None:
    """A previously written run summary, or None."""
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="test files relative to Optiland-Metal (default: all except gui/metal)",
    )
    ap.add_argument("--out", default=str(ROOT / "NOTES" / "suite-torch-mps.json"))
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--jobs", type=int, default=2, help="concurrent files (they share the GPU)"
    )
    ap.add_argument("--mode", default="df64")
    ap.add_argument("--keyword", default="torch-mps", help="pytest -k expression")
    ap.add_argument(
        "--grad",
        choices=("on", "off"),
        default=None,
        help="OPTILAND_TEST_MPS_GRAD (default 'on'); 'off' is what makes "
        "bundles eligible",
    )
    ap.add_argument(
        "--fused",
        choices=("auto", "0", "1"),
        default=None,
        help="OPTILAND_METAL_FUSED_TRACE (default 'auto'; never 'require' for "
        "a suite, plan 8.2)",
    )
    ap.add_argument(
        "--stats",
        choices=("auto", "on", "off"),
        default="auto",
        help="per-test census JSONL; 'auto' means on as soon as --grad or "
        "--fused is given, so the historical invocation is unchanged",
    )
    ap.add_argument(
        "--control",
        default=None,
        help="the grad-off per-op run's JSON; enables the subset criterion",
    )
    ap.add_argument(
        "--grad-on",
        default=None,
        help="the grad-on run's JSON; enables the '## Grad-off vs grad-on' section",
    )
    args = ap.parse_args()

    # The historical invocation passes neither flag and must stay unchanged:
    # the census fixture wraps SurfaceGroup.trace, so it is switched on only
    # once the run is about the fused path.
    explicit = args.grad is not None or args.fused is not None
    args.grad = args.grad or "on"
    args.fused = args.fused or "auto"
    want_stats = args.stats == "on" or (args.stats == "auto" and explicit)
    out_path = Path(args.out)
    stats_dir = out_path.with_suffix(".stats") if want_stats else None
    if stats_dir is not None:
        stats_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        files = args.files
    else:
        files = sorted(
            str(Path(p).relative_to(REPO))
            for p in glob.glob(str(REPO / "tests" / "**" / "test_*.py"), recursive=True)
        )
        files = [f for f in files if "/gui/" not in f and "/metal/" not in f]
    print(
        f"{len(files)} files, {args.jobs} concurrent, mode={args.mode}, "
        f"grad={args.grad}, fused={args.fused}, "
        f"census={'on' if want_stats else 'off'}",
        flush=True,
    )
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {
            ex.submit(
                run_file,
                f,
                args.timeout,
                args.mode,
                args.keyword,
                args.grad,
                args.fused,
                stats_dir,
            ): f
            for f in files
        }
        for fut in cf.as_completed(futs):
            rec = fut.result()
            results.append(rec)
            flag = "CRASH" if rec["crash"] else "TIMEOUT" if rec["timeout"] else ""
            fused_note = ""
            if rec.get("fused"):
                fused_note = (
                    f" census={rec['fused']['census']:5d} "
                    f"traces={rec['fused']['totals'].get('fused_trace:traces', 0):5d}"
                )
            print(
                f"{rec['file']:60s} pass={rec['passed']:4d} fail={rec['failed']:3d} "
                f"err={rec['errors']:3d} skip={rec['skipped']:3d} "
                f"{rec['seconds']:6.1f}s {flag}{fused_note}",
                flush=True,
            )
    results.sort(key=lambda r: r["file"])
    tot = Counter()
    for r in results:
        for k in ("passed", "failed", "errors", "skipped"):
            tot[k] += r[k]
        tot["crash"] += int(r["crash"])
        tot["timeout"] += int(r["timeout"])
    sigs: dict[str, list[str]] = defaultdict(list)
    for r in results:
        for s in set(r["signatures"]):
            sigs[s].append(r["file"])
    summary: dict[str, Any] = {
        "mode": args.mode,
        "grad": args.grad,
        "fused": args.fused,
        "provenance": provenance(),
        "totals": dict(tot),
        "results": results,
        "signatures": {
            k: v for k, v in sorted(sigs.items(), key=lambda kv: -len(kv[1]))
        },
    }
    if want_stats:
        summary["fused_trace"] = fused_summary(results, args.fused)
    control, grad_on = _load(args.control), _load(args.grad_on)
    rows = criteria(summary, args.fused, control)
    summary["criteria"] = [{"name": n, "passed": ok, "detail": d} for n, ok, d in rows]
    out_path.write_text(json.dumps(summary, indent=1))
    write_markdown(out_path.with_suffix(".md"), summary, args, control, grad_on, rows)
    print("TOTALS", dict(tot))
    if "fused_trace" in summary:
        fu = summary["fused_trace"]
        print(
            f"CENSUS {fu['census_total']} candidate bundles, "
            f"{len(fu['violations'])} identity violation(s), "
            f"{fu['unparsable_lines']} unparsable line(s)"
        )
        print("COUNTERS", fu["totals"])
    for name, ok, detail in rows:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return 0 if all(ok for _, ok, _ in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
