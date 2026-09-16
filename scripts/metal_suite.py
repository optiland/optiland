"""Run the upstream suite under the ``torch-mps`` parametrization, a file per process.

Each test file runs in a fresh interpreter (Metal state and the fast-math flag
are process-global), with a timeout, so a crash or hang in one file cannot take
the run down. Results are aggregated into JSON plus a Markdown summary grouped
by failure signature, which is what the triage loop consumes.

Usage (from the project root)::

    .venv/bin/python Optiland-Metal/scripts/metal_suite.py \
        --out NOTES/suite-torch-mps.json [--files tests/test_geometries.py ...] \
        [--timeout 900] [--jobs 2] [--mode df64|sf64]
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

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "Optiland-Metal"
PY = ROOT / ".venv" / "bin" / "python"

_SIG_RE = re.compile(
    r"^(?:E\s+)?([A-Za-z_.]+(?:Error|Exception|Warning|Failed))[:]?\s*"
    r"(.{0,140})"
)
_TAIL_RE = re.compile(r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed)")


def run_file(path: str, timeout: int, mode: str, keyword: str) -> dict:
    """Run one test file under torch-mps and summarize its outcome."""
    env = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        # A single file run alone would pick the interactive macosx backend,
        # whose ``plt.show()`` blocks (JonesPupil.view) until the timeout.
        MPLBACKEND="Agg",
        PYTORCH_MPS_FAST_MATH="0",
        OPTILAND_TEST_MPS="1",
        OPTILAND_METAL_MODE=mode,
        PYTHONUNBUFFERED="1",
    )
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
    return rec


def main() -> int:
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
    args = ap.parse_args()

    if args.files:
        files = args.files
    else:
        files = sorted(
            str(Path(p).relative_to(REPO))
            for p in glob.glob(str(REPO / "tests" / "**" / "test_*.py"), recursive=True)
        )
        files = [f for f in files if "/gui/" not in f and "/metal/" not in f]
    print(f"{len(files)} files, {args.jobs} concurrent, mode={args.mode}", flush=True)
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {
            ex.submit(run_file, f, args.timeout, args.mode, args.keyword): f
            for f in files
        }
        for fut in cf.as_completed(futs):
            rec = fut.result()
            results.append(rec)
            flag = "CRASH" if rec["crash"] else "TIMEOUT" if rec["timeout"] else ""
            print(
                f"{rec['file']:60s} pass={rec['passed']:4d} fail={rec['failed']:3d} "
                f"err={rec['errors']:3d} skip={rec['skipped']:3d} "
                f"{rec['seconds']:6.1f}s {flag}",
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
    summary = {
        "totals": dict(tot),
        "results": results,
        "signatures": {
            k: v for k, v in sorted(sigs.items(), key=lambda kv: -len(kv[1]))
        },
    }
    Path(args.out).write_text(json.dumps(summary, indent=1))
    md = [
        f"# torch-mps suite ({args.mode})",
        "",
        f"Totals: {dict(tot)}",
        "",
        "## Failure signatures (files)",
        "",
    ]
    for s, fs in summary["signatures"].items():
        names = ", ".join(Path(f).name for f in fs[:8])
        more = " ..." if len(fs) > 8 else ""
        md.append(f"- `{s}` ({len(fs)} files): {names}{more}")
    md += ["", "## Files with failures", ""]
    for r in results:
        if r["failed"] or r["errors"] or r["crash"] or r["timeout"]:
            md.append(
                f"- {r['file']}: failed={r['failed']} errors={r['errors']} "
                f"crash={r['crash']} timeout={r['timeout']}"
            )
    Path(args.out).with_suffix(".md").write_text("\n".join(md) + "\n")
    print("TOTALS", dict(tot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
