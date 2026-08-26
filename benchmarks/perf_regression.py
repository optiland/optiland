"""Relative performance-regression gate for sequential tracing.

Absolute timings are meaningless across machines, so this tool never stores
a baseline. It measures two git refs back to back on the same machine and
compares ratios:

- the base ref (default: the merge base with ``origin/master``), and
- the candidate (default: the currently checked-out tree; in pull-request
  CI that is the would-be merge result of the PR into master).

Both refs are executed from plain source checkouts (git worktrees) against
the single active environment via ``PYTHONPATH``, which works because
optiland is pure Python. Each measurement runs in a fresh interpreter, the
two refs are interleaved to cancel thermal/load drift, and the per-scenario
statistic is the minimum over repeats (the standard robust estimator for
wall-clock timing). The gate fails only when a scenario is slower than
``--fail-ratio`` AND the absolute slowdown exceeds ``--min-delta`` seconds,
so microsecond-scale jitter can never fail CI.

The project policy is zero tolerated regression. A literal 1.00 threshold
would fail on identical code (an A/A run of the same ref against itself
measures ratios up to ~1.015 with this protocol), so the default fail
ratio of 1.05 is that policy expressed with a 3x margin over the measured
noise floor. Recalibrate with an A/A run (--base X --candidate X) when
moving to a noisier machine.

Usage:
    uv run python benchmarks/perf_regression.py                # vs merge base
    uv run python benchmarks/perf_regression.py --base v0.6.0
    uv run python benchmarks/perf_regression.py --base master --candidate HEAD

Limitations: the base ref runs against the candidate's dependency set; a PR
that changes dependencies compares apples to oranges and should say so in
its description.

Kramer Harrison, 2026
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Scenarios. Each builds its inputs, then times only the operation named in
# the key. Builders use long-stable public API only, so the same file can
# measure any recent ref.
# ---------------------------------------------------------------------------


def _spherical_stack():
    import optiland.backend as be
    from optiland.optic import Optic

    optic = Optic(name="bench-spherical")
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    radii = [55.0, -260.0, 34.0, 190.0, -45.0, 28.0, -32.0, -85.0]
    thick = [4.0, 0.5, 5.0, 3.0, 1.0, 6.0, 2.5, 60.0]
    for i, (r, t) in enumerate(zip(radii, thick, strict=True), start=1):
        mat = "N-BK7" if i % 2 == 1 else "air"
        optic.surfaces.add(
            index=i, radius=r, thickness=t, material=mat, is_stop=(i == 4)
        )
    optic.surfaces.add(index=9)
    optic.set_aperture(aperture_type="EPD", value=18.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.fields.add(y=3.0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    return optic


def _oap_collimator():
    import numpy as np

    import optiland.backend as be
    from optiland.optic import Optic
    from optiland.physical_apertures import OffsetRadialAperture

    rfl = 25.4
    optic = Optic(name="bench-oap")
    optic.surfaces.add(index=0, z=0, radius=be.inf)
    optic.surfaces.add(index=1, z=10, is_stop=True)
    optic.surfaces.add(
        index=2,
        z=0,
        y=-rfl / 2,
        radius=-rfl,
        conic=-1,
        aperture=OffsetRadialAperture(r_max=12.7, offset_y=rfl),
        material="mirror",
        rx=np.pi / 2,
    )
    optic.surfaces.add(
        index=3,
        z=rfl,
        y=40,
        radius=be.inf,
        aperture=OffsetRadialAperture(r_max=12.7, offset_y=0),
        rx=np.pi / 2,
    )
    optic.set_aperture(aperture_type="EPD", value=5.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=0.633, is_primary=True)
    return optic


def _explicit_fan(n, angle_max_rad, wavelength):
    import numpy as np

    from optiland.rays import RealRays

    ang = np.linspace(-angle_max_rad, angle_max_rad, n)
    return RealRays(
        np.zeros(n),
        np.linspace(-2.0, 2.0, n),
        np.zeros(n),
        np.zeros(n),
        np.sin(ang),
        np.cos(ang),
        np.ones(n),
        np.full(n, wavelength),
    )


def scenario_surfaces_trace_spherical():
    """Forward propagation through 8 spherical surfaces (no aiming layer)."""
    optic = _spherical_stack()
    n = 180_000

    def op():
        optic.surfaces.trace(_explicit_fan(n, 0.05, 0.55))

    return op


def scenario_surfaces_trace_conic_oap():
    """Forward propagation through the off-axis parabola (conic root path)."""
    optic = _oap_collimator()
    n = 360_000

    def op():
        optic.surfaces.trace(_explicit_fan(n, 0.10, 0.633))

    return op


def scenario_optic_trace_generation():
    """Full optic.trace: field/pupil generation plus propagation."""
    optic = _spherical_stack()

    def op():
        for hy in (0.0, 1.0):
            optic.trace(
                Hx=0.0,
                Hy=hy,
                wavelength=0.55,
                num_rays=60_000,
                distribution="line_y",
            )

    return op


def scenario_iterative_aiming():
    """optic.trace with the iterative Newton aimer on a straight system."""
    optic = _spherical_stack()
    optic.ray_tracer.set_aiming("iterative", max_iter=50, tol=1e-9)

    def op():
        for hy in (0.0, 0.5, 1.0):
            optic.trace(
                Hx=0.0,
                Hy=hy,
                wavelength=0.55,
                num_rays=1024,
                distribution="line_y",
            )

    return op


def scenario_first_order_ops():
    """Repeated scalar first-order queries (path construction cost)."""
    optic = _spherical_stack()

    def op():
        for _ in range(60):
            optic.paraxial.f2()
            optic.paraxial.EPL()
            optic.paraxial.XPL()

    return op


SCENARIOS = {
    "surfaces_trace_spherical": scenario_surfaces_trace_spherical,
    "surfaces_trace_conic_oap": scenario_surfaces_trace_conic_oap,
    "optic_trace_generation": scenario_optic_trace_generation,
    "iterative_aiming": scenario_iterative_aiming,
    "first_order_ops": scenario_first_order_ops,
}


def measure(out_path: str) -> None:
    """Run every scenario once (after one warmup) in this interpreter."""
    import warnings

    warnings.simplefilter("ignore")
    import optiland

    results: dict[str, float] = {}
    for name, factory in SCENARIOS.items():
        op = factory()
        op()  # warmup: caches, imports, first-touch allocations
        t0 = time.perf_counter()
        op()
        results[name] = time.perf_counter() - t0
    payload = {"optiland_file": optiland.__file__, "seconds": results}
    Path(out_path).write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True
    ).strip()


def _run_child(source_dir: Path, out_file: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_dir)
    script = source_dir / "benchmarks" / "perf_regression.py"
    if not script.exists():
        # The base ref predates this tool: use the candidate's copy, which
        # only touches long-stable public API.
        script = Path(__file__).resolve()
    subprocess.check_call(
        [sys.executable, str(script), "--measure", str(out_file)],
        env=env,
        cwd=str(source_dir),
    )


def compare(args: argparse.Namespace) -> int:
    base_ref = args.base or _git("merge-base", "HEAD", "origin/master")
    base_sha = _git("rev-parse", base_ref)
    with tempfile.TemporaryDirectory(prefix="optiland-perf-") as tmp:
        tmp_path = Path(tmp)
        trees: dict[str, Path] = {}
        added_worktrees: list[Path] = []
        try:
            base_tree = tmp_path / "base"
            _git("worktree", "add", "--detach", str(base_tree), base_sha)
            added_worktrees.append(base_tree)
            trees["base"] = base_tree

            if args.candidate:
                cand_tree = tmp_path / "candidate"
                cand_sha = _git("rev-parse", args.candidate)
                _git("worktree", "add", "--detach", str(cand_tree), cand_sha)
                added_worktrees.append(cand_tree)
                trees["candidate"] = cand_tree
                cand_label = cand_sha[:12]
            else:
                trees["candidate"] = REPO_ROOT
                cand_label = "worktree"

            print(f"base: {base_sha[:12]} ({base_ref})   candidate: {cand_label}")
            print(
                f"repeats: {args.repeats} (interleaved), statistic: min, "
                f"fail ratio: {args.fail_ratio:.2f}, "
                f"min delta: {args.min_delta * 1e3:.0f} ms"
            )

            samples: dict[str, dict[str, list[float]]] = {
                role: {name: [] for name in SCENARIOS} for role in trees
            }
            for r in range(args.repeats):
                for role, tree in trees.items():
                    out = tmp_path / f"{role}-{r}.json"
                    _run_child(tree, out)
                    seconds = json.loads(out.read_text())["seconds"]
                    for name in SCENARIOS:
                        samples[role][name].append(seconds[name])

            failures: list[str] = []
            header = (
                f"{'scenario':30s} {'base [ms]':>10s} {'cand [ms]':>10s} {'ratio':>7s}"
            )
            print("\n" + header)
            print("-" * len(header))
            for name in SCENARIOS:
                t_base = min(samples["base"][name])
                t_cand = min(samples["candidate"][name])
                ratio = t_cand / t_base if t_base > 0 else math.inf
                slower = ratio > args.fail_ratio and (t_cand - t_base) > args.min_delta
                mark = (
                    "  FAIL"
                    if slower
                    else ("  warn" if ratio > args.warn_ratio else "")
                )
                print(
                    f"{name:30s} {t_base * 1e3:10.1f} {t_cand * 1e3:10.1f} "
                    f"{ratio:7.3f}{mark}"
                )
                if slower:
                    failures.append(f"{name}: {ratio:.3f}x")

            if failures:
                print(
                    "\nPerformance regression against "
                    f"{base_sha[:12]}: " + ", ".join(failures)
                )
                return 1
            print("\nNo performance regression.")
            return 0
        finally:
            for tree in added_worktrees:
                subprocess.call(
                    [
                        "git",
                        "-C",
                        str(REPO_ROOT),
                        "worktree",
                        "remove",
                        "--force",
                        str(tree),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--measure",
        metavar="OUT_JSON",
        help="internal: time all scenarios in this interpreter",
    )
    parser.add_argument(
        "--base",
        help="git ref to compare against (default: merge base with origin/master)",
    )
    parser.add_argument(
        "--candidate",
        help="git ref to measure as candidate (default: the current tree)",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fail-ratio", type=float, default=1.05)
    parser.add_argument("--warn-ratio", type=float, default=1.02)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.002,
        help="absolute slowdown [s] a scenario must exceed to fail",
    )
    args = parser.parse_args()

    if args.measure:
        measure(args.measure)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
