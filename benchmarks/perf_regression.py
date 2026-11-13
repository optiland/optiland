"""Relative performance-regression gate for sequential tracing.

Absolute timings are meaningless across machines, so this tool never stores
a baseline. It measures two git refs back to back on the same machine and
compares ratios:

- the base ref (default: the merge base with ``origin/master``), and
- the candidate (default: the currently checked-out tree; in pull-request
  CI that is the would-be merge result of the PR into master).

Both refs are executed from plain source checkouts (git worktrees) against
the single active environment via ``PYTHONPATH``, which works because
optiland is pure Python. Both refs run this copy of the script, so the two
sides time the same scenario definitions.

Each repeat runs each ref once, in a fresh interpreter. The order alternates
between repeats, so slow drift and the position in a repeat do not favor
one ref. Each interpreter runs each scenario once untimed as a warmup.
Each timed call starts after ``gc.collect()`` and ``gc.freeze()``: the
cyclic garbage collector then scans only the objects that the call
creates, so the objects that earlier scenarios left alive do not add a
collection to the call. The statistic is the minimum
over all timed calls of all repeats. Timing noise only adds time, so the
minimum is the best estimate of the cost without noise, but it does not
show a cost that occurs in only some calls.

A scenario fails only when its ratio (candidate / base) exceeds
``--fail-ratio`` and its slowdown exceeds ``--min-delta`` seconds, so
microsecond jitter cannot fail CI. For a base time below
min_delta / (fail_ratio - 1), 40 ms with the defaults, the min delta and
not the ratio sets the threshold: a 10 ms scenario fails only above 1.20.
Give a scenario enough work for more than 40 ms per timed call.

Timings on shared CI runners are noisy. In two gate runs of the same code
on GitHub runners, surfaces_trace_conic_oap gave ratios of 1.22 and 0.80.
A scenario that is slower than both limits therefore runs again, together
with the other slower scenarios only, with ``--confirm-repeats`` repeats
(default 10). It fails only if this confirmation run shows it slower too.
A real regression is slower in both runs. A small regression can still
pass on CI, so measure a performance change on a quiet local machine too.

The project policy is zero tolerated regression. A literal 1.00 threshold
would fail on identical code, because an A/A run (the same ref as base and
candidate) gives ratios above and below 1.00. The fail ratio must stay
above the largest A/A ratio, the noise floor. Recalibrate with an A/A run
(--base X --candidate X) when moving to a noisier machine.

In the table, FAIL marks a scenario that is slower than both limits, and
the gate exits with 1. warn marks a ratio above ``--warn-ratio`` that does
not fail. ERROR marks a scenario that raised on the candidate. It fails
the gate. n/a marks a scenario that is not compared; "n/a (base raised)"
marks one that raised on the base ref, for example because it needs API
that the base does not have. An error message follows its row. Timing
noise can also cause a FAIL or warn, so run the flagged scenarios again
with ``--only`` and more repeats before you report one.

Usage:
    uv run python benchmarks/perf_regression.py                # vs merge base
    uv run python benchmarks/perf_regression.py --base v0.6.0
    uv run python benchmarks/perf_regression.py --base master --candidate HEAD
    uv run python benchmarks/perf_regression.py --only iterative_aiming --repeats 10

Limitations: the base ref runs against the candidate's dependency set; a PR
that changes dependencies compares apples to oranges and should say so in
its description.

Kramer Harrison, 2026
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The result of one child for one scenario: the time in seconds, the error
# message if the scenario raised, or None if the child has no result for it.
Sample = float | str | None

# ---------------------------------------------------------------------------
# Scenarios. Each builds its inputs, then times only the operation named in
# the key. Builders use public API where they can, so that the file can
# measure older refs. A builder that needs newer or private API shows n/a on
# a base ref that lacks it.
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


def scenario_optimization_merit():
    """Merit evaluations of a small optimization problem (issue #934).

    One evaluation runs one first-order operand and nine traces, as an
    optimizer does for each function evaluation. Per-call costs that are
    small alone add up here.
    """
    from optiland.optimization import OptimizationProblem
    from optiland.samples.objectives import CookeTriplet

    optic = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_operand("f2", target=50.0, weight=1, input_data={"optic": optic})
    for hy in (0.0, 0.7, 1.0):
        for wavelength in optic.wavelengths.get_wavelengths():
            problem.add_operand(
                "rms_spot_size",
                target=0.0,
                weight=10,
                input_data={
                    "optic": optic,
                    "surface_number": -1,
                    "Hx": 0.0,
                    "Hy": hy,
                    "num_rays": 5,
                    "wavelength": wavelength,
                    "distribution": "hexapolar",
                },
            )

    def op():
        for _ in range(4):
            problem.sum_squared()

    return op


SCENARIOS = {
    "surfaces_trace_spherical": scenario_surfaces_trace_spherical,
    "surfaces_trace_conic_oap": scenario_surfaces_trace_conic_oap,
    "optic_trace_generation": scenario_optic_trace_generation,
    "iterative_aiming": scenario_iterative_aiming,
    "first_order_ops": scenario_first_order_ops,
    "optimization_merit": scenario_optimization_merit,
}


def _selected(only: str | None) -> list[str]:
    """Return the scenarios named in ``--only``, or all if it is empty."""
    if not only:
        return list(SCENARIOS)
    names = [n.strip() for n in only.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenarios: {', '.join(unknown)}")
    return names


# The child cuts error messages to this length, so that the table stays short.
_MAX_ERROR_CHARS = 300


def measure(out_path: str, names: list[str]) -> None:
    """Time the selected scenarios in this interpreter and write the results as JSON.

    Each scenario runs once as a warmup and then once timed. The timed call
    starts after ``gc.collect()`` and ``gc.freeze()``. Without them, the
    allocations of all earlier scenarios decide when the cyclic garbage
    collector runs, and a collection scans every object alive in this
    interpreter: about 30 ms with all scenarios built. A change in one part
    of the library could then add or remove a collection in a scenario of
    another part. After the freeze, a collection in the call scans only the
    objects that the call creates. ``gc.unfreeze()`` after each scenario
    lets the next ``gc.collect()`` free them again. A scenario that raises
    gets its error message instead of a time.

    Args:
        out_path: Path of the JSON file to write.
        names: Scenarios to time, in this order.
    """
    import warnings

    warnings.simplefilter("ignore")
    import optiland

    results: dict[str, Sample] = {}
    for name in names:
        try:
            op = SCENARIOS[name]()
            op()  # warmup: caches, imports, first-touch allocations
            gc.collect()
            gc.freeze()
            t0 = time.perf_counter()
            op()
            results[name] = time.perf_counter() - t0
        except Exception as exc:
            results[name] = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]
        finally:
            gc.unfreeze()
    payload = {"optiland_file": optiland.__file__, "seconds": results}
    Path(out_path).write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True
    ).strip()


def _run_child(source_dir: Path, out_file: Path, names: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_dir)
    # Both refs run this copy of the script, so they time the same scenarios.
    script = Path(__file__).resolve()
    only = ",".join(names)
    subprocess.check_call(
        [sys.executable, str(script), "--measure", str(out_file), "--only", only],
        env=env,
        cwd=str(source_dir),
    )


def _collect_samples(
    trees: dict[str, Path], names: list[str], repeats: int, tmp_path: Path
) -> dict[str, dict[str, list[Sample]]]:
    """Run one child per ref and repeat; return the samples per ref and scenario."""
    samples: dict[str, dict[str, list[Sample]]] = {
        role: {name: [] for name in names} for role in trees
    }
    roles = list(trees)
    for r in range(repeats):
        # The order alternates between repeats, so slow drift and the position
        # in a repeat do not favor one ref.
        pair_order = roles if r % 2 == 0 else roles[::-1]
        for role in pair_order:
            out = tmp_path / f"{role}-{r}.json"
            _run_child(trees[role], out, names)
            seconds = json.loads(out.read_text())["seconds"]
            for name in names:
                samples[role][name].append(seconds.get(name))
    return samples


def _report(
    samples: dict[str, dict[str, list[Sample]]],
    names: list[str],
    args: argparse.Namespace,
) -> list[str]:
    """Print the comparison table; return one entry per failed scenario."""
    failures: list[str] = []
    header = f"{'scenario':30s} {'base [ms]':>10s} {'cand [ms]':>10s} {'ratio':>7s}"
    print("\n" + header)
    print("-" * len(header))
    for name in names:
        base, cand = samples["base"][name], samples["candidate"][name]
        cand_error = next((v for v in cand if isinstance(v, str)), None)
        base_error = next((v for v in base if isinstance(v, str)), None)
        if cand_error is not None:
            print(f"{name:30s} {'':>10s} {'ERROR':>10s}")
            print(f"    {cand_error}")
            failures.append(f"{name}: error")
            continue
        if base_error is not None:
            print(f"{name:30s} {'n/a (base raised)':>21s}")
            print(f"    {base_error}")
            continue
        if not all(isinstance(v, float) for v in base + cand):
            print(f"{name:30s} {'n/a':>10s} {'n/a':>10s}")
            continue
        t_base = min(base)
        t_cand = min(cand)
        ratio = t_cand / t_base if t_base > 0 else math.inf
        slower = ratio > args.fail_ratio and (t_cand - t_base) > args.min_delta
        mark = "  FAIL" if slower else ("  warn" if ratio > args.warn_ratio else "")
        print(
            f"{name:30s} {t_base * 1e3:10.1f} {t_cand * 1e3:10.1f} {ratio:7.3f}{mark}"
        )
        if slower:
            failures.append(f"{name}: {ratio:.3f}x")
    return failures


def compare(args: argparse.Namespace, names: list[str]) -> int:
    """Measure base and candidate refs interleaved; return the exit code.

    Args:
        args: Parsed command-line arguments.
        names: Scenarios to measure.
    """
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
                f"repeats: {args.repeats} (alternating order), statistic: min, "
                f"fail ratio: {args.fail_ratio:.2f}, "
                f"min delta: {args.min_delta * 1e3:.0f} ms"
            )
            print(f"scenarios: {', '.join(names)}")

            samples = _collect_samples(trees, names, args.repeats, tmp_path)
            failures = _report(samples, names, args)
            slower = [f.split(":")[0] for f in failures if not f.endswith(": error")]
            if slower and args.confirm_repeats > 0:
                # A shared CI runner can be slower for part of a run. A real
                # regression is slower again, so a scenario fails only if a
                # second run of the slower scenarios alone confirms it.
                print(
                    f"\nConfirmation run of {len(slower)} slower scenario(s): "
                    f"{args.confirm_repeats} repeats"
                )
                samples = _collect_samples(
                    trees, slower, args.confirm_repeats, tmp_path
                )
                confirmed = _report(samples, slower, args)
                failures = [f for f in failures if f.endswith(": error")] + confirmed
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
    """Parse arguments and dispatch to measure or compare mode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--measure",
        metavar="OUT_JSON",
        help="internal: time the selected scenarios in this interpreter",
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
    parser.add_argument(
        "--confirm-repeats",
        type=int,
        default=10,
        help="repeats of the confirmation run of the slower scenarios (0: none)",
    )
    parser.add_argument(
        "--only",
        metavar="NAMES",
        help="comma-separated scenarios to run (default: all), for example "
        "to re-run a flagged scenario with more repeats",
    )
    parser.add_argument("--fail-ratio", type=float, default=1.05)
    parser.add_argument("--warn-ratio", type=float, default=1.02)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.002,
        help="absolute slowdown [s] a scenario must exceed to fail",
    )
    args = parser.parse_args()
    names = _selected(args.only)  # rejects unknown names before any checkout

    if args.measure:
        measure(args.measure, names)
        return 0
    return compare(args, names)


if __name__ == "__main__":
    sys.exit(main())
