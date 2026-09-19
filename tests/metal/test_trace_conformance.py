"""WP6-b: conformance over the golden systems, the shipped catalog, the xfail sites.

Plan 7.2's last four rows, at the tier-A sampling of plan 7.1:

* ``test_golden_systems_fused`` -- the four builders of
  ``tests/regression/systems.py``, the systems whose NumPy and torch-CPU
  golden values the regression harness pins.  An eligible builder is compared
  fused vs per-op under the tier-A rule; an ineligible one must be refused
  with its exact ``FusedTraceSkip`` reason and must still produce the per-op
  result [fix: L2.10].
* ``test_catalog`` -- all 29 shipped sample systems x df64/sf64 x three
  fields, tier A.  This is the row that backs the plan's preface claim that
  v1 "covers all 29 shipped sample systems end to end".
* ``test_xfail_sites_pass_in_sf64`` -- the two historical
  ``xfail_if_emulated_df64`` sites, run in a subprocess in sf64 with the hook
  on: sf64 is exact, so both must PASS rather than xfail.

Sampling (plan 7.1): the hexapolar bundle of 19 rings, N = 1141 > 1024, the
tier-A floor.  ``hexapolar_pupil`` reproduces
``optiland.distribution.HexagonalDistribution.generate_points`` in NumPy so
that one bundle definition serves both backends and both modes;
``test_hexapolar_sampling_is_the_shipped_distribution`` pins that it really is
the shipped distribution and not a lookalike.

The launch bundle is generated ONCE per (system, mode, field) and cloned
component-wise before each trace, which is what plan 7.1's R1 prescribes
("identical launch rays: generate once, clone the ``RealRays`` components
before each trace").  It also keeps the ray aimer -- the dominant cost on
``WideAngle170FOV`` and the ``ProjectionLens*`` systems -- out of the
comparison: a bundle aimed twice, once per hook setting, would not be the same
bundle if the aimer's own traces ever routed differently.

Both traces run on ONE ``SurfaceGroup``, per-op first and fused second, so the
fused trace also has to survive the state the per-op trace left behind
(``_distance_capability`` caches, recorded rows, ``reset()``).

``_trace_compare`` (WP6-a) supplies the capture, the clone and the tier-A
rule; nothing in this file re-implements a tolerance.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import warnings  # noqa: E402
from typing import TYPE_CHECKING, Any  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():  # pragma: no cover - hardware gate
    pytest.skip("Metal GPU required", allow_module_level=True)

import optiland.backend as be  # noqa: E402
from optiland.backend.torch_backend import metal  # noqa: E402
from optiland.backend.torch_backend.metal import tensor as T  # noqa: E402
from optiland.backend.torch_backend.metal import trace  # noqa: E402
from optiland.backend.torch_backend.metal.trace_adapters import (  # noqa: E402
    FusedTraceSkip,
)

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
import trace_fixtures as fx  # noqa: E402

from tests.metal import _trace_compare as tc  # noqa: E402
from tests.regression.systems import GOLDEN_SYSTEMS  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

MODES = tc.MODES

#: Plan 7.1's tier-A sampling: 19 hexapolar rings, 1 + 3*19*20 = 1141 rays.
RINGS = 19
N_RAYS = 1 + 3 * RINGS * (RINGS + 1)

#: Plan 7.1's field list.  ``(0.7, 0)`` is added only for tilted or x-field
#: systems; no shipped sample and no golden builder is tilted (measured by
#: ``trace_fixtures.audit_catalog``: every catalog pose is finite and every
#: catalog system is rotationally symmetric about z), so the y fields are the
#: sampling here and the x field is covered by ``test_trace_kernel``'s
#: ``tilted_triplet`` and ``tilted_fold_mirror`` rows.
FIELDS: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 0.7), (0.0, 1.0))


def hexapolar_pupil(rings: int = RINGS) -> tuple[np.ndarray, np.ndarray]:
    """``HexagonalDistribution.generate_points(rings)`` in plain NumPy.

    Same expressions, same order, in float64: a ring radius
    ``linspace(0, 1, rings + 1)[i + 1]`` times ``cos`` / ``sin`` of
    ``linspace(0, 2 pi, 6 (i + 1) + 1)[:-1]``, prefixed by the centre point.
    Written out here so the same bundle definition is available on every
    backend without routing a distribution through ``be.*`` (which would build
    the points in the active mode and make the NumPy and Metal bundles
    different numbers).  ``test_hexapolar_sampling_is_the_shipped_distribution``
    pins the equality.
    """
    xs = [np.zeros(1)]
    ys = [np.zeros(1)]
    r = np.linspace(0.0, 1.0, rings + 1)
    for i in range(rings):
        num_theta = 6 * (i + 1)
        theta = np.linspace(0.0, 2.0 * np.pi, num_theta + 1)[:-1]
        xs.append(r[i + 1] * np.cos(theta))
        ys.append(r[i + 1] * np.sin(theta))
    return np.concatenate(xs), np.concatenate(ys)


PX, PY = hexapolar_pupil()


def test_hexapolar_sampling_is_the_shipped_distribution():
    """:func:`hexapolar_pupil` is ``create_distribution("hexapolar")``, exactly.

    Compared on the NumPy backend, where ``be.*`` is float64 NumPy, so the
    comparison is bit-for-bit and not a tolerance.  A drift in the shipped
    distribution would otherwise silently change what "tier-A sampling" means
    in this file.
    """
    from optiland.distribution import create_distribution

    be.set_backend("numpy")
    distribution = create_distribution("hexapolar")
    distribution.generate_points(RINGS)
    assert np.array_equal(np.asarray(be.to_numpy(distribution.x)), PX)
    assert np.array_equal(np.asarray(be.to_numpy(distribution.y)), PY)
    assert PX.size == N_RAYS == 1141
    assert N_RAYS > tc.TIER_A_MIN_RAYS, "tier A needs N > 1024 (plan 7.1)"


# ---------------------------------------------------------------------------
# Environment and backend
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _driver_state():
    """Zero the counters and the driver's one-shot state around every test."""
    trace.reset_driver_state()
    T.reset_stats()
    yield
    trace.reset_driver_state()
    T.reset_stats()


@pytest.fixture
def mps_backend():
    """torch / mps / float64 with autograd off -- the fused path's setting.

    ``torch.no_grad()`` is load-bearing: ``be.grad_mode.disable()`` only
    clears the flag new arrays are created with, and with autograd merely
    enabled ``NewtonRaphsonGeometry.distance`` takes the DiffOptics branch and
    returns one extra refinement step instead of the primal root the kernel
    mirrors (WP6-a finding 1, ``test_trace_kernel.mps_backend``).
    """
    previous = metal.get_mode()
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.disable()
    with torch.no_grad():
        assert not torch.is_grad_enabled()
        yield
    metal.set_mode(previous)
    be.grad_mode.disable()
    be.set_backend("numpy")


# ---------------------------------------------------------------------------
# Building, launching, comparing
# ---------------------------------------------------------------------------


def launch_bundle(optic: Any, field: tuple[float, float]) -> Any:
    """The 1141-ray hexapolar bundle of ``optic`` at ``field``.

    Built through the optic's own ray generator, so a system with a ray aimer
    (``WideAngle170FOV``, the ``ProjectionLens*`` pair) is aimed exactly as
    ``optic.trace`` would aim it.  Called ONCE per comparison.
    """
    generator = optic.ray_tracer.ray_generator
    return generator.generate_rays(
        be.array(np.full(N_RAYS, field[0])),
        be.array(np.full(N_RAYS, field[1])),
        be.array(PX),
        be.array(PY),
        optic.primary_wavelength,
    )


def _trace(group: Any, rays: Any, mode: str) -> tc.Capture:
    """Trace a clone of ``rays`` through ``group`` and capture the raw words."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = group.trace(tc.copy_rays(rays))
    return tc.capture(group, out, mode)


def compare(
    builder: Callable[[], Any],
    field: tuple[float, float],
    mode: str,
    monkeypatch,
) -> tuple[Any, tc.Capture, tc.Capture, dict[str, int]]:
    """``(optic, per-op capture, fused capture, counter deltas)``.

    The bundle is generated once with the hook OFF (so a ray aimer's own
    traces run on the reference path and cannot move a fused counter), then
    the same bundle is traced twice through the same group.
    """
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    monkeypatch.setenv("OPTILAND_METAL_TRACE_DIAG", "0")
    optic = builder()
    rays = launch_bundle(optic, field)

    before_ref = dict(T.stats())
    ref = _trace(optic.surfaces, rays, mode)
    after_ref = T.stats()
    moved = {
        k: after_ref[k] - before_ref.get(k, 0)
        for k in after_ref
        if k.startswith(("fused_trace:", "fused_trace_skip:", "gpu:fused_trace"))
        and after_ref[k] != before_ref.get(k, 0)
    }
    assert not moved, f"the hook-off reference run moved a fused counter: {moved}"

    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "1")
    before = dict(T.stats())
    got = _trace(optic.surfaces, rays, mode)
    now = T.stats()
    deltas = {k: now.get(k, 0) - before.get(k, 0) for k in now}
    deltas = {k: v for k, v in deltas.items() if v and "fused" in k}
    return optic, ref, got, deltas


def assert_fused_once(deltas: dict[str, int], what: str) -> None:
    """Exactly one candidate, one trace, no refusal."""
    assert deltas.get("fused_trace:traces", 0) == 1, (
        f"{what}: fused_trace:traces moved by {deltas.get('fused_trace:traces', 0)}, "
        f"not 1; deltas {deltas}"
    )
    assert deltas.get("fused_trace:candidates", 0) == 1, f"{what}: {deltas}"
    refusals = {k: v for k, v in deltas.items() if k.startswith("fused_trace_skip:")}
    assert not refusals, f"{what}: unexpected refusals {refusals}"


def assert_refused(deltas: dict[str, int], reason: FusedTraceSkip, what: str) -> None:
    """Refused exactly once, for exactly ``reason``, with no launch."""
    key = f"fused_trace_skip:{reason.value}"
    assert deltas.get(key, 0) == 1, (
        f"{what}: expected exactly one {key}; deltas {deltas}"
    )
    other = {
        k: v
        for k, v in deltas.items()
        if k.startswith("fused_trace_skip:") and k != key
    }
    assert not other, f"{what}: refused for more than one reason: {deltas}"
    assert deltas.get("fused_trace:traces", 0) == 0, f"{what}: {deltas}"
    assert deltas.get("gpu:fused_trace", 0) == 0, (
        f"{what}: a refusal must not launch a kernel; {deltas}"
    )


# ---------------------------------------------------------------------------
# 1. The golden-value regression systems (plan 7.2, "golden systems")
# ---------------------------------------------------------------------------

#: builder name -> the refusal reason it must hit, or None when it fuses.
#:
#: **Measured, and one row disagrees with plan 7.2.**  The plan's row lists
#: "(coated, Forbes, robust-aimer)" as the ineligible builders.  ``coated``
#: and ``Forbes`` are ineligible as predicted (``SimpleCoating`` on surface 1
#: -> ``coating``; the Forbes Q2D geometry -> ``geometry_type``).  The
#: robust-aimer builder ``wide_fov`` (``WideAngle170FOV``) is NOT: the aimer
#: is ray *generation*, which happens before ``SurfaceGroup.trace`` is
#: called and leaves the surface list entirely within the v1 feature set, so
#: the system fuses and agrees with the per-op path bit for bit at every
#: field (measured at 1141 rays, both modes, fields (0,0) / (0,0.7) / (0,1)).
#: Asserting the plan's prediction here would have meant asserting a refusal
#: that does not happen; the measurement is the record (status.md, WP6-b).
GOLDEN_REASON: dict[str, FusedTraceSkip | None] = {
    "doublet": None,
    "wide_fov": None,
    "coated_doublet": FusedTraceSkip.COATING,
    "forbes_q2d_singlet": FusedTraceSkip.GEOMETRY_TYPE,
}

#: The golden systems are sampled at the axial and the full field.  The
#: 0.7 field is the catalog row's job; ``wide_fov``'s aimer costs ~40 s per
#: field per mode, so the fields here are the two that bracket the aimer's
#: range rather than all three.
GOLDEN_FIELDS: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 1.0))

GOLDEN_CASES: tuple[tuple[str, tuple[float, float]], ...] = tuple(
    (name, field) for name in GOLDEN_SYSTEMS for field in GOLDEN_FIELDS
)


def _golden_id(case: tuple[str, tuple[float, float]]) -> str:
    name, (hx, hy) = case
    return f"{name}-f{hx:g},{hy:g}"


def test_golden_systems_are_the_four_regression_builders():
    """``GOLDEN_REASON`` covers ``GOLDEN_SYSTEMS`` exactly.

    A builder added to ``tests/regression/systems.py`` without a decision
    here fails this test instead of silently escaping the conformance row.
    """
    assert set(GOLDEN_REASON) == set(GOLDEN_SYSTEMS), (
        "tests/regression/systems.py and GOLDEN_REASON disagree: "
        f"{set(GOLDEN_SYSTEMS) ^ set(GOLDEN_REASON)}"
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=_golden_id)
def test_golden_systems_fused(mps_backend, monkeypatch, case, mode):
    """Plan 7.2: the golden systems at tier-A sampling, or their refusal.

    An eligible builder must fuse exactly once and match the per-op path on
    every raw word of every recorded row and every final plane.  An
    ineligible builder must be refused with its exact reason, launch nothing,
    and still return the per-op result -- the fallback is transparent, so the
    two captures are equal there too (both were produced by the same loop).
    """
    metal.set_mode(mode)
    name, field = case
    what = f"{_golden_id(case)}[{mode}]"
    optic, ref, got, deltas = compare(GOLDEN_SYSTEMS[name], field, mode, monkeypatch)
    del optic

    reason = GOLDEN_REASON[name]
    if reason is None:
        assert_fused_once(deltas, what)
    else:
        assert_refused(deltas, reason, what)
    tc.assert_tier_a(got, ref, what)


# ---------------------------------------------------------------------------
# 2. The shipped catalog (plan 7.2, "catalog")
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def catalog_census() -> dict[str, tuple[str, str]]:
    """``trace_fixtures.audit_catalog()``: the independent eligibility census.

    Plan 0.2.6 forbids reading an expectation off the run, so which catalog
    systems are expected to fuse is decided by the census in
    ``scripts/trace_fixtures.py`` -- which walks every surface against the v1
    feature set of plan 1.1 without importing the gate -- and not by the
    gate's own answer.  Run on the NumPy backend, as its docstring requires.
    """
    previous = be.get_backend()
    device = be.get_device() if previous == "torch" else None
    precision = be.get_precision() if previous == "torch" else None
    be.set_backend("numpy")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            census = fx.audit_catalog()
    finally:
        be.set_backend(previous)
        if previous == "torch":  # pragma: no cover - the session starts on numpy
            be.set_device(device)
            be.set_precision(precision)
    return census


def test_catalog_census_matches_the_recorded_decision(catalog_census):
    """The census equals ``KNOWN_INELIGIBLE`` (both empty, measured at I0).

    ``KNOWN_INELIGIBLE`` is the decision on the record; ``audit_catalog`` is
    the evidence.  If a shipped sample ever leaves the v1 feature set, this
    fails before ``test_catalog`` starts asserting a refusal it never made.
    """
    assert set(catalog_census) == set(fx.KNOWN_INELIGIBLE), (
        f"census {sorted(catalog_census)} vs recorded {sorted(fx.KNOWN_INELIGIBLE)}"
    )
    assert len(fx.CATALOG) == 29, f"{len(fx.CATALOG)} catalog systems, expected 29"


CATALOG_NAMES: tuple[str, ...] = tuple(sorted(fx.CATALOG))


@pytest.mark.parametrize("field", FIELDS, ids=lambda f: f"f{f[0]:g},{f[1]:g}")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", CATALOG_NAMES)
def test_catalog(mps_backend, monkeypatch, catalog_census, name, mode, field):
    """Plan 7.2: every shipped sample system x mode x field, tier A.

    This is the row that carries the preface's claim that v1 covers all 29
    shipped samples end to end.  Whether a system is expected to fuse comes
    from ``catalog_census`` (the independent audit), never from the run.
    """
    metal.set_mode(mode)
    what = f"{name}[{mode}][f{field[0]:g},{field[1]:g}]"
    _, ref, got, deltas = compare(lambda: fx.build(name), field, mode, monkeypatch)

    if name in catalog_census:  # pragma: no cover - empty today
        reason, evidence = catalog_census[name]
        assert_refused(deltas, FusedTraceSkip(reason), f"{what} ({evidence})")
    else:
        assert_fused_once(deltas, what)
    tc.assert_tier_a(got, ref, what)


# ---------------------------------------------------------------------------
# 3. The historical df64 xfail sites, in sf64 (plan 7.2, "xfail sites")
# ---------------------------------------------------------------------------

#: The two ``xfail_if_emulated_df64`` sites of plan 7.2, by node id.
#:
#: Both are float64-exactness assertions that the 48-bit df64 representation
#: cannot meet and that ``tests/utils.xfail_if_emulated_df64`` turns into a
#: non-strict xfail there.  ``emulated_df64()`` is False in sf64, so the
#: context manager is inert and the assertion is live: in sf64 with the fused
#: hook on they must PASS.
XFAIL_SITES: tuple[str, ...] = (
    "tests/test_conic_root_selection.py::TestOAPSystemTraces"
    "::test_double_oap_relay_recollimates",
    "tests/test_wavefront_strategy.py::test_flatness_is_independent_of_pupil_sampling",
)


def test_xfail_sites_pass_in_sf64(tmp_path):
    """The two df64 xfail sites pass, unxfailed, in sf64 with the hook on.

    Run in a subprocess because ``OPTILAND_METAL_MODE`` and
    ``OPTILAND_TEST_MPS`` are read once per process (``PYTORCH_MPS_FAST_MATH``
    likewise, note 08) and because ``set_test_backend``'s parametrization is
    built at collection time from ``OPTILAND_TEST_MPS``.  The result is read
    from the JUnit XML rather than from the exit status, so that a run which
    selected nothing -- the failure mode a ``-k`` filter invites -- fails
    here instead of passing silently.

    ``-p no:randomly`` is not needed; ``-o addopts=`` drops the project's
    default options exactly as every other command in the plan does.
    """
    report = tmp_path / "xfail-sites-sf64.xml"
    env = dict(os.environ)
    env.update(
        {
            "PYTORCH_MPS_FAST_MATH": "0",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "MPLBACKEND": "Agg",
            "QT_QPA_PLATFORM": "offscreen",
            "OPTILAND_TEST_MPS": "1",
            "OPTILAND_TEST_MPS_GRAD": "0",
            "OPTILAND_METAL_MODE": "sf64",
            "OPTILAND_METAL_FUSED_TRACE": "1",
        }
    )
    env.pop("OPTILAND_TEST_MPS_STATS_FILE", None)
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            *XFAIL_SITES,
            "-k",
            "torch-mps",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            f"--junitxml={report}",
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert report.is_file(), (
        "the sf64 subprocess wrote no JUnit report\n"
        f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
    )
    suite = ET.parse(report).getroot()  # noqa: S314 - our own pytest's output
    cases = suite.iter("testcase")
    outcomes: dict[str, str] = {}
    for case in cases:
        node = f"{case.get('classname')}::{case.get('name')}"
        bad = [child.tag for child in case if child.tag != "system-out"]
        outcomes[node] = bad[0] if bad else "passed"

    assert outcomes, (
        "the sf64 run selected no test; the node ids or the -k filter are stale\n"
        f"stdout:\n{completed.stdout[-4000:]}"
    )
    # One parametrization per xfail site per `num_rays` value, all on torch-mps.
    assert len(outcomes) >= len(XFAIL_SITES), outcomes
    for node, outcome in sorted(outcomes.items()):
        assert outcome == "passed", (
            f"{node} did not pass in sf64 with the fused hook on: {outcome}\n"
            f"stdout:\n{completed.stdout[-4000:]}"
        )
    assert completed.returncode == 0, (
        f"the sf64 run exited {completed.returncode} although every selected "
        f"test passed\nstdout:\n{completed.stdout[-4000:]}"
    )
    print(json.dumps(outcomes, indent=2, sort_keys=True))
