"""WP7: batched finite-difference Jacobians (``batch_trace.fd_jacobian``).

Design 6.5's rules, each with its own test: interleaved minus/plus rows so both
members of a central pair come from one launch, bound clipping with the
**actual** stencil as the denominator, merits decoded to float64 before they
are subtracted, and a small-step NumPy reference that separates truncation
error from GPU representation noise.

The comparison that matters is ``test_fd_jacobian_vs_sequential``: the fused
Jacobian against the same stencil evaluated **one design per call on the per-op
Metal path**, which is what scipy drives today.  It is an equality, not a
tolerance -- the rows are tier-A identical (plan 7.1) and the merit is the same
expression on them, so any difference is a bug.  The only tolerance in this file
is the truncation-separation bound of ``test_fd_jacobian_vs_cpu_small_step``,
and it is derived from ``MACHINE_EPS[mode]`` and a measured path scale, never
chosen to make a number pass.

Run: ``pytest tests/metal/test_trace_fd.py -q -p no:cacheprovider -o addopts=``
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import optiland.backend as be  # noqa: E402
from optiland.optimization.variable import Variable  # noqa: E402
from optiland.raytrace import batch_trace as BT  # noqa: E402

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import trace_fixtures  # noqa: E402

torch = pytest.importorskip("torch")
HAS_MPS = bool(torch.backends.mps.is_available())

MODES = ("df64", "sf64")

#: 19 hexapolar rings = 1,141 rays: above ``_MAX_VALUE_KEY_ARRAY_SIZE`` (1024),
#: so every trace compared here is in the tier-A regime.
RINGS = 19

TRACE_KWARGS = {
    "Hx": 0.0,
    "Hy": 0.0,
    "wavelength": 0.55,
    "num_rays": RINGS,
    "distribution": "hexapolar",
    "record": "image",
}

#: The stencil width, in the variables' SCALED units.  ``RadiusVariable``
#: scales by 1/100, so 1e-3 scaled is 0.1 mm physical -- comfortably above
#: df64's 2^-48 noise floor on a 80 mm radius (design 6.5: df64 needs larger
#: steps than sf64).
STEP = 1e-3

#: The reference step for the NumPy small-step Jacobian: 100x smaller, where
#: float64 has the headroom and truncation error is ~1e4 times smaller.
STEP_SMALL = 1e-5


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """No fused-trace switch leaks into or out of a test."""
    for name in (
        "OPTILAND_METAL_FUSED_TRACE",
        "OPTILAND_METAL_FUSED_TRACE_MAX_STEPS",
        "OPTILAND_METAL_FUSED_TRACE_MEMORY_FRACTION",
        "OPTILAND_METAL_FUSED_TRACE_MIN_RAYS",
        "OPTILAND_METAL_FUSED_TRACE_DRIFT",
        "OPTILAND_METAL_TRACE_DIAG",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture(autouse=True)
def _driver_state():
    """Zero the counters and the driver's one-shot state around every test."""
    if not HAS_MPS:
        yield
        return
    from optiland.backend.torch_backend.metal import tensor as T
    from optiland.backend.torch_backend.metal import trace

    trace.reset_driver_state()
    T.reset_stats()
    yield
    trace.reset_driver_state()
    T.reset_stats()


@pytest.fixture
def mps_backend():
    """torch / mps / float64 with autograd off (the fused path's setting)."""
    if not HAS_MPS:  # pragma: no cover - hardware gate
        pytest.skip("Metal GPU required")
    from optiland.backend.torch_backend import metal

    previous = metal.get_mode()
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    be.grad_mode.disable()
    yield
    be.set_backend("torch")
    be.set_device("mps")
    be.set_precision("float64")
    metal.set_mode(previous)
    be.grad_mode.disable()
    be.set_backend("numpy")


def use_mode(mode):
    """Select the Metal representation for this test."""
    from optiland.backend.torch_backend import metal

    metal.set_mode(mode)


# ---------------------------------------------------------------------------
# The merit function
# ---------------------------------------------------------------------------
def rms_merit(result, b):
    """RMS spot radius of design ``b``, from the recorded image row.

    Deliberately written on the **(N,)** plane of one design rather than on
    ``BatchTraceResult.rms_spot()``: the reduction then sees the same shape
    whether the batch holds one design or ``2n``, so a difference between the
    fused and the sequential Jacobian can only come from the rows themselves.
    """
    row = result.rows[max(result.rows)]
    x = BT._select(result.x, b, row)
    y = BT._select(result.y, b, row)
    r2 = (x - be.nanmean(x)) ** 2 + (y - be.nanmean(y)) ** 2
    return be.sqrt(be.nanmean(r2))


def cooke_variables():
    """A CookeTriplet with two post-stop variables (so the launch is shared)."""
    optic = trace_fixtures.cooke()
    return optic, [
        Variable(optic, "radius", surface_number=5),
        Variable(optic, "conic", surface_number=5),
    ]


def sequential_jacobian(optic, variables, step, monkeypatch, **kwargs):
    """The same stencil, one design per call, on the per-op Metal path.

    This is today's cost model: scipy perturbs one variable, calls
    ``optic.trace``, reads the merit, and repeats.  ``trace_batch`` with a
    single row and ``OPTILAND_METAL_FUSED_TRACE=0`` is exactly that, with the
    identical merit expression on top.
    """
    rows, actual, _ = BT._fd_rows(variables, step, True)
    merged = dict(TRACE_KWARGS)
    merged.update(kwargs)
    monkeypatch.setenv("OPTILAND_METAL_FUSED_TRACE", "0")
    try:
        merits = []
        for r in range(rows.shape[0]):
            single = BT.trace_batch(optic, variables, rows[r : r + 1], **merged)
            assert single.fused is False
            merits.append(float(be.to_numpy(rms_merit(single, 0))))
    finally:
        monkeypatch.delenv("OPTILAND_METAL_FUSED_TRACE", raising=False)
    jacobian = np.array(
        [
            (merits[2 * j + 1] - merits[2 * j]) / actual[j]
            for j in range(len(variables))
        ],
        dtype=np.float64,
    )
    return jacobian, np.asarray(merits, dtype=np.float64), actual


# ---------------------------------------------------------------------------
# The stencil itself (no GPU needed)
# ---------------------------------------------------------------------------
def test_fd_stencil_is_interleaved(mps_backend):
    """Central differences: ``2n`` rows, minus/plus per variable, one launch."""
    use_mode("df64")
    optic, variables = cooke_variables()
    base = np.array([float(v.value) for v in variables], dtype=np.float64)

    rows, actual, clipped = BT._fd_rows(variables, STEP, True)
    assert rows.shape == (2 * len(variables), len(variables))
    assert not clipped.any()
    for j in range(len(variables)):
        assert rows[2 * j, j] == base[j] - STEP
        assert rows[2 * j + 1, j] == base[j] + STEP
        # Every other column of the pair is the unperturbed value.
        for k in range(len(variables)):
            if k != j:
                assert rows[2 * j, k] == base[k]
                assert rows[2 * j + 1, k] == base[k]
        # The denominator is the width the rows actually span.
        assert actual[j] == (base[j] + STEP) - (base[j] - STEP)


def test_fd_stencil_forward_mode(mps_backend):
    """Forward differences: the base row first, then one row per variable."""
    use_mode("df64")
    optic, variables = cooke_variables()
    base = np.array([float(v.value) for v in variables], dtype=np.float64)

    rows, actual, clipped = BT._fd_rows(variables, STEP, False)
    assert rows.shape == (len(variables) + 1, len(variables))
    assert np.array_equal(rows[0], base)
    assert not clipped.any()
    for j in range(len(variables)):
        assert rows[j + 1, j] == base[j] + STEP
        assert actual[j] == (base[j] + STEP) - base[j]


def test_fd_stencil_bounds_clip_the_denominator(mps_backend):
    """A bound shortens the stencil AND the denominator, never just one."""
    use_mode("df64")
    optic = trace_fixtures.cooke()
    radius = Variable(optic, "radius", surface_number=5)
    base = float(radius.value)
    # Pin the upper bound exactly half a step above the nominal value.
    bounded = Variable(
        optic,
        "radius",
        surface_number=5,
        min_val=None,
        max_val=radius.variable.inverse_scale(base + STEP / 2),
    )
    assert bounded.bounds[1] == pytest.approx(base + STEP / 2, rel=1e-12)

    rows, actual, clipped = BT._fd_rows([bounded], STEP, True)
    assert clipped.tolist() == [True]
    assert rows[1, 0] == pytest.approx(base + STEP / 2, rel=1e-12)
    assert rows[0, 0] == pytest.approx(base - STEP, rel=1e-12)
    assert actual[0] == rows[1, 0] - rows[0, 0]
    assert actual[0] != 2 * STEP


def test_fd_step_shape_is_validated(mps_backend):
    """A per-variable step must have one entry per variable."""
    use_mode("df64")
    optic, variables = cooke_variables()
    with pytest.raises(ValueError, match=r"step must be scalar or \(2,\)"):
        BT._fd_rows(variables, np.array([STEP, STEP, STEP]), True)


# ---------------------------------------------------------------------------
# The Jacobian
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_fd_jacobian_vs_sequential(mps_backend, monkeypatch, mode):
    """The fused Jacobian equals the sequential per-op one, exactly."""
    use_mode(mode)
    optic, variables = cooke_variables()

    result = BT.fd_jacobian(optic, variables, rms_merit, STEP, **TRACE_KWARGS)
    assert result.central is True
    assert result.result.fused is True, (
        "fd_jacobian fell back to the contract loop, so this would compare the "
        f"per-op path with itself; refusals: "
        f"{ {k: v for k, v in result.result.stats.items() if 'skip' in k} }"
    )
    assert result.result.launch_shared is True
    assert result.merits.shape == (2 * len(variables),)
    assert np.isnan(result.base)
    assert not result.clipped.any()

    want, merits, actual = sequential_jacobian(
        optic, variables, STEP, monkeypatch, **TRACE_KWARGS
    )
    assert np.array_equal(result.steps, actual)
    assert np.array_equal(result.merits, merits), (
        f"merits differ: {result.merits!r} vs {merits!r}"
    )
    assert np.array_equal(result.jacobian, want), (
        f"jacobian differs: {result.jacobian!r} vs {want!r}"
    )
    # The derivative is real, not a wash of noise: a 0.1 mm radius change on
    # surface 5 moves the RMS spot of a Cooke triplet measurably.
    assert np.all(np.isfinite(result.jacobian))
    assert abs(result.jacobian[0]) > 1e-4


@pytest.mark.parametrize("mode", MODES)
def test_fd_jacobian_vs_cpu_small_step(mps_backend, monkeypatch, mode):
    """Truncation error is separable: the fused Jacobian is no further from a
    small-step float64 reference than the sequential one is.

    ``J_fused`` and ``J_seq`` use the same stencil, so they carry the same
    truncation error; what separates them is representation noise, bounded by
    ``64 * MACHINE_EPS[mode] * |merit| / h`` -- the merit's own noise floor
    divided by the stencil width.
    """
    from optiland.backend.torch_backend.metal.tensor import MACHINE_EPS

    use_mode(mode)
    optic, variables = cooke_variables()

    fused = BT.fd_jacobian(optic, variables, rms_merit, STEP, **TRACE_KWARGS)
    assert fused.result.fused is True

    reference = BT.fd_reference_cpu(
        optic, variables, rms_merit, STEP_SMALL, **TRACE_KWARGS
    )
    # ``fd_reference_cpu`` restores the backend it found; make sure it did.
    assert be.get_backend() == "torch"
    use_mode(mode)

    sequential, _, _ = sequential_jacobian(
        optic, variables, STEP, monkeypatch, **TRACE_KWARGS
    )

    scale = float(np.max(np.abs(fused.merits)))
    for j in range(len(variables)):
        bound = 64 * MACHINE_EPS[mode] * scale / abs(fused.steps[j])
        near = abs(fused.jacobian[j] - reference[j])
        far = abs(sequential[j] - reference[j])
        assert near <= far + bound, (
            f"variable {j}: |J_fused - J_cpu| = {near!r} exceeds "
            f"|J_seq - J_cpu| = {far!r} + {bound!r}"
        )
        # The reference is a real derivative of the same quantity, not noise.
        assert np.sign(reference[j]) == np.sign(fused.jacobian[j]) or (
            abs(reference[j]) < bound
        )


@pytest.mark.parametrize("mode", MODES)
def test_fd_jacobian_forward_matches_its_own_stencil(mps_backend, monkeypatch, mode):
    """Forward mode: ``base`` is the unperturbed merit and the rows are n+1."""
    use_mode(mode)
    optic, variables = cooke_variables()

    result = BT.fd_jacobian(
        optic, variables, rms_merit, STEP, central=False, **TRACE_KWARGS
    )
    assert result.result.fused is True
    assert result.central is False
    assert result.merits.shape == (len(variables) + 1,)
    assert result.base == result.merits[0]

    nominal = BT.trace_batch(
        optic,
        variables,
        np.array([[float(v.value) for v in variables]]),
        **TRACE_KWARGS,
    )
    assert nominal.fused is True
    assert result.base == float(be.to_numpy(rms_merit(nominal, 0)))
    for j in range(len(variables)):
        assert result.jacobian[j] == (
            (result.merits[j + 1] - result.merits[0]) / result.steps[j]
        )


def test_fd_jacobian_runs_on_numpy():
    """``fd_jacobian`` is backend-agnostic: no Metal, no torch needed."""
    be.set_backend("numpy")
    optic, variables = cooke_variables()
    result = BT.fd_jacobian(optic, variables, rms_merit, STEP, **TRACE_KWARGS)
    assert result.result.fused is False
    assert result.jacobian.shape == (len(variables),)
    assert np.all(np.isfinite(result.jacobian))

    # The optic is back where it started.
    assert float(optic.surfaces[5].geometry.radius) == pytest.approx(79.68360, abs=1e-9)
