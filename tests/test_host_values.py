"""Tests for the host-value scope of a CUDA trace (optiland._host_values).

The scope copies the values that a trace checks in Python to the host in one
device-to-host copy. The checks take the same branches, so a trace must give
the same bits with and without the scope, forward and in the gradients. The
tests compare bits with ``assert_bits_equal``: a result that is only close
can hide a check that took another branch.

The scope copies only values on a CUDA device. The tests add the CPU to its
device types, so that the same code runs without a GPU. They use the Torch
backend on the CPU with grad mode off (the ``torch_cpu`` fixture), not the
``set_test_backend`` fixture. That fixture enables grad mode; the
wavelengths then require gradients, and the scope does not copy any(k > 0).

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import functools
import math
import traceback
import warnings
from pathlib import Path

import numpy as np
import pytest

import optiland
import optiland._host_values as host_values
import optiland.backend as be
from optiland.geometries import EvenAsphere
from optiland.rays import RealRays
from optiland.samples.objectives import CookeTriplet

from .utils import assert_bits_equal

torch = pytest.importorskip("torch")

# The tilted test system is outside the scalar paraxial domain; ray aiming
# warns about it, as expected.
pytestmark = pytest.mark.filterwarnings("ignore:paraxial ray tracing")

RAY_ATTRS = ("x", "y", "z", "L", "M", "N", "i", "opd")
RECORD_ATTRS = ("x", "y", "z", "intensity")


@pytest.fixture
def torch_cpu():
    """The Torch backend on the CPU, in float64, with grad mode off."""
    be.set_backend("torch")
    precision = be.get_precision()
    be.set_device("cpu")
    be.set_precision("float64")
    be.grad_mode.disable()
    yield
    be.grad_mode.disable()
    be.set_precision(f"float{precision}")
    be.set_backend("numpy")


@pytest.fixture
def cpu_copies(monkeypatch):
    """Let the scope copy CPU tensors, so that the tests run without a GPU."""
    monkeypatch.setattr(host_values, "_DEVICE_TYPES", frozenset({"cuda", "cpu"}))


@pytest.fixture
def memo_sizes(cpu_copies, monkeypatch):
    """The number of copies in the memo of each scope, in order."""
    sizes = []
    original = host_values._copy

    def spy(scalars, absorption):
        memo = original(scalars, absorption)
        sizes.append(0 if memo is None else len(memo))
        return memo

    monkeypatch.setattr(host_values, "_copy", spy)
    return sizes


@pytest.fixture
def absorption_counts(cpu_copies, monkeypatch):
    """The number of k arrays that each scope receives, in order."""
    counts = []
    original = host_values._copy

    def spy(scalars, absorption):
        counts.append(len(absorption))
        return original(scalars, absorption)

    monkeypatch.setattr(host_values, "_copy", spy)
    return counts


def _tilted_asphere():
    """The Cooke triplet with a tilted, decentered even asphere."""
    optic = CookeTriplet()
    surface = optic.surfaces.surfaces[3]
    surface.geometry = EvenAsphere(
        surface.geometry.cs, surface.geometry.radius, -0.4, coefficients=[1e-4, -2e-6]
    )
    surface.geometry.cs.rx = be.array(0.01)
    surface.geometry.cs.y = be.array(0.05)
    return optic


def _state(optic, rays):
    """The ray arrays and the records of each surface, as NumPy copies."""
    state = {name: be.to_numpy(getattr(rays, name)).copy() for name in RAY_ATTRS}
    for k, surface in enumerate(optic.surfaces.surfaces):
        for name in RECORD_ATTRS:
            state[f"s{k}.{name}"] = be.to_numpy(getattr(surface, name)).copy()
    return state


def _assert_same(a, b):
    """Assert that two dicts of arrays hold the same arrays, bit for bit."""
    assert a.keys() == b.keys()
    for key in a:
        assert_bits_equal(a[key], b[key])


def _trace(build, monkeypatch, enabled):
    """Trace a hexapolar bundle through ``build()``, with or without the scope."""
    monkeypatch.setattr(host_values, "_ENABLED", enabled)
    optic = build()
    rays = optic.trace(Hx=0.0, Hy=0.8, wavelength=0.55, num_rays=8)
    return _state(optic, rays)


def _fan(n):
    """A fan of n rays in the y-z plane, on the device of the backend."""
    angle = np.linspace(-0.05, 0.05, n)
    return RealRays(
        np.zeros(n),
        np.linspace(-2.0, 2.0, n),
        np.zeros(n),
        np.zeros(n),
        np.sin(angle),
        np.cos(angle),
        np.ones(n),
        np.full(n, 0.55),
    )


def _loss(rays):
    """The sum of x**2 + y**2 over the rays that did not miss a surface."""
    valid = ~torch.isnan(rays.x)
    return (rays.x[valid] ** 2 + rays.y[valid] ** 2).sum()


@pytest.mark.parametrize("build", [CookeTriplet, _tilted_asphere])
def test_traces_are_bit_identical(build, torch_cpu, memo_sizes, monkeypatch):
    """The checks read the copies and take the same branches."""
    with_copies = _trace(build, monkeypatch, True)
    assert memo_sizes and all(size > 0 for size in memo_sizes)
    _assert_same(with_copies, _trace(build, monkeypatch, False))


def test_one_copy_for_each_trace(torch_cpu, memo_sizes):
    """optic.trace makes one device-to-host copy for its trace of real rays."""
    _tilted_asphere().trace(Hx=0.0, Hy=0.8, wavelength=0.55, num_rays=8)
    assert len(memo_sizes) == 1


def test_gradients_are_bit_identical(torch_cpu, cpu_copies, monkeypatch):
    """Grad mode on: the radius and the tilt get the same gradients.

    The wavelengths require gradients, so the scope does not copy any(k > 0),
    and each absorption check reads the tensor.
    """

    def gradients(enabled):
        monkeypatch.setattr(host_values, "_ENABLED", enabled)
        be.grad_mode.enable()
        optic = _tilted_asphere()
        geometry = optic.surfaces.surfaces[3].geometry
        geometry.radius.requires_grad_(True)
        geometry.cs.rx.requires_grad_(True)
        rays = optic.trace(Hx=0.0, Hy=0.8, wavelength=0.55, num_rays=8)
        _loss(rays).backward()
        return {
            "radius": geometry.radius.grad.numpy().copy(),
            "tilt": geometry.cs.rx.grad.numpy().copy(),
        }

    _assert_same(gradients(True), gradients(False))


def test_gradients_with_copied_absorption_are_bit_identical(
    torch_cpu, absorption_counts, monkeypatch
):
    """Grad mode off and a tracked radius: the scope copies any(k > 0).

    With grad mode off, the wavelengths do not require gradients, so the
    scope copies any(k > 0) of each medium. The radius requires a gradient,
    so the trace still builds an autograd graph through it. With grad mode
    off, optic.trace runs under torch.no_grad, so the test calls
    SurfaceGroup.trace, which opens an epoch for more than 128 rays.
    """

    def gradient(enabled):
        monkeypatch.setattr(host_values, "_ENABLED", enabled)
        optic = _tilted_asphere()
        radius = optic.surfaces.surfaces[3].geometry.radius
        radius.requires_grad_(True)
        rays = optic.surfaces.trace(_fan(200))
        _loss(rays).backward()
        return {"radius": radius.grad.numpy().copy()}

    with_copies = gradient(True)
    assert len(absorption_counts) == 1 and absorption_counts[0] > 0
    _assert_same(with_copies, gradient(False))


def test_a_tracked_zero_tilt_keeps_its_gradient(torch_cpu, cpu_copies, monkeypatch):
    """A zero tilt that requires a gradient is applied, as without the copy.

    ``_needs_rotation`` checks ``requires_grad`` before it reads the host
    copy, so the tilt enters the graph and its gradient is defined.
    """

    def gradient(enabled):
        monkeypatch.setattr(host_values, "_ENABLED", enabled)
        be.grad_mode.enable()
        optic = CookeTriplet()
        tilt = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        optic.surfaces.surfaces[3].geometry.cs.ry = tilt
        rays = optic.trace(Hx=0.0, Hy=0.8, wavelength=0.55, num_rays=8)
        valid = ~torch.isnan(rays.x)
        (rays.x[valid] ** 2).sum().backward()
        assert tilt.grad is not None
        return {"tilt": tilt.grad.numpy().copy()}

    _assert_same(gradient(True), gradient(False))


def test_absorption_is_copied_only_where_the_epoch_memo_returns_k(
    torch_cpu, absorption_counts
):
    """The scope gets k only where the trace gets the same k array again.

    Then k is not evaluated a second time. optic.trace traces in an
    evaluation epoch, and the material memo of the epoch returns the same k
    array. With grad mode on, the wavelengths require gradients, and the
    memo does not keep their k.
    """
    optic = CookeTriplet()
    optic.trace(Hx=0.0, Hy=0.8, wavelength=0.55, num_rays=8)
    assert absorption_counts and absorption_counts[-1] > 0

    absorption_counts.clear()
    be.grad_mode.enable()
    optic.trace(Hx=0.0, Hy=0.8, wavelength=0.55, num_rays=8)
    assert absorption_counts == [0]


def test_traces_off_cuda_open_no_scope(monkeypatch):
    """NumPy rays and Torch rays on the CPU take the code path without a scope."""
    calls = []
    monkeypatch.setattr(host_values, "_copy", lambda *values: calls.append(values))
    CookeTriplet().trace(Hx=0.0, Hy=0.8, wavelength=0.55, num_rays=8)
    be.set_backend("torch")
    try:
        be.set_device("cpu")
        CookeTriplet().trace(Hx=0.0, Hy=0.8, wavelength=0.55, num_rays=8)
    finally:
        be.set_backend("numpy")
    assert calls == []


@pytest.mark.parametrize(
    "value", [0.0, -0.0, 0.25, -3.5, math.nan, math.inf, -math.inf]
)
def test_host_copy_has_the_value_of_its_tensor(value, cpu_copies):
    """A copy has the truth value, the sign and the infinity of its tensor."""
    for dtype in (torch.float64, torch.float32):
        angle = torch.tensor(value, dtype=dtype)
        with host_values.host_scope([angle]):
            copy = host_values.host(angle)
        assert bool(copy) == bool(angle)
        assert math.isinf(copy) == bool(torch.isinf(angle))
        assert math.copysign(1.0, copy) == math.copysign(1.0, value)
        assert copy == float(angle) or (math.isnan(copy) and math.isnan(value))


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        ([0.0, 0.0], False),
        ([0.0, 1e-9], True),
        ([math.nan, 0.0], False),
        ([-1.0, 0.0], False),
        ([], False),
    ],
)
def test_absorbs_is_any_positive(k, expected, cpu_copies):
    k = torch.tensor(k, dtype=torch.float64)
    with host_values.host_scope(absorption=[k]):
        assert host_values.absorbs(k) is expected
    assert expected == bool(torch.any(k > 0))


def test_nothing_out_of_scope_or_for_other_objects(cpu_copies):
    angle = torch.tensor(0.5, dtype=torch.float64)
    other = torch.tensor(0.5, dtype=torch.float64)
    vector = torch.zeros(3)
    complex_value = torch.tensor(1j)
    assert host_values.host(angle) is None
    scalars = [angle, np.float64(1.0), None, vector, complex_value]
    with host_values.host_scope(scalars, absorption=[complex_value]):
        assert host_values.host(angle) == 0.5
        assert host_values.host(other) is None
        assert host_values.host(vector) is None
        assert host_values.host(complex_value) is None
        assert host_values.absorbs(complex_value) is None
        assert host_values.absorbs(angle) is None
    assert host_values.host(angle) is None


def test_tensors_on_another_device_keep_their_own_check(monkeypatch):
    """One copy needs one device: the device of the first CUDA tensor.

    The test uses the CPU and the meta device as the two devices, so that it
    runs without a GPU.
    """
    monkeypatch.setattr(host_values, "_DEVICE_TYPES", frozenset({"cpu", "meta"}))
    first = torch.tensor(0.5, dtype=torch.float64)
    other = torch.tensor(0.5, dtype=torch.float64, device="meta")
    k = torch.ones(2, device="meta")
    with host_values.host_scope([first, other], absorption=[k]):
        assert host_values.host(first) == 0.5
        assert host_values.host(other) is None
        assert host_values.absorbs(k) is None


def test_cpu_tensors_are_not_copied_by_default():
    angle = torch.tensor(0.5, dtype=torch.float64)
    with host_values.host_scope([angle]):
        assert host_values.host(angle) is None


def _sync_sources(trace):
    """Return the Optiland function that caused each CUDA synchronization.

    ``torch.cuda.set_sync_debug_mode("warn")`` warns at each synchronization.
    The source of a warning is the innermost frame in the optiland package
    and outside ``optiland/backend``, so a check that calls ``be.any()``
    counts in its own function.

    Args:
        trace: Called without arguments, with the warnings on.

    Returns:
        "file:function" for each synchronization, in order.
    """
    package = Path(optiland.__file__).parent
    backend = package / "backend"
    sources = []

    def show(message, category, filename, lineno, file=None, line=None):
        if "synchroniz" not in str(message):
            return
        for frame in reversed(traceback.extract_stack()):
            path = Path(frame.filename)
            if path.is_relative_to(package) and not path.is_relative_to(backend):
                sources.append(f"{path.name}:{frame.name}")
                return
        sources.append("outside optiland")

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = show
        torch.cuda.set_sync_debug_mode("warn")
        try:
            trace()
        finally:
            torch.cuda.set_sync_debug_mode(0)
    return sources


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_cuda_trace_does_not_wait_in_the_batched_checks(monkeypatch):
    """The checks that the scope batches cause no CUDA synchronization.

    A manual check on a machine with a CUDA device. CI has no GPU, so it
    skips this test. The bundle has 200 rays, more than the 128 rays from
    which SurfaceGroup.trace opens an epoch, so the scope copies any(k > 0).
    The first trace is a warm-up: it also evaluates each glass for the
    first time, and the material caches keep the values. The checked traces
    then show the synchronizations of each later trace. The trace without
    the scope shows that the test sees the checks.
    """
    be.set_backend("torch")
    precision = be.get_precision()
    be.set_device("cuda")
    be.set_precision("float64")
    try:
        optic = _tilted_asphere()
        optic.surfaces.trace(_fan(200))
        sources = {}
        for enabled in (True, False):
            monkeypatch.setattr(host_values, "_ENABLED", enabled)
            trace = functools.partial(optic.surfaces.trace, _fan(200))
            sources[enabled] = _sync_sources(trace)
    finally:
        be.set_precision(f"float{precision}")
        be.set_device("cpu")
        be.set_backend("numpy")
    checks = (
        "coordinate_system.py:_needs_rotation",
        "standard.py:_is_radius_infinite",
        "homogeneous.py:propagate",
    )
    for check in checks:
        assert check not in sources[True]
        assert check in sources[False]
    assert sources[True].count("_host_values.py:_copy") == 1
    # One copy of the four counts of the Newton-Raphson warning of the asphere.
    assert sources[True].count("newton_raphson.py:_warn_rejected_rays") == 1
