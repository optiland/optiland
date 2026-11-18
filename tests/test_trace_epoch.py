"""Each real trace runs inside an evaluation epoch (speed PR, item 2)."""

from __future__ import annotations

import contextlib

import pytest

import optiland.backend as be
from optiland import _epoch, _instrument
from optiland.raytrace import real_ray_tracer
from optiland.samples.objectives import CookeTriplet, DoubleGauss

from .utils import assert_allclose

RAY_FIELDS = ("x", "y", "z", "L", "M", "N", "opd")


def _record_epoch_during_trace(monkeypatch, optic) -> list:
    seen = []
    # SurfaceGroup.trace and _trace_with_final_propagation, which
    # RealRayTracer calls, both trace through SurfaceGroup._trace_in_epoch.
    original = optic.surfaces._trace_in_epoch

    def spy(*args, **kwargs):
        seen.append(_epoch.current())
        return original(*args, **kwargs)

    monkeypatch.setattr(optic.surfaces, "_trace_in_epoch", spy)
    return seen


def test_trace_runs_inside_an_epoch(set_test_backend, monkeypatch):
    optic = CookeTriplet()
    seen = _record_epoch_during_trace(monkeypatch, optic)
    optic.trace_generic(0, 1, 0, 0, 0.55)
    assert seen
    assert all(epoch is not None for epoch in seen)
    assert _epoch.current() is None


def test_outer_epoch_is_reused(set_test_backend, monkeypatch):
    optic = CookeTriplet()
    seen = _record_epoch_during_trace(monkeypatch, optic)
    with _epoch.evaluation_epoch() as outer:
        optic.trace(0, 1, 0.55, num_rays=3)
    assert seen
    assert all(epoch is outer for epoch in seen)


def test_standalone_trace_memoizes_paraxial(set_test_backend):
    optic = CookeTriplet()
    with _instrument.counting() as counts:
        optic.trace_generic(0, 1, 0, 0, 0.55)
    # Before trace epochs: paraxial_trace == 3, material_n == 45 (both backends).
    # Torch runs with grad mode on here, which keeps some lookups unmemoized.
    assert counts["paraxial_trace"] == 1
    assert counts["material_n"] == (29 if be.get_backend() == "torch" else 16)


def test_state_change_between_traces_is_seen(set_test_backend):
    optic = CookeTriplet()
    before = optic.trace_generic(0, 1, 0, 0.7, 0.55)
    new_radius = 1.05 * float(be.to_numpy(optic.surfaces[1].geometry.radius))
    optic.updater.set_radius(new_radius, 1)
    after = optic.trace_generic(0, 1, 0, 0.7, 0.55)

    fresh = CookeTriplet()
    fresh.updater.set_radius(new_radius, 1)
    expected = fresh.trace_generic(0, 1, 0, 0.7, 0.55)

    for name in RAY_FIELDS[:-1]:
        assert_allclose(getattr(after, name), getattr(expected, name))
    assert not bool(be.allclose(after.y, before.y))


@contextlib.contextmanager
def _no_epoch(*args, **kwargs):
    yield None


@pytest.mark.parametrize(
    "factory, call",
    [
        (CookeTriplet, lambda o: o.trace(0, 1, 0.55, num_rays=5)),
        (DoubleGauss, lambda o: o.trace_generic(0, 0.7, 0.3, -0.5, 0.4861)),
    ],
    ids=["cooke_trace", "double_gauss_trace_generic"],
)
def test_results_match_unmemoized_trace(set_test_backend, monkeypatch, factory, call):
    memoized = call(factory())
    monkeypatch.setattr(real_ray_tracer._epoch, "evaluation_epoch", _no_epoch)
    plain = call(factory())
    for name in RAY_FIELDS:
        assert_allclose(getattr(memoized, name), getattr(plain, name))


def test_epoch_guard_passes_on_trace(monkeypatch):
    be.set_backend("numpy")
    monkeypatch.setenv(_epoch.CHECK_ENV, "1")
    optic = CookeTriplet()
    optic.trace(0, 1, 0.55, num_rays=5)
    optic.trace_generic(0, 1, 0, 0, 0.55)


def test_torch_gradients_flow_through_trace():
    torch = pytest.importorskip("torch")
    try:
        be.set_backend("torch")
        be.set_precision("float64")
        with be.grad_mode.temporary_enable():
            optic = CookeTriplet()
            geometry = optic.surfaces[1].geometry
            radius = torch.tensor(
                float(be.to_numpy(geometry.radius)),
                dtype=torch.float64,
                requires_grad=True,
            )
            geometry.radius = radius
            rays = optic.trace_generic(0, 1, 0, 0.7, 0.55)
            rays.y.sum().backward()
        assert radius.grad is not None
        assert bool(torch.isfinite(radius.grad))
        assert float(radius.grad) != 0.0
    finally:
        be.grad_mode.disable()
        be.set_backend("numpy")
