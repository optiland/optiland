"""Third-order aberrations share one computation inside an evaluation epoch.

The tests compare values as bytes instead of using assert_allclose, because
the memo must not change a bit. Only the gradient tests allow for rounding:
autograd adds the contributions of a reused tensor in another order.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

import optiland.backend as be
from optiland import _epoch
from optiland.aberrations.third_order import ThirdOrderAberrations
from optiland.optimization import OptimizationProblem
from optiland.samples.objectives import CookeTriplet, DoubleGauss
from optiland.samples.telescopes import HubbleTelescope

from .utils import assert_allclose

METHODS = ("TSC", "SC", "CC", "TCC", "TAC", "AC", "TPC", "PC", "DC")
# The facade computes these with two other ThirdOrderAberrations objects.
FACADE_METHODS = ("seidels", "seidels_per_surface", "TAchC", "LchC", "TchC")


def _as_numpy(value):
    return np.asarray(be.to_numpy(value))


def _values(optic):
    third = optic.aberrations._third_order
    out = [_as_numpy(getattr(third, name)()) for name in METHODS]
    out += [_as_numpy(v) for v in third.third_order()]
    out += [_as_numpy(getattr(optic.aberrations, name)()) for name in FACADE_METHODS]
    return out


def _count_precalculations(monkeypatch):
    calls = []
    original = ThirdOrderAberrations._compute_precalculations

    def counted(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(ThirdOrderAberrations, "_compute_precalculations", counted)
    return calls


def _loss(optic):
    """SC and TSC reuse one memoized term, and SC also reads ``_ua``."""
    aberrations = optic.aberrations
    return aberrations.SC().sum() + aberrations.TSC().sum() + aberrations.CC().sum()


def _radius_gradient(loss):
    """Gradient of ``loss(optic)`` with respect to the radius of surface 1."""
    import torch

    optic = CookeTriplet()
    geometry = optic.surfaces[1].geometry
    radius = torch.tensor(
        float(be.to_numpy(geometry.radius)), dtype=torch.float64, requires_grad=True
    )
    geometry.radius = radius
    (gradient,) = torch.autograd.grad(loss(optic), radius)
    return gradient


@pytest.mark.parametrize("factory", [CookeTriplet, DoubleGauss, HubbleTelescope])
def test_values_are_bit_identical_inside_an_epoch(set_test_backend, factory):
    plain = _values(factory())
    optic = factory()
    with _epoch.evaluation_epoch((optic,)):
        memoized = _values(optic)
        again = _values(optic)
    for a, b, c in zip(plain, memoized, again, strict=True):
        assert a.shape == b.shape == c.shape
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
        assert a.dtype == c.dtype and a.tobytes() == c.tobytes()


def test_one_precalculation_per_epoch(set_test_backend, monkeypatch):
    calls = _count_precalculations(monkeypatch)
    optic = CookeTriplet()
    third = optic.aberrations._third_order
    with _epoch.evaluation_epoch((optic,)):
        _values(optic)
    assert calls.count(third) == 1
    calls.clear()
    _values(optic)  # outside an epoch every call computes
    # Each method computes once (TCC computes through its call of CC), and
    # third_order once.
    assert calls.count(third) == len(METHODS) + 1


def test_a_term_is_computed_once_per_epoch(set_test_backend, monkeypatch):
    """TSC, SC and third_order share one TSC term over the surfaces."""
    calls = []
    original = ThirdOrderAberrations._TSC_term

    def counted(self, k):
        calls.append(k)
        return original(self, k)

    monkeypatch.setattr(ThirdOrderAberrations, "_TSC_term", counted)
    optic = CookeTriplet()
    third = optic.aberrations._third_order
    with _epoch.evaluation_epoch((optic,)):
        third.TSC()
        third.SC()
        third.third_order()
    assert calls == list(range(1, optic.surfaces.num_surfaces - 1))


def test_seidel_operands_share_the_precalculation(set_test_backend, monkeypatch):
    optic = CookeTriplet()
    problem = OptimizationProblem()
    for name in ("TSC", "TCC", "TAC", "PC"):
        for surface in range(optic.surfaces.num_surfaces - 2):
            problem.add_operand(
                name,
                target=0.0,
                weight=1.0,
                input_data={"optic": optic, "surface_number": surface},
            )
    calls = _count_precalculations(monkeypatch)
    memoized = _as_numpy(problem.sum_squared())
    assert len(calls) == 1

    # The evaluator now opens no epoch, so each operand runs its own
    # precalculation.
    monkeypatch.setattr(_epoch, "evaluation_epoch", contextlib.nullcontext)
    calls.clear()
    plain = _as_numpy(problem.sum_squared())
    assert len(calls) == 4 * (optic.surfaces.num_surfaces - 2)
    assert memoized.dtype == plain.dtype and memoized.tobytes() == plain.tobytes()


def test_a_change_between_epochs_is_seen(set_test_backend):
    optic = CookeTriplet()
    with _epoch.evaluation_epoch((optic,)):
        before = _as_numpy(optic.aberrations.TSC()).copy()
    radius = 1.05 * float(be.to_numpy(optic.surfaces[1].geometry.radius))
    optic.updater.set_radius(radius, 1)
    with _epoch.evaluation_epoch((optic,)):
        after = _as_numpy(optic.aberrations.TSC())

    fresh = CookeTriplet()
    fresh.updater.set_radius(radius, 1)
    expected = _as_numpy(fresh.aberrations.TSC())
    assert after.dtype == expected.dtype and after.tobytes() == expected.tobytes()
    assert not np.array_equal(after, before)


@pytest.mark.skipif(
    "torch" not in be.list_available_backends(), reason="Requires the Torch backend"
)
@pytest.mark.parametrize("set_test_backend", ["torch"], indirect=True)
def test_torch_gradients_are_the_same_with_the_memo(set_test_backend):
    """Memoized tensors are reused in one autograd graph.

    The gradient must not change, apart from rounding.
    """

    def memoized(optic):
        with _epoch.evaluation_epoch((optic,)):
            return _loss(optic)

    assert_allclose(
        _radius_gradient(memoized), _radius_gradient(_loss), rtol=1e-12, atol=0.0
    )


@pytest.mark.skipif(
    "torch" not in be.list_available_backends(), reason="Requires the Torch backend"
)
@pytest.mark.parametrize("set_test_backend", ["torch"], indirect=True)
def test_values_are_written_back_after_a_call_without_gradients(set_test_backend):
    """A call under torch.no_grad replaces the precalculated values.

    The next call with gradients must write its stored values back, or SC
    divides by a ``_ua`` without a graph and the gradient changes.
    """
    import torch

    def memoized(optic):
        with _epoch.evaluation_epoch((optic,)):
            optic.aberrations.SC()
            with torch.no_grad():
                optic.aberrations.SC()
            return _loss(optic)

    assert_allclose(
        _radius_gradient(memoized), _radius_gradient(_loss), rtol=1e-12, atol=0.0
    )
