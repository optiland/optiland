"""Per-epoch memo of aim solutions in ``BaseRayAimer`` (opt v2 PR2.3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import optiland.backend as be
from optiland import _epoch
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.samples.objectives import CookeTriplet

AIM_KEY = "BaseRayAimer.aim_rays"


class _CountingAimer(BaseRayAimer):
    """Deterministic aimer that counts how often its body runs."""

    def __init__(self, optic: Any = None, inner: BaseRayAimer | None = None):
        super().__init__(optic)
        self.calls = 0
        self.inner = inner

    def aim_rays(
        self,
        fields: tuple,
        wavelengths: Any,
        pupil_coords: tuple,
        initial_guess: tuple | None = None,
    ) -> tuple:
        self.calls += 1
        if self.inner is not None:
            self.inner.aim_rays(fields, wavelengths, pupil_coords)
        Px = be.as_array_1d(pupil_coords[0])
        return tuple(Px * float(i) + wavelengths for i in range(6))


def _aim_entries(epoch: _epoch.EvaluationEpoch) -> int:
    return sum(1 for key in epoch._memo if key[1][0] == AIM_KEY)


def _inputs(px: float = 0.5) -> tuple:
    return ((0.0, 1.0), 0.55, (be.array([0.0, px]), be.array([0.0, 0.0])))


def test_outside_epoch_runs_every_call(set_test_backend):
    aimer = _CountingAimer()
    aimer.aim_rays(*_inputs())
    aimer.aim_rays(*_inputs())
    assert aimer.calls == 2


def test_epoch_memoizes_by_value_of_rebuilt_arrays():
    aimer = _CountingAimer()
    with _epoch.evaluation_epoch() as epoch:
        first = aimer.aim_rays(*_inputs())
        # Fresh arrays with equal content: the pupil sample set is the same.
        second = aimer.aim_rays(*_inputs())
        assert aimer.calls == 1
        assert _aim_entries(epoch) == 1
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
        assert a is not b
    aimer.aim_rays(*_inputs())
    assert aimer.calls == 2  # memo dies with the epoch


@pytest.mark.parametrize(
    "other",
    [
        ((0.0, 0.7), 0.55, (be.array([0.0, 0.5]), be.array([0.0, 0.0]))),
        ((0.0, 1.0), 0.65, (be.array([0.0, 0.5]), be.array([0.0, 0.0]))),
        ((0.0, 1.0), 0.55, (be.array([0.0, 0.4]), be.array([0.0, 0.0]))),
        ((0, 1), 0.55, (be.array([0.0, 0.5]), be.array([0.0, 0.0]))),
        ((0.0, 1.0), 0.55, (np.array([0.0, 0.5], np.float32), be.array([0, 0.0]))),
    ],
    ids=["field", "wavelength", "pupil", "scalar-type", "dtype"],
)
def test_every_key_component_separates_entries(other):
    aimer = _CountingAimer()
    with _epoch.evaluation_epoch():
        aimer.aim_rays(*_inputs())
        aimer.aim_rays(*other)
    assert aimer.calls == 2


def test_aimers_do_not_share_entries():
    first, second = _CountingAimer(), _CountingAimer()
    with _epoch.evaluation_epoch():
        first.aim_rays(*_inputs())
        second.aim_rays(*_inputs())
    assert (first.calls, second.calls) == (1, 1)


def test_hits_return_copies():
    aimer = _CountingAimer()
    with _epoch.evaluation_epoch():
        first = aimer.aim_rays(*_inputs())
        expected = [np.array(a) for a in first]
        first[1][:] = -1.0  # a caller mutating its arrays in place
        second = aimer.aim_rays(*_inputs())
        second[2][:] = -2.0
        third = aimer.aim_rays(*_inputs())
    assert aimer.calls == 1
    for a, b in zip(third, expected, strict=True):
        np.testing.assert_array_equal(a, b)


def test_initial_guess_bypasses_memo():
    aimer = _CountingAimer()
    guess = tuple(be.zeros(2) for _ in range(6))
    with _epoch.evaluation_epoch() as epoch:
        aimer.aim_rays(*_inputs(), initial_guess=guess)
        aimer.aim_rays(*_inputs(), guess)
        assert _aim_entries(epoch) == 0
    assert aimer.calls == 2


def test_nested_aimer_calls_pass_through():
    inner = _CountingAimer()
    outer = _CountingAimer(inner=inner)
    with _epoch.evaluation_epoch() as epoch:
        outer.aim_rays(*_inputs())
        outer.aim_rays(*_inputs())
        assert _aim_entries(epoch) == 1  # the entry call only
        inner.aim_rays(*_inputs())  # a direct call memoizes on its own
        inner.aim_rays(*_inputs())
        assert _aim_entries(epoch) == 2
    assert (outer.calls, inner.calls) == (1, 2)


def test_gradient_tensors_are_keyed_by_identity():
    torch = pytest.importorskip("torch")
    be.set_backend("torch")
    try:
        aimer = _CountingAimer()
        py = torch.zeros(2, dtype=torch.float64)
        with _epoch.evaluation_epoch():
            px1 = torch.tensor([0.0, 0.5], dtype=torch.float64, requires_grad=True)
            px2 = torch.tensor([0.0, 0.5], dtype=torch.float64, requires_grad=True)
            out1 = aimer.aim_rays((0.0, 1.0), 0.55, (px1, py))
            out2 = aimer.aim_rays((0.0, 1.0), 0.55, (px2, py))
            again = aimer.aim_rays((0.0, 1.0), 0.55, (px1, py))
        assert aimer.calls == 2
        out2[1].sum().backward()
        assert px2.grad is not None and px1.grad is None
        torch.testing.assert_close(again[1], out1[1])
    finally:
        be.set_backend("numpy")


@pytest.mark.parametrize("mode", ["paraxial", "iterative", "robust"])
def test_real_aimers_identical_inside_epoch(set_test_backend, mode):
    optic = CookeTriplet()
    if mode != "paraxial":
        optic.ray_tracer.set_aiming(mode)
    Hx, Hy = be.array([0.0, 0.0]), be.array([0.7, 0.7])
    Px, Py = be.array([0.0, 0.3]), be.array([1.0, -0.6])

    def trace() -> list:
        rays = optic.trace_generic(Hx, Hy, Px, Py, 0.55)
        return [be.to_numpy(v) for v in (rays.x, rays.y, rays.L, rays.M, rays.N)]

    reference = trace()
    with _epoch.evaluation_epoch() as epoch:
        first = trace()
        hits = epoch.hits
        second = trace()
        if be.get_backend() == "numpy":
            assert _aim_entries(epoch) == 1
            assert epoch.hits > hits  # the second aim came from the memo
    for a, b, c in zip(reference, first, second, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)
