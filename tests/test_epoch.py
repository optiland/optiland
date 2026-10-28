"""Tests for evaluation epochs and the material epoch memo (opt v2 PR2.1).

Kramer Harrison, 2026
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import optiland.backend as be
from optiland import _epoch, _instrument
from optiland.materials.base import BaseMaterial

from .utils import assert_allclose


class CountingMaterial(BaseMaterial):
    """Dispersive material that records every real evaluation."""

    def __init__(self, a: float = 1.5, b: float = 0.004):
        super().__init__()
        self.a = a
        self.b = b
        self.calls = 0

    def _cache_state(self) -> tuple | None:
        return self._state_key((self.a, self.b))

    def _calculate_n(self, wavelength, **kwargs):
        self.calls += 1
        return self.a + self.b / be.asarray(wavelength) ** 2

    def _calculate_k(self, wavelength, **kwargs):
        return 0.0 * be.asarray(wavelength)


class Owner:
    def __init__(self):
        self.calls = 0

    @_epoch.epoch_memo()
    def double(self, x, scale=1):
        self.calls += 1
        return 2 * x * scale


@pytest.fixture
def epoch_check(monkeypatch):
    monkeypatch.setenv(_epoch.CHECK_ENV, "1")


class TestEvaluationEpoch:
    def test_no_epoch_by_default(self):
        assert _epoch.current() is None

    def test_get_or_compute_memoizes_per_owner_and_key(self):
        a, b = object(), object()
        with _epoch.evaluation_epoch() as epoch:
            assert epoch.get_or_compute(a, "k", lambda: 1) == 1
            assert epoch.get_or_compute(a, "k", lambda: 2) == 1
            assert epoch.get_or_compute(b, "k", lambda: 3) == 3
            assert epoch.get_or_compute(a, "j", lambda: 4) == 4
            assert (epoch.hits, epoch.misses, len(epoch)) == (1, 3, 3)
        assert _epoch.current() is None

    def test_nested_epoch_reuses_outer(self):
        with _epoch.evaluation_epoch() as outer:
            with _epoch.evaluation_epoch() as inner:
                assert inner is outer
            assert _epoch.current() is outer
        assert _epoch.current() is None

    def test_memo_is_discarded_on_exit(self):
        owner = object()
        with _epoch.evaluation_epoch() as epoch:
            epoch.get_or_compute(owner, "k", lambda: 1)
        with _epoch.evaluation_epoch() as epoch:
            assert epoch.get_or_compute(owner, "k", lambda: 2) == 2

    def test_epoch_reset_on_exception(self):
        with pytest.raises(ValueError), _epoch.evaluation_epoch():
            raise ValueError
        assert _epoch.current() is None

    def test_entries_keep_owner_and_keyed_objects_alive(self):
        with _epoch.evaluation_epoch() as epoch:
            array = np.ones(3)
            key = _epoch.array_key(array)
            epoch.get_or_compute(object(), key, lambda: 1, keep=(array,))
            del array
            owner, keep, _ = next(iter(epoch._memo.values()))
            assert owner is not None
            assert key[1] == id(keep[0])

    def test_epochs_are_per_thread(self):
        seen = []
        with _epoch.evaluation_epoch():
            thread = threading.Thread(target=lambda: seen.append(_epoch.current()))
            thread.start()
            thread.join()
        assert seen == [None]


class TestEpochMemo:
    def test_passthrough_outside_epoch(self):
        owner = Owner()
        assert owner.double(2) == 4
        assert owner.double(2) == 4
        assert owner.calls == 2

    def test_memoizes_inside_epoch(self):
        owner = Owner()
        with _epoch.evaluation_epoch():
            assert owner.double(2) == 4
            assert owner.double(2) == 4
            assert owner.double(2, scale=3) == 12
            assert owner.double(3) == 6
        assert owner.calls == 3

    def test_owners_do_not_share_entries(self):
        first, second = Owner(), Owner()
        with _epoch.evaluation_epoch():
            first.double(2)
            second.double(2)
        assert (first.calls, second.calls) == (1, 1)

    def test_arrays_keyed_by_identity(self):
        owner = Owner()
        x = np.ones(3)
        with _epoch.evaluation_epoch():
            owner.double(x)
            owner.double(x)
            owner.double(np.ones(3))
        assert owner.calls == 2

    def test_unhashable_arguments_pass_through(self):
        owner = Owner()
        with _epoch.evaluation_epoch():
            owner.double(1, scale=[1])
            owner.double(1, scale=[1])
        assert owner.calls == 2

    def test_custom_key_fn(self):
        class Rounded:
            calls = 0

            @_epoch.epoch_memo(key_fn=lambda x: round(x))
            def f(self, x):
                self.calls += 1
                return x

        owner = Rounded()
        with _epoch.evaluation_epoch():
            assert owner.f(1.1) == 1.1
            assert owner.f(0.9) == 1.1
        assert owner.calls == 1

    def test_torch_key_includes_requires_grad(self):
        torch = pytest.importorskip("torch")
        x = torch.ones(2)
        key = _epoch.array_key(x)
        assert key[-1] is False
        assert _epoch.array_key(x.requires_grad_())[-1] is True

    def test_torch_reuse_keeps_autograd_correct(self):
        torch = pytest.importorskip("torch")
        owner = Owner()
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        with _epoch.evaluation_epoch():
            total = owner.double(x).sum() + owner.double(x).sum()
        total.backward()
        assert owner.calls == 1
        assert torch.equal(x.grad, torch.tensor([4.0, 4.0]))


class TestDebugGuard:
    def test_guard_off_by_default(self, monkeypatch):
        monkeypatch.delenv(_epoch.CHECK_ENV, raising=False)
        assert not _epoch.check_enabled()
        with _epoch.evaluation_epoch():
            _epoch.assert_writable()

    def test_write_inside_epoch_raises(self, epoch_check):
        _epoch.assert_writable()  # no epoch open: fine
        with (
            pytest.raises(_epoch.EpochViolationError),
            _epoch.evaluation_epoch(),
        ):
            _epoch.assert_writable()

    def test_variable_update_inside_epoch_raises(self, epoch_check):
        from optiland import optimization
        from optiland.samples.objectives import CookeTriplet

        lens = CookeTriplet()
        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=1)
        variable = problem.variables[0]
        with (
            pytest.raises(_epoch.EpochViolationError),
            _epoch.evaluation_epoch([lens]),
        ):
            variable.update(variable.value)

    def test_state_change_inside_epoch_raises(self, epoch_check):
        from optiland.samples.objectives import CookeTriplet

        lens = CookeTriplet()
        with (
            pytest.raises(_epoch.EpochViolationError),
            _epoch.evaluation_epoch([lens]),
        ):
            lens.updater.set_radius(25.0, 1)

    def test_unchanged_state_passes(self, epoch_check):
        from optiland.samples.objectives import CookeTriplet

        lens = CookeTriplet()
        with _epoch.evaluation_epoch([lens]):
            lens.paraxial.f2()

    def test_fingerprint_tracks_state(self):
        from optiland.samples.objectives import CookeTriplet

        lens = CookeTriplet()
        before = _epoch.fingerprint(lens)
        assert _epoch.fingerprint(lens) == before
        lens.updater.set_radius(25.0, 1)
        assert _epoch.fingerprint(lens) != before


class TestMaterialEpochMemo:
    def test_values_match_regular_path(self, set_test_backend):
        material = CountingMaterial()
        scalar = 0.55
        array = be.linspace(0.45, 0.65, 7)
        uniform = be.full((5,), 0.55)
        expected = [material.n(w) for w in (scalar, array, uniform)]
        with _epoch.evaluation_epoch():
            for _ in range(2):
                got = [material.n(w) for w in (scalar, array, uniform)]
                for g, e in zip(got, expected, strict=True):
                    assert_allclose(g, e)
        assert be.shape(got[2]) == (5,)

    def test_regular_cache_untouched_outside_epoch(self):
        material = CountingMaterial()
        material.n(0.55)
        material.n(0.55)
        assert material.calls == 1
        material.a = 1.6
        assert_allclose(material.n(0.55), 1.6 + 0.004 / 0.55**2)

    def test_scalar_memo_skips_regular_path(self):
        material = CountingMaterial()
        with _instrument.counting() as counts, _epoch.evaluation_epoch():
            for _ in range(5):
                material.n(0.55)
        assert counts["material_n"] == 1

    def test_equal_constant_arrays_share_one_evaluation(self):
        material = CountingMaterial()
        with _instrument.counting() as counts, _epoch.evaluation_epoch():
            for _ in range(4):
                n = material.n(np.full(9, 0.55))
                assert n.shape == (9,)
            material.n(np.full(4, 0.55))
            material.n(np.full(9, 0.65))
        assert counts["material_n"] == 2

    def test_varying_arrays_keyed_by_identity(self):
        material = CountingMaterial()
        array = np.linspace(0.45, 0.65, 7)
        with _instrument.counting() as counts, _epoch.evaluation_epoch():
            material.n(array)
            material.n(array)
            other = material.n(np.linspace(0.5, 0.6, 7))
        assert counts["material_n"] == 2
        assert_allclose(other, material.n(np.linspace(0.5, 0.6, 7)))

    def test_kwargs_are_part_of_the_key(self):
        material = CountingMaterial()
        with _instrument.counting() as counts, _epoch.evaluation_epoch():
            material.n(0.55, temperature=20.0)
            material.n(0.55, temperature=20.0)
            material.n(0.55, temperature=30.0)
        assert counts["material_n"] == 2

    def test_k_is_memoized_separately(self):
        material = CountingMaterial()
        with _epoch.evaluation_epoch() as epoch:
            material.n(0.55)
            material.k(0.55)
            material.k(0.55)
        assert (epoch.misses, epoch.hits) == (2, 1)

    def test_trainable_wavelength_bypasses_memo(self):
        torch = pytest.importorskip("torch")
        be.set_backend("torch")
        try:
            material = CountingMaterial()
            wavelength = torch.tensor([0.55, 0.6], requires_grad=True)
            with _instrument.counting() as counts, _epoch.evaluation_epoch():
                material.n(wavelength).sum().backward()
                material.n(wavelength)
            assert counts["material_n"] == 2
            assert wavelength.grad is not None
        finally:
            be.set_backend("numpy")
