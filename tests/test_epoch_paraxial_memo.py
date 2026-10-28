"""Evaluation-epoch memo of paraxial, aperture and field quantities (PR2.2).

Each memoized method is keyed by every argument (defaults applied, defaults
standing for optic state resolved) and by its hidden inputs: backend,
precision, grad mode and, where it changes the outcome, the paraxial seed
scope.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import optiland.backend as be
from optiland import _epoch, _instrument
from optiland.aperture import ImageFNOAperture
from optiland.paraxial_path import (
    ParaxialDomainWarning,
    UnsupportedParaxialGeometryError,
    paraxial_seed_key,
    paraxial_seed_scope,
)
from optiland.samples.objectives import CookeTriplet

from .test_folded_paraxial_hardening import folded_steering_plate
from .utils import assert_allclose

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

QUANTITIES = (
    "f1",
    "f2",
    "EPL",
    "XPL",
    "EPD",
    "XPD",
    "FNO",
    "marginal_ray",
    "chief_ray",
    "entrance_pupil_axial_position",
)


def _values(paraxial) -> list:
    """Every quantity as numpy, the ray tuples flattened to their arrays."""
    values = []
    for name in QUANTITIES:
        value = getattr(paraxial, name)()
        parts = value if isinstance(value, tuple) else (value,)
        values.extend(be.to_numpy(part) for part in parts)
    return values


class Owner:
    def __init__(self):
        self.calls = 0

    @_epoch.epoch_memo()
    def f(self, x, y=1):
        self.calls += 1
        return x * y

    @_epoch.epoch_memo()
    def same(self, x):
        self.calls += 1
        return x


class TestKeys:
    def test_positional_keyword_and_default_spellings_share_a_key(self):
        owner = Owner()
        with _epoch.evaluation_epoch():
            owner.f(2)
            owner.f(2, 1)
            owner.f(x=2, y=1)
            owner.f(2, y=3)
        assert owner.calls == 2

    def test_objects_key_by_identity_and_stay_alive(self):
        @dataclass(frozen=True)
        class Snapshot:
            values: np.ndarray

        owner = Owner()
        first, second = Snapshot(np.ones(2)), Snapshot(np.ones(2))
        with _epoch.evaluation_epoch() as epoch:
            owner.same(first)
            owner.same(first)
            owner.same(second)
            kept = [keep for _, keep, _ in epoch._memo.values()]
        assert owner.calls == 2
        assert kept == [(first,), (second,)]

    def test_resolve_adds_hidden_inputs(self):
        state = {"mode": "a"}

        class Resolved:
            calls = 0

            @_epoch.epoch_memo(
                resolve=lambda self, args: args.update({"<mode>": state["mode"]})
            )
            def g(self):
                self.calls += 1
                return state["mode"]

        owner = Resolved()
        with _epoch.evaluation_epoch():
            assert owner.g() == "a"
            state["mode"] = "b"
            assert owner.g() == "b"
            assert owner.g() == "b"
        assert owner.calls == 2

    def test_bad_call_raises_the_method_error(self):
        with _epoch.evaluation_epoch(), pytest.raises(TypeError):
            Owner().f()

    def test_backend_is_a_hidden_input(self):
        pytest.importorskip("torch")
        owner = Owner()
        try:
            with _epoch.evaluation_epoch():
                be.set_backend("numpy")
                owner.f(2.0)
                be.set_backend("torch")
                owner.f(2.0)
        finally:
            be.set_backend("numpy")
        assert owner.calls == 2

    def test_grad_mode_is_a_hidden_input(self):
        torch = pytest.importorskip("torch")
        owner = Owner()
        x = torch.tensor([1.0], requires_grad=True)
        try:
            be.set_backend("torch")
            with _epoch.evaluation_epoch():
                with torch.no_grad():
                    detached = owner.f(x)
                tracked = owner.f(x)
        finally:
            be.set_backend("numpy")
        assert owner.calls == 2
        assert not detached.requires_grad
        assert tracked.requires_grad


class TestParaxialMemo:
    def test_values_match_outside_epoch(self, set_test_backend):
        optic = CookeTriplet()
        expected = _values(optic.paraxial)
        with _epoch.evaluation_epoch():
            first = _values(optic.paraxial)
            again = _values(optic.paraxial)
        for value, memo, repeat in zip(expected, first, again, strict=True):
            assert_allclose(memo, value, rtol=1e-12, atol=0.0)
            assert_allclose(repeat, value, rtol=1e-12, atol=0.0)

    def test_no_memo_outside_epoch(self, set_test_backend):
        optic = CookeTriplet()
        with _instrument.counting() as counts:
            optic.paraxial.f2()
            optic.paraxial.f2()
        assert counts["paraxial_trace"] == 2

    def test_path_is_shared_inside_epoch_only(self, set_test_backend):
        surfaces = CookeTriplet().surfaces
        assert surfaces.build_paraxial_path() is not surfaces.build_paraxial_path()
        with _epoch.evaluation_epoch():
            path = surfaces.build_paraxial_path()
            assert surfaces.build_paraxial_path() is path

    def test_explicit_and_default_path_share_an_entry(self, set_test_backend):
        optic = CookeTriplet()
        with _epoch.evaluation_epoch(), _instrument.counting() as counts:
            optic.paraxial.EPL()
            optic.paraxial.EPL(path=optic.surfaces.build_paraxial_path())
            optic.paraxial.EPL(optic.surfaces.build_paraxial_path())
        assert counts["paraxial_trace"] == 1

    def test_state_change_between_epochs_is_seen(self, set_test_backend):
        optic = CookeTriplet()
        with _epoch.evaluation_epoch():
            before = be.to_numpy(optic.paraxial.f2())
        optic.surfaces[1].geometry.radius = be.array(20.0)
        with _epoch.evaluation_epoch():
            after = be.to_numpy(optic.paraxial.f2())
        assert after != before
        assert_allclose(after, optic.paraxial.f2())

    def test_seed_scope_moot_in_the_scalar_domain(self, set_test_backend):
        optic = CookeTriplet()
        assert paraxial_seed_key(optic.surfaces) is None
        with _epoch.evaluation_epoch(), _instrument.counting() as counts:
            with paraxial_seed_scope():
                optic.paraxial.f2()
            optic.paraxial.f2()
        assert counts["paraxial_trace"] == 1

    def test_seed_scope_keyed_where_it_changes_the_outcome(self, set_test_backend):
        # A value memoized as a seed must not hide the strict rejection.
        optic = folded_steering_plate()
        with paraxial_seed_scope():
            assert paraxial_seed_key(optic.surfaces) is True
        assert paraxial_seed_key(optic.surfaces) is False
        with _epoch.evaluation_epoch():
            with paraxial_seed_scope(), pytest.warns(ParaxialDomainWarning):
                optic.paraxial.f2()
            with pytest.raises(UnsupportedParaxialGeometryError):
                optic.paraxial.f2()


class TestApertureAndFieldMemo:
    def test_shared_aperture_keys_on_the_paraxial_engine(self, set_test_backend):
        first, second = CookeTriplet(), CookeTriplet()
        second.surfaces[1].geometry.radius = be.array(20.0)
        aperture = ImageFNOAperture(5.0)
        first.aperture = second.aperture = aperture
        expected = [
            be.to_numpy(aperture.compute_epd(optic.paraxial))
            for optic in (first, second)
        ]
        with _epoch.evaluation_epoch():
            memo = [
                be.to_numpy(aperture.compute_epd(optic.paraxial))
                for optic in (first, second)
            ]
        assert expected[0] != expected[1]
        assert_allclose(memo, expected)

    def test_default_wavelength_resolves_to_primary(self, set_test_backend):
        optic = CookeTriplet()
        optic.set_aperture(aperture_type="float_by_stop_size", value=10.0)
        paraxial = optic.paraxial
        with _epoch.evaluation_epoch(), _instrument.counting() as counts:
            optic.aperture.compute_epd(paraxial)
            optic.aperture.compute_epd(paraxial, optic.primary_wavelength)
            optic.aperture.compute_epd(paraxial, wavelength=None)
        assert counts["paraxial_trace"] == 1

    def test_field_scales_memoized_per_optic(self, set_test_backend):
        optic = CookeTriplet()
        optic.fields.set_type("paraxial_image_height")
        definition = optic.fields.field_definition
        expected = definition._trace_unit_chief_ray(optic, plane="object")
        with _epoch.evaluation_epoch(), _instrument.counting() as counts:
            for _ in range(3):
                image = definition._trace_unit_chief_ray(optic, plane="image")
                obj = definition._trace_unit_chief_ray(optic, plane="object")
        assert counts["paraxial_trace"] == 2
        assert image is not obj
        assert_allclose(obj[0], expected[0])
        assert_allclose(obj[1], expected[1])

    def test_max_field_memoized(self, set_test_backend):
        fields = CookeTriplet().fields
        with _epoch.evaluation_epoch():
            assert fields.max_field is fields.max_field
            assert fields.max_y_field is fields.max_y_field
