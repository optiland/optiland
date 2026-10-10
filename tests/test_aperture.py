from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.aperture import (
    BaseSystemAperture,
    EPDAperture,
    FloatByStopAperture,
    ImageFNOAperture,
    ObjectNAAperture,
    make_system_aperture,
)
from optiland.optic import Optic

from .utils import assert_allclose

# ── Factory tests ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ap_type, value",
    [
        ("EPD", 10),
        ("imageFNO", 3.2),
        ("objectNA", 0.265),
        ("float_by_stop_size", 1.142857),
    ],
)
def test_aperture_generate(set_test_backend, ap_type, value):
    """Check instantiation via factory for all aperture types."""
    ap = make_system_aperture(ap_type, value)
    assert ap.value == value
    assert ap.ap_type == ap_type


def test_confirm_invalid_ap_type(set_test_backend):
    """Confirm invalid ap_type raises ValueError."""
    with pytest.raises(ValueError):
        make_system_aperture("invalid_type", 5.0)


# ── Capability flags ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ap_type, expected_telecentric, expected_scalable",
    [
        ("EPD", False, True),
        ("imageFNO", False, False),
        ("objectNA", True, False),
        ("float_by_stop_size", True, True),
    ],
)
def test_capability_flags(
    set_test_backend, ap_type, expected_telecentric, expected_scalable
):
    """Confirm supports_telecentric and is_scalable match expectations."""
    ap = make_system_aperture(ap_type, 5.0)
    assert ap.supports_telecentric is expected_telecentric
    assert ap.is_scalable is expected_scalable


def test_obj_space_telecentric_via_capability(set_test_backend):
    """Only EPD and imageFNO are incompatible with telecentric object space."""
    assert not EPDAperture(5.0).supports_telecentric
    assert not ImageFNOAperture(5.0).supports_telecentric
    assert ObjectNAAperture(0.1).supports_telecentric is True
    assert FloatByStopAperture(5.0).supports_telecentric is True


# ── direct_fno ────────────────────────────────────────────────────────────────


def test_direct_fno_image_fno(set_test_backend):
    """ImageFNOAperture.direct_fno() returns its value directly."""
    ap = ImageFNOAperture(3.2)
    assert ap.direct_fno() == 3.2


@pytest.mark.parametrize("ap_type", ["EPD", "objectNA", "float_by_stop_size"])
def test_direct_fno_returns_none(set_test_backend, ap_type):
    """Non-imageFNO apertures return None from direct_fno()."""
    ap = make_system_aperture(ap_type, 5.0)
    assert ap.direct_fno() is None


# ── Scaling ───────────────────────────────────────────────────────────────────


def test_epd_scale(set_test_backend):
    ap = EPDAperture(10.0)
    scaled = ap.scale(2.0)
    assert isinstance(scaled, EPDAperture)
    assert scaled.value == 20.0
    assert ap.value == 10.0  # original unchanged


def test_float_by_stop_scale(set_test_backend):
    ap = FloatByStopAperture(8.4)
    scaled = ap.scale(0.5)
    assert isinstance(scaled, FloatByStopAperture)
    assert scaled.value == pytest.approx(4.2)


def test_image_fno_scale_returns_self(set_test_backend):
    """Non-scalable types return self unchanged."""
    ap = ImageFNOAperture(5.6)
    assert ap.scale(2.0) is ap


def test_object_na_scale_returns_self(set_test_backend):
    ap = ObjectNAAperture(0.1)
    assert ap.scale(0.001) is ap


# ── Serialization ─────────────────────────────────────────────────────────────


def test_to_dict(set_test_backend):
    """Check to_dict produces the expected schema."""
    ap = EPDAperture(10)
    assert ap.to_dict() == {"type": "EPD", "value": 10}


def test_from_dict_roundtrip(set_test_backend):
    """Check to_dict / from_dict round-trip for all types."""
    for ap_type, value in [
        ("EPD", 25.0),
        ("imageFNO", 4.0),
        ("objectNA", 0.133),
        ("float_by_stop_size", 8.4),
    ]:
        ap = make_system_aperture(ap_type, value)
        data = ap.to_dict()
        ap2 = BaseSystemAperture.from_dict(data)
        assert ap2.ap_type == ap_type
        assert ap2.value == value
        assert ap2.to_dict() == data


def test_from_dict(set_test_backend):
    """Check from_dict method."""
    ap = EPDAperture(10)
    ap_dict = ap.to_dict()
    ap2 = BaseSystemAperture.from_dict(ap_dict)
    assert ap2.to_dict() == ap.to_dict()


def test_from_dict_none(set_test_backend):
    """from_dict(None) returns None."""
    assert BaseSystemAperture.from_dict(None) is None


def test_invalid_from_dict(set_test_backend):
    """Check from_dict raises ValueError for invalid dict."""
    with pytest.raises(ValueError):
        BaseSystemAperture.from_dict(
            {"invalid": "I am invalid, unfortunately", "value": 5.0},
        )


def test_from_dict_unknown_type(set_test_backend):
    """Check from_dict raises ValueError for unknown type string."""
    with pytest.raises(ValueError):
        BaseSystemAperture.from_dict({"type": "UnknownType", "value": 5.0})


# ── Integration: float_by_stop_size ──────────────────────────────────────────


def test_confirm_stop_size_floating_stop(set_test_backend):
    lens = Optic()

    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(index=1, radius=22.01359, thickness=3.25896, material="SK16")
    lens.surfaces.add(index=2, radius=-435.76044, thickness=30, is_stop=True)
    lens.surfaces.add(index=3)

    stop_diam = 8.4
    stop_idx = 2
    lens.set_aperture(aperture_type="float_by_stop_size", value=stop_diam)

    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.fields.add(y=14)
    lens.fields.add(y=20)

    lens.wavelengths.add(value=0.48)
    lens.wavelengths.add(value=0.55, is_primary=True)
    lens.wavelengths.add(value=0.65)

    lens.paraxial.trace(Hy=0, Py=1, wavelength=0.55)
    assert_allclose(lens.surfaces.y[stop_idx], stop_diam / 2)


# ── Integration: imageFNO with an odd number of mirrors ──────────────────────


def _paraboloid(aperture_type, value):
    """A single parabolic mirror with f = 200 mm: one mirror, so f2 < 0."""
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=-400.0,
        thickness=-200.0,
        conic=-1.0,
        material="mirror",
        is_stop=True,
    )
    lens.surfaces.add(index=2)
    lens.set_aperture(aperture_type=aperture_type, value=value)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.fields.add(y=1.0)
    lens.wavelengths.add(value=0.5876, is_primary=True)
    return lens


def test_image_fno_epd_is_positive_with_odd_number_of_mirrors(set_test_backend):
    """The EPD is a diameter: positive even when f2 < 0 (GH #918)."""
    lens = _paraboloid("imageFNO", 8.0)
    assert_allclose(lens.paraxial.f2(), -200.0)
    assert_allclose(lens.paraxial.EPD(), 25.0)
    assert_allclose(lens.paraxial.FNO(), 8.0)


def test_image_fno_matches_equivalent_epd_with_odd_number_of_mirrors(
    set_test_backend,
):
    """An f/8 mirror is the same system set up as imageFNO = 8 or as EPD = 25.

    With a negative EPD the marginal ray started below the axis, which flipped
    the sign of the Lagrange invariant (and with it third-order coma and
    distortion) and mirrored the pupil for real rays (GH #918).
    """
    by_fno = _paraboloid("imageFNO", 8.0)
    by_epd = _paraboloid("EPD", 25.0)

    # First order: pupils, Lagrange invariant and marginal ray.
    for name in ("EPD", "XPD", "invariant"):
        assert_allclose(
            getattr(by_fno.paraxial, name)(), getattr(by_epd.paraxial, name)()
        )
    for fno, epd in zip(
        by_fno.paraxial.marginal_ray(), by_epd.paraxial.marginal_ray(), strict=True
    ):
        assert_allclose(fno, epd)

    # Third order: the signs of coma and distortion follow the invariant.
    assert_allclose(
        by_fno.aberrations.seidels(), by_epd.aberrations.seidels(), atol=1e-12
    )

    # Real rays: Py = +1 is the +y edge of the pupil in both.
    for lens in (by_fno, by_epd):
        lens.trace_generic(Hx=0, Hy=1, Px=0, Py=1, wavelength=0.5876)
    assert_allclose(by_fno.surfaces.y[1], by_epd.surfaces.y[1])
    assert_allclose(by_fno.surfaces.y[-1], by_epd.surfaces.y[-1])
