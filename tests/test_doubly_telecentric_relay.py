"""First-order and chief-ray wavefront results for a doubly telecentric relay.

In an exact 4f relay the entrance and exit pupils are both at infinity. The
paraxial marginal ray must take its slope from the object-space NA, and the
chief-ray OPD must be the limit of an infinite reference radius.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import optiland.backend as be
from optiland.mtf import FFTMTF
from optiland.optic import Optic
from optiland.wavefront import OPD
from optiland.wavefront.reference_geometry import (
    SphericalReference,
    TelecentricSphericalReference,
)
from optiland.wavefront.strategy import chief_reference_sphere
from tests.utils import assert_allclose

NA = 0.02
# Stop diameter that sets the same object-space cone as NA: the stop sits at the
# back focal plane of L1 (f1 = 100), so its semi-diameter is f1 * tan(u).
STOP_DIAMETER = 4.0


def _relay(
    shift: float = 0.0,
    telecentric: bool = True,
    defocus: float = 0.0,
    aperture: str = "objectNA",
) -> Optic:
    """4f relay, f1 = 100 and f2 = 50; the stop sits ``shift`` past the focus.

    ``defocus`` moves the image plane away from the second lens.
    """
    optic = Optic()
    optic.surfaces.add(index=0, thickness=100.0)
    optic.surfaces.add(
        index=1, surface_type="paraxial", f=100.0, thickness=100.0 + shift
    )
    optic.surfaces.add(index=2, thickness=50.0 - shift, is_stop=True)
    optic.surfaces.add(
        index=3, surface_type="paraxial", f=50.0, thickness=50.0 + defocus
    )
    optic.surfaces.add(index=4)
    if aperture == "objectNA":
        optic.set_aperture("objectNA", NA)
    else:
        optic.set_aperture("float_by_stop_size", STOP_DIAMETER)
    optic.fields.set_type("object_height")
    optic.obj_space_telecentric = telecentric
    optic.fields.add(y=0.0)
    optic.fields.add(y=5.0)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _scalar(value) -> float:
    return float(be.to_numpy(be.ravel(be.as_array_1d(value)))[0])


def test_pupils_are_at_infinity(set_test_backend):
    optic = _relay()
    assert not math.isfinite(_scalar(optic.paraxial.EPL()))
    assert not math.isfinite(_scalar(optic.paraxial.XPL()))


def test_magnification_is_finite(set_test_backend):
    assert_allclose(_relay().paraxial.magnification(), -0.5, atol=1e-12)


def test_marginal_ray_slope_from_object_na(set_test_backend):
    ya, ua = _relay().paraxial.marginal_ray()
    slope = math.tan(math.asin(NA))
    assert_allclose(ua[0], slope, atol=1e-15)
    assert_allclose(ya[1], 100.0 * slope, atol=1e-12)
    assert_allclose(ya[-1], 0.0, atol=1e-12)
    assert bool(be.all(be.isfinite(ya))) and bool(be.all(be.isfinite(ua)))


def test_marginal_ray_slope_from_stop_size(set_test_backend):
    ya, ua = _relay(aperture="float_by_stop_size").paraxial.marginal_ray()
    assert_allclose(ua[0], STOP_DIAMETER / 2 / 100.0, atol=1e-15)
    assert_allclose(ya[2], STOP_DIAMETER / 2, atol=1e-12)
    assert_allclose(ya[-1], 0.0, atol=1e-12)
    assert bool(be.all(be.isfinite(ya))) and bool(be.all(be.isfinite(ua)))


def test_stop_size_magnification_is_finite(set_test_backend):
    optic = _relay(aperture="float_by_stop_size")
    assert_allclose(optic.paraxial.magnification(), -0.5, atol=1e-12)


@pytest.mark.parametrize("shift", [-1e-6, 1e-6])
def test_stop_size_marginal_ray_continuous_with_near_telecentric(
    set_test_backend, shift
):
    _, exact = _relay(aperture="float_by_stop_size").paraxial.marginal_ray()
    near = _relay(shift=shift, telecentric=False, aperture="float_by_stop_size")
    _, ua = near.paraxial.marginal_ray()
    assert_allclose(exact[0], ua[0], rtol=1e-6)


def test_stop_size_slope_is_none_for_object_at_infinity(set_test_backend):
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(index=1, surface_type="paraxial", f=100.0, thickness=100.0)
    optic.surfaces.add(index=2, is_stop=True)
    optic.set_aperture("float_by_stop_size", STOP_DIAMETER)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(0.55, is_primary=True)
    assert optic.aperture.object_space_marginal_slope(optic.paraxial) is None


def test_stop_size_slope_requires_object_surface(set_test_backend):
    optic = _relay(aperture="float_by_stop_size")
    paraxial = SimpleNamespace(optic=SimpleNamespace(object_surface=None))
    with pytest.raises(ValueError, match="requires a defined object surface"):
        optic.aperture.object_space_marginal_slope(paraxial)


def test_aperture_without_fixed_cone_raises(set_test_backend):
    optic = _relay(telecentric=False)
    optic.set_aperture("EPD", 4.0)
    with pytest.raises(ValueError, match="entrance pupil is at infinity"):
        optic.paraxial.marginal_ray()


def test_chief_ray_opd_on_axis_is_zero(set_test_backend):
    opd = OPD(_relay(), (0, 0), "primary", strategy="chief_ray")
    assert_allclose(opd.rms(), 0.0, atol=1e-9)


def test_chief_ray_opd_continuous_with_near_telecentric(set_test_backend):
    exact = OPD(_relay(), (0, 1), "primary", strategy="chief_ray")
    near = OPD(
        _relay(shift=1e-6, telecentric=False), (0, 1), "primary", strategy="chief_ray"
    )
    assert math.isfinite(_scalar(exact.rms()))
    assert_allclose(exact.rms(), near.rms(), rtol=1e-6)


def test_fft_mtf_default_strategy(set_test_backend):
    mtf = FFTMTF(_relay(), num_rays=32, max_freq=100)
    near = FFTMTF(_relay(shift=1e-6, telecentric=False), num_rays=32, max_freq=100)
    for exact_curve, near_curve in zip(mtf.mtf, near.mtf, strict=True):
        for a, b in zip(exact_curve, near_curve, strict=True):
            assert_allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("shift", [-1e-6, 1e-6])
def test_defocused_chief_ray_opd_is_continuous(set_test_backend, shift):
    exact = OPD(_relay(defocus=10.0), (0, 1), 0.55, strategy="chief_ray")
    near = OPD(
        _relay(shift=shift, telecentric=False, defocus=10.0),
        (0, 1),
        0.55,
        strategy="chief_ray",
    )
    assert_allclose(exact.rms(), near.rms(), atol=1e-6, rtol=0)


def test_defocused_fft_mtf_is_continuous(set_test_backend):
    exact = FFTMTF(_relay(defocus=10.0), num_rays=32, max_freq=50)
    near = FFTMTF(
        _relay(shift=1e-6, telecentric=False, defocus=10.0), num_rays=32, max_freq=50
    )
    for exact_curve, near_curve in zip(exact.mtf, near.mtf, strict=True):
        for a, b in zip(exact_curve, near_curve, strict=True):
            assert_allclose(a, b, atol=1e-6)


def test_chief_reference_sphere_finite_pupil(set_test_backend):
    ref = chief_reference_sphere(
        (be.array(0.0), be.array(3.0), be.array(0.0)),
        (be.array(0.0), be.array(0.0), be.array(4.0)),
        be.array(10.0),
        1.0,
    )
    assert type(ref) is SphericalReference
    assert_allclose(ref.radius, 5.0)


def test_chief_reference_sphere_pupil_at_infinity(set_test_backend):
    ref = chief_reference_sphere(
        (be.array(0.0), be.array(0.0), be.array(0.0)),
        (be.array(math.nan), be.array(math.nan), be.array(-math.inf)),
        be.array(-300.0),
        1.5,
    )
    assert isinstance(ref, TelecentricSphericalReference)
    # The finite radius only places the pupil samples.
    assert_allclose(ref.radius, 200.0)
    with pytest.raises(ValueError, match="exit pupil is at infinity"):
        chief_reference_sphere(
            (be.array(0.0), be.array(0.0), be.array(0.0)),
            (be.array(math.nan), be.array(math.nan), be.array(-math.inf)),
            None,
            1.0,
        )


def test_telecentric_reference_is_limit_of_large_sphere(set_test_backend):
    rays = SimpleNamespace(
        x=be.array([0.3, -1.2]),
        y=be.array([2.0, 0.4]),
        z=be.array([10.0, 10.0]),
        L=be.array([0.01, -0.02]),
        M=be.array([0.015, 0.005]),
        N=be.array(
            [math.sqrt(1 - 0.01**2 - 0.015**2), math.sqrt(1 - 0.02**2 - 0.005**2)]
        ),
    )
    center = (0.0, 0.5, 0.0)
    limit = TelecentricSphericalReference(center, 300.0)._relative_path_length(
        rays, 1.5
    )
    large = SphericalReference(center, 1e12)._relative_path_length(rays, 1.5)
    assert_allclose(limit, large, atol=1e-9)
