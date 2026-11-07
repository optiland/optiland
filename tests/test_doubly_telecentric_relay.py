"""First-order and chief-ray wavefront results for a doubly telecentric relay.

In an exact 4f relay the entrance and exit pupils are both at infinity. The
paraxial marginal ray must take its slope from the object-space NA, and the
chief-ray reference sphere must stay finite.
"""

from __future__ import annotations

import math

import pytest

import optiland.backend as be
from optiland.mtf import FFTMTF
from optiland.optic import Optic
from optiland.wavefront import OPD
from optiland.wavefront.strategy import chief_reference_radius
from tests.utils import assert_allclose

NA = 0.02


def _relay(shift: float = 0.0, telecentric: bool = True) -> Optic:
    """4f relay, f1 = 100 and f2 = 50; the stop sits ``shift`` past the focus."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=100.0)
    optic.surfaces.add(
        index=1, surface_type="paraxial", f=100.0, thickness=100.0 + shift
    )
    optic.surfaces.add(index=2, thickness=50.0 - shift, is_stop=True)
    optic.surfaces.add(index=3, surface_type="paraxial", f=50.0, thickness=50.0)
    optic.surfaces.add(index=4)
    optic.set_aperture("objectNA", NA)
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


def test_chief_reference_radius_finite_pupil(set_test_backend):
    radius = chief_reference_radius(
        (be.array(0.0), be.array(3.0), be.array(0.0)),
        (be.array(0.0), be.array(0.0), be.array(4.0)),
        be.array(10.0),
        1.0,
    )
    assert_allclose(radius, 5.0)


def test_chief_reference_radius_pupil_at_infinity(set_test_backend):
    radius = chief_reference_radius(
        (be.array(0.0), be.array(0.0), be.array(0.0)),
        (be.array(math.nan), be.array(math.nan), be.array(-math.inf)),
        be.array(-300.0),
        1.5,
    )
    assert_allclose(radius, 200.0)
    with pytest.raises(ValueError, match="exit pupil is at infinity"):
        chief_reference_radius(
            (be.array(0.0), be.array(0.0), be.array(0.0)),
            (be.array(math.nan), be.array(math.nan), be.array(-math.inf)),
            None,
            1.0,
        )
