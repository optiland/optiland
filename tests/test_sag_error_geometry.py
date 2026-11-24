"""Tests for SagErrorGeometry: a Zernike sag error added to a base geometry."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    BaseGeometry,
    EvenAsphere,
    Plane,
    SagErrorGeometry,
    StandardGeometry,
)
from optiland.rays import RealRays
from optiland.samples.objectives import CookeTriplet
from tests.utils import assert_allclose

NORM = 5.0


def _bases():
    return {
        "standard": StandardGeometry(CoordinateSystem(), radius=20.0, conic=-0.5),
        "asphere": EvenAsphere(
            CoordinateSystem(), radius=-25.0, conic=0.0, coefficients=[1e-4, -2e-7]
        ),
        "plane": Plane(CoordinateSystem()),
    }


def _coefficients():
    c = np.zeros(11)
    c[4] = 2e-4  # Z5: rho^2 cos(2 phi)
    c[5] = -1e-4  # Z6: rho^2 sin(2 phi)
    c[6] = 5e-5  # Z7: (3 rho^3 - 2 rho) cos(phi)
    c[8] = 3e-4  # Z9: 6 rho^4 - 6 rho^2 + 1
    c[10] = -7e-5  # Z11: rho^3 sin(3 phi)
    return c


def _zernike(x, y):
    """The analytic Fringe Zernike sum of ``_coefficients``."""
    c = _coefficients()
    rho = np.hypot(x, y) / NORM
    phi = np.arctan2(y, x)
    return (
        c[4] * rho**2 * np.cos(2 * phi)
        + c[5] * rho**2 * np.sin(2 * phi)
        + c[6] * (3 * rho**3 - 2 * rho) * np.cos(phi)
        + c[8] * (6 * rho**4 - 6 * rho**2 + 1)
        + c[10] * rho**3 * np.sin(3 * phi)
    )


def _points():
    rng = np.random.default_rng(1)
    r = 4.5 * np.sqrt(rng.uniform(0, 1, 50))
    phi = rng.uniform(0, 2 * np.pi, 50)
    return r * np.cos(phi), r * np.sin(phi)


def _rays():
    rng = np.random.default_rng(2)
    n = 40
    x = rng.uniform(-3, 3, n)
    y = rng.uniform(-3, 3, n)
    L = rng.uniform(-0.1, 0.1, n)
    M = rng.uniform(-0.1, 0.1, n)
    N = np.sqrt(1 - L**2 - M**2)
    return RealRays(
        be.array(x),
        be.array(y),
        be.full((n,), -2.0),
        be.array(L),
        be.array(M),
        be.array(N),
        be.ones(n),
        be.full((n,), 0.55),
    )


def _np(value):
    return np.asarray(be.to_numpy(value))


@pytest.mark.parametrize("name", ["standard", "asphere", "plane"])
def test_zero_coefficients_are_the_base(set_test_backend, name):
    base = _bases()[name]
    geometry = SagErrorGeometry(base, be.zeros(9), NORM)
    x, y = (be.array(v) for v in _points())
    np.testing.assert_array_equal(_np(geometry.sag(x, y)), _np(base.sag(x, y)))
    rays = _rays()
    for a, b in zip(
        geometry.surface_normal(rays), base.surface_normal(rays), strict=True
    ):
        np.testing.assert_array_equal(_np(a), _np(b))
    np.testing.assert_array_equal(
        _np(geometry.distance(_rays())), _np(base.distance(_rays()))
    )
    assert geometry.cs is base.cs
    paraxial = getattr(base, "paraxial_radius", base.radius)
    assert _np(geometry.paraxial_radius) == _np(paraxial)


def test_zero_coefficients_trace_is_the_base(set_test_backend):
    lens = CookeTriplet()
    nominal = lens.trace(0.0, 1.0, 0.55, 64, "hexapolar")
    nominal = [_np(getattr(nominal, a)).copy() for a in "xyzLMN"]
    surface = lens.surfaces[3]
    surface.geometry = SagErrorGeometry(surface.geometry, be.zeros(9), NORM)
    traced = lens.trace(0.0, 1.0, 0.55, 64, "hexapolar")
    for a, b in zip([_np(getattr(traced, a)) for a in "xyzLMN"], nominal, strict=True):
        assert_allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", ["standard", "asphere", "plane"])
def test_sag_is_base_plus_zernike(set_test_backend, name):
    base = _bases()[name]
    geometry = SagErrorGeometry(base, be.array(_coefficients()), NORM)
    x, y = _points()
    expected = _np(base.sag(be.array(x), be.array(y))) + _zernike(x, y)
    assert_allclose(geometry.sag(be.array(x), be.array(y)), expected, atol=1e-14)


@pytest.mark.parametrize("name", ["standard", "asphere", "plane"])
def test_normal_matches_finite_difference(set_test_backend, name):
    geometry = SagErrorGeometry(_bases()[name], be.array(_coefficients()), NORM)
    x, y = _points()
    h = 1e-6

    def sag(u, v):
        return _np(geometry.sag(be.array(u), be.array(v)))

    fx = (sag(x + h, y) - sag(x - h, y)) / (2 * h)
    fy = (sag(x, y + h) - sag(x, y - h)) / (2 * h)
    norm = np.sqrt(1 + fx**2 + fy**2)
    nx, ny, nz = geometry._surface_normal(be.array(x), be.array(y))
    assert_allclose(nx, fx / norm, atol=1e-8)
    assert_allclose(ny, fy / norm, atol=1e-8)
    assert_allclose(nz, -1 / norm, atol=1e-8)


@pytest.mark.parametrize("name", ["standard", "asphere", "plane"])
def test_intersection_lies_on_the_surface(set_test_backend, name):
    geometry = SagErrorGeometry(_bases()[name], be.array(_coefficients()), NORM)
    rays = _rays()
    t = geometry.distance(rays)
    x = rays.x + t * rays.L
    y = rays.y + t * rays.M
    z = rays.z + t * rays.N
    assert np.all(np.isfinite(_np(t)))
    assert np.max(np.abs(_np(geometry.sag(x, y) - z))) < 1e-9
    # the hit differs from the hit on the base
    t_base = geometry.base.distance(_rays())
    assert np.max(np.abs(_np(t - t_base))) > 1e-6


def test_trace_plan_falls_back_to_the_generic_trace():
    be.set_backend("numpy")
    from optiland.surfaces import _trace_plan

    lens = CookeTriplet()
    n = 64
    rays = RealRays(
        np.linspace(-5, 5, n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.zeros(n),
        np.ones(n),
        np.ones(n),
        np.full(n, 0.55),
    )
    surfaces = list(lens.surfaces)
    kernel = _trace_plan._kernel()
    assert _trace_plan._build(surfaces, rays, n, kernel) is not None
    surface = lens.surfaces[3]
    surface.geometry = SagErrorGeometry(
        surface.geometry, np.array(_coefficients()), NORM
    )
    assert _trace_plan._build(surfaces, rays, n, kernel) is None
    # the generic trace runs and hits the perturbed surface
    lens.trace(0.0, 1.0, 0.55, 64, "hexapolar")
    z = _np(surface.z) - float(_np(surface.geometry.cs.z))
    sag = _np(surface.geometry.sag(surface.x, surface.y))
    assert np.nanmax(np.abs(z - sag)) < 1e-9


def test_paraxial_radius_includes_symmetric_power(set_test_backend):
    base = _bases()["standard"]
    c = np.zeros(9)
    c[8] = 1e-3
    c[4] = 1e-3  # astigmatism has no mean power
    geometry = SagErrorGeometry(base, be.array(c), NORM)
    # 1e-3 (-6 rho^2) = (curvature / 2) r^2
    curvature = 1 / 20.0 + 2 * (-6e-3) / NORM**2
    assert_allclose(geometry.paraxial_radius, 1 / curvature)
    assert_allclose(geometry.radius, 20.0)


def test_symmetry(set_test_backend):
    base = _bases()["standard"]
    c = np.zeros(9)
    c[8] = 1e-3
    assert SagErrorGeometry(base, be.array(c), NORM).is_symmetric
    c[4] = 1e-3
    assert not SagErrorGeometry(base, be.array(c), NORM).is_symmetric


def test_radius_and_conic_change_the_base(set_test_backend):
    base = _bases()["standard"]
    geometry = SagErrorGeometry(base, be.array(_coefficients()), NORM)
    geometry.set_radius(30.0)
    geometry.k = -1.0
    assert_allclose(base.radius, 30.0)
    assert_allclose(base.k, -1.0)


def test_flip_and_scale(set_test_backend):
    geometry = SagErrorGeometry(
        _bases()["asphere"], be.array(_coefficients()), NORM
    )
    x, y = (be.array(v) for v in _points())
    sag = _np(geometry.sag(x, y))
    geometry.flip()
    assert_allclose(geometry.sag(x, y), -sag, atol=1e-14)
    geometry.flip()
    geometry.scale(2.0)
    assert_allclose(geometry.norm_radius, 2 * NORM)
    assert_allclose(geometry.sag(2 * x, 2 * y), 2 * sag, atol=1e-13)


@pytest.mark.parametrize("name", ["standard", "asphere", "plane"])
def test_serialization_round_trip(set_test_backend, name):
    geometry = SagErrorGeometry(_bases()[name], be.array(_coefficients()), NORM)
    data = geometry.to_dict()
    restored = BaseGeometry.from_dict(data)
    assert isinstance(restored, SagErrorGeometry)
    assert type(restored.base) is type(geometry.base)
    x, y = (be.array(v) for v in _points())
    assert_allclose(restored.sag(x, y), geometry.sag(x, y), atol=1e-15)
    assert restored.to_dict() == data


def test_invalid_norm_radius():
    with pytest.raises(ValueError, match="norm_radius"):
        SagErrorGeometry(_bases()["standard"], np.zeros(9), 0.0)


def test_from_dict_missing_keys():
    with pytest.raises(ValueError, match="Missing required keys"):
        SagErrorGeometry.from_dict({"type": "SagErrorGeometry"})
