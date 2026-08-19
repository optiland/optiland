from __future__ import annotations

from optiland import backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.nurbs.nurbs_geometry import NurbsGeometry
from tests.utils import assert_allclose


def test_nurbs_geometry_init(set_test_backend):
    cs = CoordinateSystem()
    geo = NurbsGeometry(
        cs, nurbs_norm_x=20, nurbs_norm_y=20, n_points_u=10, n_points_v=10
    )
    geo.fit_surface()
    assert geo is not None


def test_nurbs_geometry_sag(set_test_backend):
    cs = CoordinateSystem()
    geo = NurbsGeometry(
        cs,
        radius=100,
        conic=-1,
        nurbs_norm_x=20,
        nurbs_norm_y=20,
        n_points_u=10,
        n_points_v=10,
    )
    geo.fit_surface()
    sag = geo.sag(
        x=be.asarray([0, 10], dtype=be.float64), y=be.asarray([0, 0], dtype=be.float64)
    )
    assert_allclose(sag[0], 0.0, atol=1e-4)
    assert_allclose(sag[1], 0.5, atol=1e-4)


class MockRays:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_nurbs_geometry_normal(set_test_backend):
    cs = CoordinateSystem()
    geo = NurbsGeometry(
        cs,
        radius=100,
        conic=-1,
        nurbs_norm_x=20,
        nurbs_norm_y=20,
        n_points_u=10,
        n_points_v=10,
    )
    geo.fit_surface()
    rays = MockRays(
        x=be.asarray([0], dtype=be.float64), y=be.asarray([0], dtype=be.float64)
    )
    nx, ny, nz = geo.surface_normal(rays)
    assert_allclose(nx, 0.0, atol=1e-4)
    assert_allclose(ny, 0.0, atol=1e-4)
    assert_allclose(nz, 1.0, atol=1e-4)


def test_nurbs_get_value(set_test_backend):
    cs = CoordinateSystem()
    geo = NurbsGeometry(
        cs,
        radius=100,
        conic=-1,
        nurbs_norm_x=20,
        nurbs_norm_y=20,
        n_points_u=10,
        n_points_v=10,
    )
    geo.fit_surface()
    # Test a point that should be on the surface
    val = geo.get_value(
        u=be.asarray([0.5], dtype=be.float64), v=be.asarray([0.5], dtype=be.float64)
    )
    assert val.shape == (3, 1)


def test_nurbs_get_derivative(set_test_backend):
    cs = CoordinateSystem()
    geo = NurbsGeometry(
        cs,
        radius=100,
        conic=-1,
        nurbs_norm_x=20,
        nurbs_norm_y=20,
        n_points_u=10,
        n_points_v=10,
    )
    geo.fit_surface()
    # Test derivative at a point
    derivative = geo.get_derivative(
        u=be.asarray([0.5], dtype=be.float64),
        v=be.asarray([0.5], dtype=be.float64),
        order_u=1,
        order_v=0,
    )
    assert derivative.shape == (3, 1)


class MockRaysDistance:
    def __init__(self, x, y, z, L, M, N):
        self.x = x
        self.y = y
        self.z = z
        self.L = L
        self.M = M
        self.N = N


def test_nurbs_distance(set_test_backend):
    cs = CoordinateSystem()
    geo = NurbsGeometry(
        cs,
        radius=100,
        conic=-1,
        nurbs_norm_x=20,
        nurbs_norm_y=20,
        n_points_u=10,
        n_points_v=10,
    )
    geo.fit_surface()
    # Test distance to a ray
    rays = MockRaysDistance(
        x=be.asarray([0.0], dtype=be.float64),
        y=be.asarray([0.0], dtype=be.float64),
        z=be.asarray([-10.0], dtype=be.float64),
        L=be.asarray([0.0], dtype=be.float64),
        M=be.asarray([0.0], dtype=be.float64),
        N=be.asarray([1.0], dtype=be.float64),
    )
    distance = geo.distance(rays)
    assert_allclose(distance, 10, atol=1e-4)
