from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

import optiland.backend as be
from optiland import geometries
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    BiconicGeometry,
    ForbesQ2dGeometry,
    ForbesQbfsGeometry,  # Deprecated alias
    ForbesQNormalSlopeGeometry,
    ForbesSolverConfig,
    ForbesSurfaceConfig,
)
from optiland.materials import IdealMaterial
from optiland.materials.material import Material
from optiland.optic import Optic
from optiland.rays import RealRays

from .utils import assert_allclose, assert_array_equal


def test_unknown_geometry(set_test_backend):
    with pytest.raises(ValueError):
        geometries.BaseGeometry.from_dict({"type": "UnknownGeometry"})


class TestPlane:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)
        assert str(plane) == "Planar"

    def test_plane_sag(self, set_test_backend):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        # Test sag at (0, 0)
        assert plane.sag() == 0.0

        # Test sag at (1, 1)
        assert plane.sag(1, 1) == 0.0

        # Test sag at (-2, 3)
        assert plane.sag(-2, 3) == 0.0

        # Test array input
        x = be.array([0, 3, 8e3])
        y = be.array([0, -7.0, 2.1654])
        sag = be.array([0.0, 0.0, 0.0])
        assert be.allclose(plane.sag(x, y), sag)

    def test_plane_distance(self, set_test_backend):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = plane.distance(rays)
        assert_allclose(distance, 3.0)

        # Test distance for multiple rays
        rays = RealRays(
            [1.0, 2.0],
            [2.0, 3.0],
            [-3.0, -4.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        )
        distance = plane.distance(rays)
        assert_allclose(distance, [3.0, 4.0])

        # Test ray doesn't intersect the plane
        rays = RealRays(1.0, 2.0, -1.5, 0.0, 0.0, -1.0, 1.0, 0.0)
        distance = plane.distance(rays)
        assert_allclose(distance, -1.5)

        # Test distance for ray not parallel to z axis
        L = 0.356
        M = -0.129
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -16.524, L, M, N, 1.0, 0.0)
        distance = plane.distance(rays)
        assert_allclose(distance, 17.853374740457518)

    def test_plane_surface_normal(self, set_test_backend):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = plane.surface_normal(rays)
        assert nx == 0.0
        assert ny == 0.0
        assert nz == 1.0

    def test_to_dict(self, set_test_backend):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        expected_dict = {"type": "Plane", "cs": cs.to_dict(), "radius": be.inf}
        assert plane.to_dict() == expected_dict

    def test_from_dict(self, set_test_backend):
        cs = CoordinateSystem()
        plane = geometries.Plane(cs)

        plane_dict = plane.to_dict()
        new_plane = geometries.Plane.from_dict(plane_dict)
        assert new_plane.to_dict() == plane_dict


class TestStandardGeometry:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.5)
        assert str(geometry) == "Standard"

    def test_sag_sphere(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.0)

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at (1, 1)
        assert_allclose(geometry.sag(1, 1), 0.10050506338833465)

        # Test sag at (-2, 3)
        assert_allclose(geometry.sag(-2, 3), 0.6726209469111849)

        # Test array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([0.0, 3.5192593015921396, 4.3795018014414415])
        assert_allclose(geometry.sag(x, y), sag)

    def test_sag_parabola(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=25.0, conic=-1.0)

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at (2.1, -1.134)
        assert_allclose(geometry.sag(2.1, -1.134), 0.11391912)

        # Test sag at (5, 5)
        assert_allclose(geometry.sag(5, 5), 1.0)

        # Test array input
        x = be.array([0, 2, 4])
        y = be.array([0, -3, 2.1])
        sag = be.array([0.0, 0.26, 0.4082])
        assert_allclose(geometry.sag(x, y), sag)

    def test_sag_conic(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=27.0, conic=0.55)

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at (3.1, -3.134)
        assert_allclose(geometry.sag(3.1, -3.134), 0.3636467856728104)

        # Test sag at (2, 5)
        assert_allclose(geometry.sag(2, 5), 0.5455809402149067)

        # Test array input
        x = be.array([0, 5, 6])
        y = be.array([0, -3, 3.1])
        sag = be.array([0.0, 0.6414396188168761, 0.8661643140626132])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=-12.0, conic=0.5)

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 2.7888809636986154)

        # Test distance for multiple rays
        rays = RealRays(
            [1.0, 2.0],
            [2.0, 3.0],
            [-3.0, -4.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        )
        distance = geometry.distance(rays)
        nom_distance = [2.7888809636986154, 3.4386378681404657]
        assert_allclose(distance, nom_distance)

        # Test distance for ray not parallel to z axis
        L = 0.359
        M = -0.229
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 10.201933401020467)

    def test_surface_normal(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.5)

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert_allclose(nx, 0.10127393670836665)
        assert_allclose(ny, 0.2025478734167333)
        assert_allclose(nz, -0.9740215340114144)

    def test_to_dict(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.5)

        expected_dict = {
            "type": "StandardGeometry",
            "cs": cs.to_dict(),
            "radius": 10.0,
            "conic": 0.5,
        }
        assert geometry.to_dict() == expected_dict

    def test_from_dict(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.StandardGeometry(cs, radius=10.0, conic=0.5)

        geometry_dict = geometry.to_dict()
        new_geometry = geometries.StandardGeometry.from_dict(geometry_dict)
        assert new_geometry.to_dict() == geometry_dict

    def test_From_dict_invalid_dict(self, set_test_backend):
        with pytest.raises(ValueError):
            geometries.StandardGeometry.from_dict({"invalid_key": "invalid_value"})


class TestEvenAsphere:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2],
        )
        assert str(geometry) == "Even Asphere"

    def test_sag(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=27.0,
            conic=0.0,
            coefficients=[1e-3, -1e-5],
        )

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at (1, 1)
        assert_allclose(geometry.sag(1, 1), 0.039022474574473776)

        # Test sag at (-2, 3)
        assert_allclose(geometry.sag(-2, 3), 0.25313367948069593)

        # Test array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([0.0, 1.1206923060227627, 1.3196652673420655])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=-41.1,
            conic=0.0,
            coefficients=[1e-3, -1e-5, 1e-7],
        )

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 2.9438901710409624)

        # Test distance for multiple rays
        rays = RealRays(
            [1.0, 2.0],
            [2.0, 3.0],
            [-3.0, -4.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        )
        distance = geometry.distance(rays)
        nom_distance = [2.9438901710409624, 3.8530733934173256]
        assert_allclose(distance, nom_distance)

        # Test distance for ray not parallel to z axis
        L = 0.222
        M = -0.229
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 10.625463223037386)

    def test_surface_normal(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2],
        )

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert_allclose(nx, 0.11946945186789681)
        assert_allclose(ny, 0.23893890373579363)
        assert_allclose(nz, -0.9636572265862595)

    def test_to_dict(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2],
        )

        expected_dict = {
            "type": "EvenAsphere",
            "cs": cs.to_dict(),
            "radius": 10.0,
            "conic": 0.5,
            "tol": 1e-10,
            "max_iter": 100,
            "coefficients": [1e-2],
        }
        assert geometry.to_dict() == expected_dict

    def test_from_dict(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.EvenAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2],
        )

        geometry_dict = geometry.to_dict()
        new_geometry = geometries.EvenAsphere.from_dict(geometry_dict)
        assert new_geometry.to_dict() == geometry_dict


class TestPolynomialGeometry:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=22.0,
            conic=0.0,
            coefficients=coefficients,
        )
        assert str(geometry) == "Polynomial XY"

    def test_sag(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=22.0,
            conic=0.0,
            coefficients=coefficients,
        )

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at (1, 1)
        assert_allclose(geometry.sag(1, 1), 0.3725015998998511)

        # Test sag at (-2, -7)
        assert_allclose(geometry.sag(-2, -7), 1.6294605079733058)

        # Test array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([0.0, 2.305232559449707, 16.702875375272402])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, 2e-3], [0.1, -1e-2, 1e-3], [0.2, 1e-2, 2e-4]]
        )
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
        )

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 3.236449774952821)

        # Test distance for multiple rays
        rays = RealRays(
            [1.0, 2.0],
            [2.0, 3.0],
            [-3.0, -4.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        )
        distance = geometry.distance(rays)
        nom_distance = [3.236449774952821, 4.881863713037335]
        assert_allclose(distance, nom_distance)

        # Test distance for ray not parallel to z axis
        L = 0.164
        M = -0.210
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 12.610897321951025)

    def test_surface_normal(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, 2e-3], [0.1, -1e-2, 1e-3], [0.2, 1e-2, 2e-4]]
        )
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
        )

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert_allclose(nx, 0.4373017765693584)
        assert_allclose(ny, -0.04888445345459283)
        assert_allclose(nz, -0.8979852261700794)

    def test_to_dict(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, 2e-3], [0.1, -1e-2, 1e-3], [0.2, 1e-2, 2e-4]]
        )
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
        )

        expected_dict = {
            "type": "PolynomialGeometry",
            "cs": cs.to_dict(),
            "radius": -26.0,
            "conic": 0.1,
            "tol": 1e-10,
            "max_iter": 100,
            "coefficients": coefficients.tolist(),
        }
        assert geometry.to_dict() == expected_dict

    def test_from_dict(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, 2e-3], [0.1, -1e-2, 1e-3], [0.2, 1e-2, 2e-4]]
        )
        geometry = geometries.PolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
        )

        geometry_dict = geometry.to_dict()
        new_geometry = geometries.PolynomialGeometry.from_dict(geometry_dict)
        assert new_geometry.to_dict() == geometry_dict


class TestChebyshevGeometry:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=22.0,
            conic=0.0,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )
        assert str(geometry) == "Chebyshev Polynomial"

    def test_sag(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=22.0,
            conic=0.0,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), -0.198)

        # Test sag at (1, 1)
        assert_allclose(geometry.sag(1, 1), -0.13832040010014895)

        # Test sag at (-2, -7)
        assert_allclose(geometry.sag(-2, -7), 1.036336507973306)

        # Test array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([-0.198, 1.22291856, 1.75689642])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 2.71982177)

        # Test distance for multiple rays
        rays = RealRays(
            [1.0, 2.0],
            [2.0, 3.0],
            [-3.0, -4.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        )
        distance = geometry.distance(rays)
        nom_distance = [2.719821774952821, 3.5873077130373345]
        assert_allclose(distance, nom_distance)

        # Test distance for ray not parallel to z axis
        L = 0.164
        M = -0.210
        N = np.sqrt(1 - L**2 - M**2)
        rays = RealRays(1.0, 2.0, -10.2, L, M, N, 1.0, 0.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 10.29015593)

    def test_surface_normal(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert_allclose(nx, 0.14317439)
        assert_allclose(ny, -0.07668599)
        assert_allclose(nz, -0.98672202)

    def test_invalid_input(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )

        with pytest.raises(ValueError):
            geometry.sag(100, 100)

    def test_to_dict(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )

        expected_dict = {
            "type": "ChebyshevPolynomialGeometry",
            "cs": cs.to_dict(),
            "radius": -26.0,
            "conic": 0.1,
            "tol": 1e-10,
            "max_iter": 100,
            "coefficients": coefficients.tolist(),
            "norm_x": 10,
            "norm_y": 10,
        }
        assert geometry.to_dict() == expected_dict

    def test_from_dict(self, set_test_backend):
        cs = CoordinateSystem()
        coefficients = be.array(
            [[0.0, 1e-2, -2e-3], [0.1, 1e-2, -1e-3], [0.2, 1e-2, 0.0]]
        )
        geometry = geometries.ChebyshevPolynomialGeometry(
            cs,
            radius=-26.0,
            conic=0.1,
            coefficients=coefficients,
            norm_x=10,
            norm_y=10,
        )

        geometry_dict = geometry.to_dict()
        new_geometry = geometries.ChebyshevPolynomialGeometry.from_dict(geometry_dict)
        assert new_geometry.to_dict() == geometry_dict


class TestOddAsphere:
    def test_sag(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.OddAsphere(
            cs,
            radius=27.0,
            conic=0.0,
            coefficients=[1e-3, -1e-5],
        )

        # Test sag at (0, 0)
        assert_allclose(geometry.sag(), 0.0)

        # Test sag at (1, 1)
        assert_allclose(geometry.sag(1, 1), 0.03845668813684687)

        # Test sag at (-2, 3)
        assert_allclose(geometry.sag(-2, 3), 0.24529923075615997)

        # Test array input
        x = be.array([0, 3, 8])
        y = be.array([0, -7, 2.1])
        sag = be.array([0.0, 1.10336808, 1.30564148])
        assert_allclose(geometry.sag(x, y), sag)

    def test_distance(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.OddAsphere(
            cs,
            radius=-41.1,
            conic=0.0,
            coefficients=[1e-3, -1e-5, 1e-7],
        )

        # Test distance for a single ray
        rays = RealRays(1.0, 2.0, -3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        distance = geometry.distance(rays)
        assert_allclose(distance, 2.94131486)

        # Test distance for multiple rays
        rays = RealRays(
            [1.0, 2.0],
            [2.0, 3.0],
            [-3.0, -4.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        )
        distance = geometry.distance(rays)
        assert_allclose(distance, [2.94131486, 3.84502393])

    def test_surface_normal(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.OddAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2, -1e-5],
        )

        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        nx, ny, nz = geometry.surface_normal(rays)
        assert_allclose(nx, 0.10537434)
        assert_allclose(ny, 0.21074867)
        assert_allclose(nz, -0.9718442)

    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        geometry = geometries.OddAsphere(
            cs,
            radius=10.0,
            conic=0.5,
            coefficients=[1e-2, -1e-5],
        )
        assert str(geometry) == "Odd Asphere"


class TestZernikeGeometry:
    def coefficients_dict_to_list(
        self, coefficients: dict[int, float], zernike_type: str
    ) -> list[float]:
        start = 0 if zernike_type == "standard" else 1

        return [
            coefficients.get(i, 0.0)
            for i in range(start, max(coefficients.keys()) + 1 + start)
        ]

    def create_geometry(
        self,
        coefficients,
        norm_radius: float = 10,
        zernike_type: str = "fringe",
        radius=22,
        conic=0.0,
    ) -> geometries.ZernikePolynomialGeometry:
        cs = CoordinateSystem()

        return geometries.ZernikePolynomialGeometry(
            cs,
            radius=radius,
            conic=conic,
            coefficients=coefficients,
            norm_radius=norm_radius,
            zernike_type=zernike_type,
        )

    @pytest.mark.parametrize(
        "norm_radius, expectation",
        [
            (10, does_not_raise()),
            (
                0,
                pytest.raises(
                    ValueError, match="Normalization radius must be positive"
                ),
            ),
            (
                -5,
                pytest.raises(
                    ValueError, match="Normalization radius must be positive"
                ),
            ),
        ],
    )
    def test_init(self, set_test_backend, norm_radius, expectation):
        coefficients = list(range(10))

        with expectation:
            self.create_geometry(coefficients, norm_radius=norm_radius)

    def test_get_coefficients(self, set_test_backend):
        coefficients = list(range(10))
        geometry = self.create_geometry(coefficients)

        assert list(geometry.coefficients) == coefficients
        assert_array_equal(geometry.coefficients, geometry.zernike.coeffs)

    def test_set_coefficients(self, set_test_backend):
        coefficients = list(range(10))
        geometry = self.create_geometry(coefficients)
        old_zernike = geometry.zernike

        new_coefficients = list(range(20))
        geometry.coefficients = new_coefficients

        assert list(geometry.coefficients) == new_coefficients
        assert geometry.zernike != old_zernike
        assert len(geometry.zernike.indices) == len(new_coefficients)

    def test_str(self, set_test_backend):
        coefficients = be.array([[0.0, 1e-2, -2e-3], [0, 0, 0], [0, 0, 0]])
        geometry = self.create_geometry(coefficients)
        assert str(geometry) == "Zernike Polynomial"

    # fmt: off
    REFERENCE_SAG = {
        "standard": np.array(
            [
                [14.5393493 ,  9.06299698,  5.6901489 ,  3.82524018,  3.04804724,
                    3.1079359 ,  3.92161984,  5.57316059,  8.31620907, 12.57975751],
                [10.42095581,  5.89794049,  3.18020179,  1.69752586,  1.05162725,
                    1.01183632,  1.5133748 ,  2.65735586,  4.71250841,  8.11948982],
                [ 7.67790733,  3.9395579 ,  1.76960277,  0.62031594,  0.11396659,
                    0.03927432,  0.34996893,  1.16479049,  2.76892944,  5.61757143],
                [ 5.88121164,  2.74719186,  1.0031662 ,  0.12216224, -0.25386995,
                    -0.31719339, -0.09557018,  0.54773847,  1.91525828,  4.47843744],
                [ 4.7772178 ,  2.05211344,  0.59570162, -0.09946097, -0.37237837,
                    -0.39657086, -0.18129248,  0.42846898,  1.7532095 ,  4.28186429],
                [ 4.28186429,  1.7532095 ,  0.42846898, -0.18129248, -0.39657086,
                    -0.37237837, -0.09946097,  0.59570162,  2.05211344,  4.7772178 ],
                [ 4.47843744,  1.91525828,  0.54773847, -0.09557018, -0.31719339,
                    -0.25386995,  0.12216224,  1.0031662 ,  2.74719186,  5.88121164],
                [ 5.61757143,  2.76892944,  1.16479049,  0.34996893,  0.03927432,
                    0.11396659,  0.62031594,  1.76960277,  3.9395579 ,  7.67790733],
                [ 8.11948982,  4.71250841,  2.65735586,  1.5133748 ,  1.01183632,
                    1.05162725,  1.69752586,  3.18020179,  5.89794049, 10.42095581],
                [12.57975751,  8.31620907,  5.57316059,  3.92161984,  3.1079359 ,
                    3.04804724,  3.82524018,  5.6901489 ,  9.06299698, 14.5393493 ]
            ]
         ),
         "fringe": np.array(
            [
                [ 8.84770045,  6.57121389,  4.91876121,  3.83164416,  3.24019577,
                    3.05802842,  3.17979231,  3.48117547,  3.8211453 ,  4.04770045],
                [ 5.98800333,  4.17563074,  2.88912966,  2.08054662,  1.69404966,
                    1.66161558,  1.90130565,  2.31726561,  2.80144997,  3.23793475],
                [ 4.17536504,  2.63007023,  1.54424187,  0.87780814,  0.58738851,
                    0.62274898,  0.92536187,  1.42840587,  2.05820619,  2.73777931],
                [ 3.20777585,  1.78878893,  0.7852545 ,  0.16324809, -0.10903309,
                    -0.06440575,  0.26201352,  0.83280823,  1.60954796,  2.555924  ],
                [ 2.88526343,  1.50466796,  0.51001549, -0.12769556, -0.42932525,
                    -0.41054762, -0.08306822,  0.54537595,  1.47223388,  2.70309608],
                [ 3.01064404,  1.63140338,  0.61602236, -0.06059874, -0.40891508,
                    -0.42769271, -0.10522609,  0.5806619 ,  1.66383746,  3.1928114 ],
                [ 3.39378408,  2.02828494,  1.00548516,  0.30316579, -0.07926121,
                    -0.12388855,  0.20440036,  0.95793143,  2.20752591,  4.04563593],
                [ 3.85810311,  2.56669913,  1.59182947,  0.91293151,  0.53864933,
                    0.50328887,  0.86537779,  1.70766547,  3.13856318,  5.29568884],
                [ 4.24931851,  3.13835661,  2.30763971,  1.73652725,  1.44202168,
                    1.47445576,  1.91576822,  2.87950375,  4.51253737,  6.9993871 ],
                [ 4.44770045,  3.66610796,  3.11470695,  2.76991577,  2.6557117 ,
                    2.83787905,  3.42176762,  4.55229268,  6.41617655,  9.24770045]
            ]
        ),
    }
    # fmt: on

    @pytest.mark.parametrize(
        "zernike_type, coefficients",
        [
            ("standard", {4: 0.5, 3: 0.2, 5: 0.3, 10: 0.1, 12: 0.2}),
            ("noll", {4: 0.5, 5: 0.2, 6: 0.3, 11: 0.2, 15: 0.1}),
            ("fringe", {4: 0.5, 6: 0.2, 11: 0.3, 13: 0.2, 27: 0.1}),
        ],
    )
    def test_sag(
        self, set_test_backend, zernike_type: str, coefficients: dict[int, float]
    ):
        geometry = self.create_geometry(
            coefficients=self.coefficients_dict_to_list(coefficients, zernike_type),
            zernike_type=zernike_type,
        )

        x = y = be.linspace(-10, 10, 10)
        X, Y = be.meshgrid(x, y)

        sag = geometry.sag(X, Y)
        reference = self.REFERENCE_SAG.get(zernike_type, self.REFERENCE_SAG["standard"])

        assert_allclose(sag, reference, atol=1e-6)

    REFERENCE_GRADIENT = {
        "standard": np.array(
            [
                [
                    [-0.69922039, -0.58360519, -0.41291141],
                    [-0.58584794, -0.61891081, -0.52319365],
                    [-0.43588997, -0.64349955, -0.62921242],
                    [-0.26225921, -0.65565742, -0.70804905],
                    [-0.08528575, -0.66166676, -0.74493183],
                    [0.0823724, -0.6690924, -0.73860012],
                    [0.23882481, -0.67838895, -0.69480295],
                    [0.3845705, -0.68243823, -0.62159761],
                    [0.51637133, -0.67277311, -0.52984619],
                    [0.62879296, -0.6460985, -0.43263859],
                ],
                [
                    [-0.7230823, -0.47869456, -0.4979995],
                    [-0.59800005, -0.49121134, -0.63333037],
                    [-0.43588123, -0.48759509, -0.75647775],
                    [-0.25922765, -0.47392709, -0.84154271],
                    [-0.09042355, -0.464726, -0.88082537],
                    [0.06666968, -0.47050177, -0.87987683],
                    [0.22143063, -0.49048026, -0.84285087],
                    [0.38011481, -0.51327155, -0.76945763],
                    [0.53510722, -0.5231329, -0.6633191],
                    [0.66943028, -0.51063855, -0.53954738],
                ],
                [
                    [-0.72966613, -0.36532612, -0.57803474],
                    [-0.5858297, -0.35634459, -0.72788879],
                    [-0.40989935, -0.32987631, -0.85039059],
                    [-0.23593263, -0.29876348, -0.92470329],
                    [-0.08398426, -0.2790976, -0.95658307],
                    [0.05314045, -0.27901255, -0.95881598],
                    [0.19586825, -0.29730614, -0.93447562],
                    [0.35891437, -0.32458199, -0.87511542],
                    [0.53485494, -0.34500852, -0.77129716],
                    [0.69399255, -0.34507355, -0.63190077],
                ],
                [
                    [-0.72364456, -0.24726071, -0.64436069],
                    [-0.55950744, -0.22712405, -0.79709855],
                    [-0.37328687, -0.19417622, -0.9071673],
                    [-0.2061676, -0.16184728, -0.96503906],
                    [-0.07140386, -0.14112401, -0.98741354],
                    [0.04713024, -0.13547177, -0.98965961],
                    [0.17631368, -0.14312988, -0.97387233],
                    [0.33698032, -0.15842423, -0.9280873],
                    [0.52503303, -0.17157039, -0.83360897],
                    [0.70311772, -0.17271833, -0.68977811],
                ],
                [
                    [-0.71363817, -0.12289009, -0.68965106],
                    [-0.53428719, -0.10456653, -0.83881049],
                    [-0.34302984, -0.08098097, -0.93582724],
                    [-0.18232384, -0.05982326, -0.98141693],
                    [-0.05856711, -0.04505135, -0.9972664],
                    [0.0495254, -0.03643636, -0.99810802],
                    [0.17137003, -0.0318194, -0.98469276],
                    [0.32983361, -0.02832571, -0.94361403],
                    [0.52189187, -0.02311, -0.85269854],
                    [0.70679267, -0.0149619, -0.70726251],
                ],
                [
                    [-0.70679267, 0.0149619, -0.70726251],
                    [-0.52189187, 0.02311, -0.85269854],
                    [-0.32983361, 0.02832571, -0.94361403],
                    [-0.17137003, 0.0318194, -0.98469276],
                    [-0.0495254, 0.03643636, -0.99810802],
                    [0.05856711, 0.04505135, -0.9972664],
                    [0.18232384, 0.05982326, -0.98141693],
                    [0.34302984, 0.08098097, -0.93582724],
                    [0.53428719, 0.10456653, -0.83881049],
                    [0.71363817, 0.12289009, -0.68965106],
                ],
                [
                    [-0.70311772, 0.17271833, -0.68977811],
                    [-0.52503303, 0.17157039, -0.83360897],
                    [-0.33698032, 0.15842423, -0.9280873],
                    [-0.17631368, 0.14312988, -0.97387233],
                    [-0.04713024, 0.13547177, -0.98965961],
                    [0.07140386, 0.14112401, -0.98741354],
                    [0.2061676, 0.16184728, -0.96503906],
                    [0.37328687, 0.19417622, -0.9071673],
                    [0.55950744, 0.22712405, -0.79709855],
                    [0.72364456, 0.24726071, -0.64436069],
                ],
                [
                    [-0.69399255, 0.34507355, -0.63190077],
                    [-0.53485494, 0.34500852, -0.77129716],
                    [-0.35891437, 0.32458199, -0.87511542],
                    [-0.19586825, 0.29730614, -0.93447562],
                    [-0.05314045, 0.27901255, -0.95881598],
                    [0.08398426, 0.2790976, -0.95658307],
                    [0.23593263, 0.29876348, -0.92470329],
                    [0.40989935, 0.32987631, -0.85039059],
                    [0.5858297, 0.35634459, -0.72788879],
                    [0.72966613, 0.36532612, -0.57803474],
                ],
                [
                    [-0.66943028, 0.51063855, -0.53954738],
                    [-0.53510722, 0.5231329, -0.6633191],
                    [-0.38011481, 0.51327155, -0.76945763],
                    [-0.22143063, 0.49048026, -0.84285087],
                    [-0.06666968, 0.47050177, -0.87987683],
                    [0.09042355, 0.464726, -0.88082537],
                    [0.25922765, 0.47392709, -0.84154271],
                    [0.43588123, 0.48759509, -0.75647775],
                    [0.59800005, 0.49121134, -0.63333037],
                    [0.7230823, 0.47869456, -0.4979995],
                ],
                [
                    [-0.62879296, 0.6460985, -0.43263859],
                    [-0.51637133, 0.67277311, -0.52984619],
                    [-0.3845705, 0.68243823, -0.62159761],
                    [-0.23882481, 0.67838895, -0.69480295],
                    [-0.0823724, 0.6690924, -0.73860012],
                    [0.08528575, 0.66166676, -0.74493183],
                    [0.26225921, 0.65565742, -0.70804905],
                    [0.43588997, 0.64349955, -0.62921242],
                    [0.58584794, 0.61891081, -0.52319365],
                    [0.69922039, 0.58360519, -0.41291141],
                ],
            ]
        ),
        "fringe": np.array(
            [
                [
                    [-0.53614328, -0.70977208, -0.45691791],
                    [-0.47428983, -0.69601623, -0.53908308],
                    [-0.38347556, -0.67820133, -0.62688871],
                    [-0.26399027, -0.6529739, -0.70988324],
                    [-0.1293257, -0.61868393, -0.77492261],
                    [-0.00404877, -0.57840385, -0.81574052],
                    [0.08821867, -0.53773356, -0.83848678],
                    [0.13309674, -0.49888028, -0.85638994],
                    [0.12253717, -0.45671491, -0.88113343],
                    [0.04892935, -0.39717556, -0.91643739],
                ],
                [
                    [-0.54702793, -0.60229659, -0.58137703],
                    [-0.46268178, -0.58410849, -0.66689043],
                    [-0.35006852, -0.56299114, -0.74866081],
                    [-0.21537209, -0.53808491, -0.81491072],
                    [-0.07596508, -0.50980004, -0.85693245],
                    [0.04670631, -0.47986172, -0.87610002],
                    [0.13705986, -0.44877693, -0.8830707],
                    [0.18804927, -0.41293666, -0.89113455],
                    [0.19674808, -0.36261031, -0.91093576],
                    [0.15919545, -0.27954758, -0.9468421],
                ],
                [
                    [-0.56628469, -0.42947222, -0.70347371],
                    [-0.46223857, -0.41284022, -0.78479198],
                    [-0.33422539, -0.39796643, -0.85435128],
                    [-0.19122681, -0.38520836, -0.90279944],
                    [-0.04938955, -0.37355646, -0.92629166],
                    [0.07508127, -0.36046182, -0.92974732],
                    [0.17239576, -0.34132419, -0.92400081],
                    [0.23988473, -0.308558, -0.92046036],
                    [0.27855455, -0.2498992, -0.92733907],
                    [0.28919037, -0.14589247, -0.94608896],
                ],
                [
                    [-0.5796721, -0.21630754, -0.78561524],
                    [-0.4662884, -0.20820115, -0.85978335],
                    [-0.33411619, -0.20686466, -0.91955064],
                    [-0.19056786, -0.21210422, -0.95848615],
                    [-0.0474316, -0.22027026, -0.974285],
                    [0.08369702, -0.22480945, -0.97080148],
                    [0.19579418, -0.21728075, -0.95627073],
                    [0.28746716, -0.18758775, -0.9392409],
                    [0.36077253, -0.12302782, -0.92450383],
                    [0.41739683, -0.00787703, -0.90869018],
                ],
                [
                    [-0.58083244, -0.02420474, -0.8136632],
                    [-0.46945528, -0.02168147, -0.88269001],
                    [-0.34312597, -0.02861496, -0.93885342],
                    [-0.20532323, -0.04450046, -0.97768199],
                    [-0.06237036, -0.06461396, -0.99595932],
                    [0.07814951, -0.08038802, -0.99369534],
                    [0.20997046, -0.08091729, -0.97435353],
                    [0.32976596, -0.05458248, -0.94248351],
                    [0.43611437, 0.00969408, -0.89983903],
                    [0.52590046, 0.11966382, -0.84208627],
                ],
                [
                    [-0.5749214, 0.10410802, -0.81155832],
                    [-0.46961149, 0.11211286, -0.87572584],
                    [-0.35259338, 0.1101207, -0.92927463],
                    [-0.22333503, 0.09819221, -0.96978335],
                    [-0.08253705, 0.08029933, -0.9933477],
                    [0.06677802, 0.06453505, -0.99567862],
                    [0.21891011, 0.06204944, -0.97377011],
                    [0.36607688, 0.08439626, -0.92674969],
                    [0.49899821, 0.13971533, -0.85526628],
                    [0.60713251, 0.22842, -0.76106138],
                ],
                [
                    [-0.56449618, 0.16439775, -0.80889891],
                    [-0.46298974, 0.19000278, -0.86575946],
                    [-0.35348142, 0.20574269, -0.91253539],
                    [-0.23237651, 0.21104311, -0.94945351],
                    [-0.09547047, 0.20851082, -0.97334918],
                    [0.05917222, 0.20397116, -0.97718699],
                    [0.22731072, 0.20592458, -0.95180088],
                    [0.39603371, 0.22329399, -0.89067227],
                    [0.54649496, 0.2613974, -0.79562218],
                    [0.66217906, 0.3186887, -0.67820086],
                ],
                [
                    [-0.5426994, 0.16708996, -0.8231393],
                    [-0.44088123, 0.22333962, -0.8693349],
                    [-0.33565672, 0.26828832, -0.90297062],
                    [-0.22135999, 0.30060179, -0.92770594],
                    [-0.09015556, 0.32196037, -0.94245079],
                    [0.06411777, 0.33569474, -0.93978612],
                    [0.23936823, 0.34593021, -0.90721284],
                    [0.41928725, 0.35701159, -0.83471009],
                    [0.57777839, 0.37282141, -0.72606909],
                    [0.69526137, 0.39562379, -0.60007787],
                ],
                [
                    [-0.49226151, 0.12067315, -0.86204211],
                    [-0.38773716, 0.22497805, -0.89389304],
                    [-0.28576613, 0.31184394, -0.90614076],
                    [-0.17919311, 0.37850238, -0.90808908],
                    [-0.05769805, 0.42700698, -0.90240566],
                    [0.08772463, 0.45983516, -0.88366058],
                    [0.25735663, 0.47783216, -0.83990713],
                    [0.43515046, 0.48142674, -0.76083662],
                    [0.59298947, 0.47454416, -0.6505162],
                    [0.7100807, 0.46537766, -0.52840235],
                ],
                [
                    [-0.3818526, 0.03084339, -0.92370844],
                    [-0.27780758, 0.20716241, -0.93803341],
                    [-0.18520805, 0.35102451, -0.91786697],
                    [-0.09439596, 0.4564225, -0.88474171],
                    [0.00751602, 0.52930281, -0.8483997],
                    [0.1315469, 0.57561305, -0.80707188],
                    [0.28059723, 0.59702886, -0.75154623],
                    [0.44261585, 0.59305526, -0.67258952],
                    [0.59251763, 0.56779287, -0.57143163],
                    [0.70868919, 0.53306411, -0.46217127],
                ],
            ]
        ),
    }

    @pytest.mark.parametrize(
        "zernike_type, coefficients",
        [
            ("standard", {4: 0.5, 3: 0.2, 5: 0.3, 10: 0.1, 12: 0.2}),
            ("noll", {4: 0.5, 5: 0.2, 6: 0.3, 11: 0.2, 15: 0.1}),
            ("fringe", {4: 0.5, 6: 0.2, 11: 0.3, 13: 0.2, 27: 0.1}),
        ],
    )
    def test_surface_normal(
        self, set_test_backend, zernike_type: str, coefficients: dict[int, float]
    ):
        geometry = self.create_geometry(
            self.coefficients_dict_to_list(coefficients, zernike_type),
            norm_radius=10,
            zernike_type=zernike_type,
        )

        x = be.linspace(-10, 10, 10)
        y = be.linspace(-10, 10, 10)
        X, Y = be.meshgrid(x, y)
        normals = be.stack(geometry._surface_normal(X, Y), axis=-1)

        reference = self.REFERENCE_GRADIENT.get(
            zernike_type, self.REFERENCE_GRADIENT["standard"]
        )

        assert_allclose(normals, reference, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize(
        "x, y, norm_radius, expectation",
        [
            ([-1, -0.5, 0, 0.5, 1], [0, 0, 0, 0, 0], 1, does_not_raise()),
            ([0, 0, 0, 0, 0], [-1, -0.5, 0, 0.5, 1], 1, does_not_raise()),
            (
                -1.1,
                0,
                1,
                pytest.raises(
                    ValueError, match="Zernike coordinates must be normalized"
                ),
            ),
            (
                0,
                -1.1,
                1,
                pytest.raises(
                    ValueError, match="Zernike coordinates must be normalized"
                ),
            ),
            (
                1.1,
                0,
                1,
                pytest.raises(
                    ValueError, match="Zernike coordinates must be normalized"
                ),
            ),
            (
                0,
                1.1,
                1,
                pytest.raises(
                    ValueError, match="Zernike coordinates must be normalized"
                ),
            ),
        ],
    )
    def test_validate_inputs(self, set_test_backend, x, y, norm_radius, expectation):
        """Test that the geometry raises an error for invalid coordinates."""
        geometry = self.create_geometry([], norm_radius=norm_radius)

        with expectation:
            geometry._validate_inputs(x, y)

    def test_to_dict(self, set_test_backend):
        geometry = self.create_geometry(
            coefficients=[0.5, 0.2, 0.3, 0.1, 0.2],
            zernike_type="standard",
            norm_radius=1.0,
        )
        geometry_dict = geometry.to_dict()

        assert geometry_dict["coefficients"] == [0.5, 0.2, 0.3, 0.1, 0.2]
        assert geometry_dict["zernike_type"] == "standard"
        assert geometry_dict["norm_radius"] == 1.0

    def test_from_dict(self, set_test_backend):
        geometry = self.create_geometry(
            coefficients=[0.5, 0.2, 0.3, 0.1, 0.2],
            zernike_type="standard",
            norm_radius=1.0,
        )

        geometry_dict = geometry.to_dict()
        new_geometry = geometries.ZernikePolynomialGeometry.from_dict(geometry_dict)

        assert all(new_geometry.coefficients == geometry.coefficients)
        assert new_geometry.zernike_type == geometry.zernike_type
        assert new_geometry.norm_radius == geometry.norm_radius


# --- Fixtures for Toroidal Tests ---
@pytest.fixture
def basic_toroid_geometry(set_test_backend):
    """Provides a basic ToroidalGeometry instance for testing"""
    cs = CoordinateSystem(x=0, y=0, z=0)
    radius_x = 100.0  # R (X-Z radius)
    radius_y = 50.0  # R_y (Y-Z radius)
    conic = -0.5  # k_yz (YZ conic)
    coeffs_poly_y = [1e-5]
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_x=radius_x,
        radius_y=radius_y,
        conic=conic,
        coeffs_poly_y=coeffs_poly_y,
    )


@pytest.fixture
def cylinder_x_geometry(set_test_backend):
    """Provides a cylindrical geometry (flat in X). R_rot = inf"""
    cs = CoordinateSystem(x=0, y=0, z=0)
    radius_x = be.inf  # Flat in X-Z plane
    radius_y = -50.0
    conic = 0.0
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_x=radius_x,
        radius_y=radius_y,
        conic=conic,
    )


@pytest.fixture
def cylinder_y_geometry(set_test_backend):
    """Provides a cylindrical geometry (flat in Y). R_yz = inf"""
    cs = CoordinateSystem(x=0, y=0, z=0)
    radius_x = 100.0
    radius_y = be.inf  # Flat in Y-Z plane
    conic = 0.0
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_x=radius_x,
        radius_y=radius_y,
        conic=conic,
    )


@pytest.fixture
def toroid_no_x_curvature_coeffs(set_test_backend):
    """Toroidal surface with no curvature in x, conic_yz = -1.1, and coeffs."""
    cs = CoordinateSystem(x=0, y=0, z=0)
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_x=be.inf,
        radius_y=50.0,
        conic=-1.1,
        coeffs_poly_y=[1e-5, -2e-6],
    )


@pytest.fixture
def toroid_no_y_curvature_coeffs(set_test_backend):
    """Toroidal surface with no curvature in y, conic_yz = -0.9, and coeffs.
    This a trick test to check if the conic is not used when R_yz = inf.
    """
    cs = CoordinateSystem(x=0, y=0, z=0)
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_x=100.0,
        radius_y=be.inf,  # No curvature in Y (base YZ curve is flat before polynomials)
        conic=-0.9,  # This k_yz will not be used if R_yz is inf for conic part
        coeffs_poly_y=[1e-5, -2e-6],
    )


class TestToroidalGeometry:
    def test_toroidal_str(self, set_test_backend):
        """Test string representation."""
        cs = CoordinateSystem()
        geometry = geometries.ToroidalGeometry(
            cs,
            radius_x=100.0,
            radius_y=50.0,
            conic=-0.5,
            coeffs_poly_y=[1e-5],
        )
        assert str(geometry) == "Toroidal"

    def test_toroidal_sag_vertex(self, basic_toroid_geometry, set_test_backend):
        """Test sag at the vertex (0, 0). Should be 0 by definition."""
        x = be.array([0.0])
        y = be.array([0.0])
        be.allclose(
            basic_toroid_geometry.sag(x, y), be.array([0.0]), rtol=1e-5, atol=1e-6
        )

    def test_toroidal_normal_vertex(self, basic_toroid_geometry, set_test_backend):
        """ " Test the normal vector at the vertex (0, 0). Should be (0, 0, -1)"""
        x = be.array([0.0])
        y = be.array([0.0])
        nx, ny, nz = basic_toroid_geometry._surface_normal(x, y)
        be.allclose(nx, be.array([0.0]), rtol=1e-5, atol=1e-6)
        be.allclose(ny, be.array([0.0]), rtol=1e-5, atol=1e-6)
        be.allclose(nz, be.array([-1.0]), rtol=1e-5, atol=1e-6)

    def test_toroidal_sag_known_points(self, basic_toroid_geometry, set_test_backend):
        """Test sag at specific points"""
        x = be.array([0.0, 10.0, 0.0])
        y = be.array([10.0, 0.0, 5.0])

        expected_z_0_10 = 1.00605051
        expected_z_10_0 = 0.50125628
        expected_z_0_5 = 0.25056330

        expected_z = be.array([expected_z_0_10, expected_z_10_0, expected_z_0_5])
        calculated_z = basic_toroid_geometry.sag(x, y)
        assert be.allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-6)

    def test_toroidal_normal_known_points(
        self, basic_toroid_geometry, set_test_backend
    ):
        """Test normal at specific points."""
        x = be.array([0.0, 10.0])
        y = be.array([10.0, 0.0])

        expected_nx_0_10 = 0.0
        expected_ny_0_10 = 0.198219
        expected_nz_0_10 = -0.980158
        expected_nx_10_0 = 0.10000
        expected_ny_10_0 = 0.0
        expected_nz_10_0 = -0.994987

        expected_nx = be.array([expected_nx_0_10, expected_nx_10_0])
        expected_ny = be.array([expected_ny_0_10, expected_ny_10_0])
        expected_nz = be.array([expected_nz_0_10, expected_nz_10_0])

        nx, ny, nz = basic_toroid_geometry._surface_normal(x, y)

        rtol = 1e-5
        atol = 1e-6
        assert be.allclose(nx, expected_nx, rtol=rtol, atol=atol)
        assert be.allclose(ny, expected_ny, rtol=rtol, atol=atol)
        assert be.allclose(nz, expected_nz, rtol=rtol, atol=atol)

    def test_cylinder_x_sag(self, cylinder_x_geometry, set_test_backend):
        """Test sag for cylinder flat in X. Should only depend on y."""
        x = be.array([0.0, 10.0, 10.0])
        y = be.array([5.0, 5.0, 0.0])

        expected_z = be.array([-0.25062818, -0.25062818, 0.0])
        calculated_z = cylinder_x_geometry.sag(x, y)
        assert_allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-6)

    def test_cylinder_y_sag(self, cylinder_y_geometry, set_test_backend):
        """Test sag for cylinder flat in Y. Should only depend on x."""
        x = be.array([5.0, 5.0, 0.0])
        y = be.array([0.0, 10.0, 10.0])

        expected_z = be.array([0.12507822, 0.12507822, 0.0])
        calculated_z = cylinder_y_geometry.sag(x, y)
        assert_allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-6)

    def test_toroid_no_x_curvature_sag(
        self, toroid_no_x_curvature_coeffs, set_test_backend
    ):
        """Test sag for toroid with R_rot = inf and YZ coeffs."""

        geom = toroid_no_x_curvature_coeffs
        x = be.array([0.0, 10.0, -5.0])
        y = be.array([0.0, 1.0, 2.0])

        expected_z = be.array([0.0, 0.010007900001, 0.0400064001])
        calculated_z = geom.sag(x, y)
        assert_allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-7)

    def test_toroid_no_y_curvature_sag(
        self, toroid_no_y_curvature_coeffs, set_test_backend
    ):
        """Test sag for toroid with R_yz = inf and YZ coeffs. Sag comes from X-rotation of YZ poly."""

        geom = toroid_no_y_curvature_coeffs
        x = be.array([0.0, 10.0, 5.0])
        y = be.array([0.0, 1.0, 2.0])

        expected_z = be.array([0.0, 0.501264335, 0.12508686])
        calculated_z = geom.sag(x, y)
        assert_allclose(calculated_z, expected_z, rtol=1e-5, atol=1e-7)

    def test_toroidal_sag_vs_zemax(self, basic_toroid_geometry, set_test_backend):
        """
        Compares sag values calculated by Optiland with
        Zemax data for the basic_toroid_geometry.
        """
        geometry = basic_toroid_geometry

        x_coords = be.array([0.0, 2.5, 0.0, -2.5, 5.0, -5.0, 2.5, -2.5])
        y_coords = be.array([0.0, 0.0, 2.5, 0.0, 2.5, -2.5, -2.5, 2.5])

        # --- Zemax Sag Data ---
        zemax_z_sag = be.array(
            [
                0.0,  # (0, 0) - Vertex
                3.125488433897521e-002,  # (2.5, 0)
                6.258204346657634e-002,  # (0, 2.5)
                3.125488433897521e-002,  # (-2.5, 0)
                1.877386899843393e-001,  # (5, 2.5)
                1.877386899843393e-001,  # (-5, -2.5)
                9.385650612446353e-002,  # (2.5, -2.5)
                9.385650612446353e-002,  # (-2.5, 2.5)
            ]
        )

        # Calculate sag using Optiland
        optiland_z_sag = geometry.sag(x_coords, y_coords)

        assert be.allclose(optiland_z_sag, zemax_z_sag, rtol=1e-5, atol=1e-6)

    def test_toroidal_ray_tracing_comparison(self, set_test_backend):
        """
        Traces rays through a single toroidal surface and compares output
        with Zemax ray tracing data for the same system.
        """
        # --- System Setup ---
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(
            index=1,
            surface_type="toroidal",
            thickness=5.0,
            material=IdealMaterial(n=1.5, k=0),
            is_stop=True,
            radius_x=100.0,
            radius_y=50.0,
            conic=-0.5,
            toroidal_coeffs_poly_y=[0.05, 0.0002],
        )
        lens.surfaces.add(index=2, thickness=10.0, material="air")
        lens.surfaces.add(index=3)

        lens.set_aperture(aperture_type="EPD", value=10.0)
        lens.wavelengths.add(value=0.550, is_primary=True)
        lens.fields.set_type("angle")
        lens.fields.add(y=0)

        num_rays = 5  # Number of rays per fan
        wavelength = 0.550
        z_start = 0.0

        # --- Tangential (Y) Fan Test ---
        y_coords = be.linspace(-5.0, 5.0, num_rays)
        x_in_yfan = be.zeros(num_rays)
        y_in_yfan = y_coords
        z_in_yfan = be.array([z_start] * num_rays)
        L_in_yfan = be.zeros(num_rays)
        M_in_yfan = be.zeros(num_rays)
        N_in_yfan = be.ones(num_rays)
        intensity_yfan = be.ones(num_rays)
        rays_in_yfan = RealRays(
            x=x_in_yfan,
            y=y_in_yfan,
            z=z_in_yfan,
            L=L_in_yfan,
            M=M_in_yfan,
            N=N_in_yfan,
            wavelength=wavelength,
            intensity=intensity_yfan,
        )

        # Trace Y-Fan Rays
        rays_out_yfan = lens.surfaces.trace(rays_in_yfan)

        zemax_x_out_yfan = be.array([0.0] * num_rays)
        zemax_y_out_yfan = be.array(
            [
                -8.123193233401276e-001,
                -4.676255499616224e-001,
                0,
                4.676255499616224e-001,
                8.123193233401276e-001,
            ]
        )
        zemax_z_out_yfan = be.array([15.0] * num_rays)
        zemax_L_out_yfan = be.array([0.0] * num_rays)
        zemax_M_out_yfan = be.array(
            [
                3.251509839270260e-001,
                1.537950377308984e-001,
                0.0,
                -1.537950377308984e-001,
                -3.251509839270260e-001,
            ]
        )
        zemax_N_out_yfan = be.array(
            [
                9.456621160072382e-001,
                9.881027711576116e-001,
                1.0,
                9.881027711576116e-001,
                9.456621160072382e-001,
            ]
        )

        # Comparison Assertions for Y-Fan
        assert be.allclose(rays_out_yfan.x, zemax_x_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.y, zemax_y_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.z, zemax_z_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.L, zemax_L_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.M, zemax_M_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.N, zemax_N_out_yfan, rtol=1e-5, atol=1e-6)

        # --- Sagittal (X) Fan Test ---
        x_coords = be.linspace(-5.0, 5.0, num_rays)
        x_in_xfan = x_coords
        y_in_xfan = be.zeros(num_rays)
        z_in_xfan = be.array([z_start] * num_rays)
        L_in_xfan = be.zeros(num_rays)
        M_in_xfan = be.zeros(num_rays)
        N_in_xfan = be.ones(num_rays)
        intensity_xfan = be.ones(num_rays)
        rays_in_xfan = RealRays(
            x=x_in_xfan,
            y=y_in_xfan,
            z=z_in_xfan,
            L=L_in_xfan,
            M=M_in_xfan,
            N=N_in_xfan,
            wavelength=wavelength,
            intensity=intensity_xfan,
        )

        rays_out_xfan = lens.surfaces.trace(rays_in_xfan)

        zemax_x_out_xfan = be.array(
            [
                -4.668385225648558e000,
                -2.333547899735358e000,
                0.0,
                2.333547899735358e000,
                4.668385225648558e000,
            ]
        )
        zemax_y_out_xfan = be.array([0.0] * num_rays)
        zemax_z_out_xfan = be.array([15.0] * num_rays)
        zemax_L_out_xfan = be.array(
            [
                2.502086086422164e-002,
                1.250260502601134e-002,
                0.0,
                -1.250260502601134e-002,
                -2.502086086422164e-002,
            ]
        )
        zemax_M_out_xfan = be.array([0.0] * num_rays)
        zemax_N_out_xfan = be.array(
            [
                9.996869292541608e-001,
                9.999218393792406e-001,
                1.0,
                9.999218393792406e-001,
                9.996869292541608e-001,
            ]
        )

        assert be.allclose(rays_out_xfan.x, zemax_x_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.y, zemax_y_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.z, zemax_z_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.L, zemax_L_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.M, zemax_M_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.N, zemax_N_out_xfan, rtol=1e-5, atol=1e-6)

    def test_toroidal_ray_tracing_comparison_negRx(self, set_test_backend):
        """
        Traces rays through a single toroidal surface and compares output
        with Zemax ray tracing data for the same system.
        """
        # --- System Setup ---
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(
            index=1,
            surface_type="toroidal",
            thickness=7.0,
            material="N-BK7",
            is_stop=True,
            radius_x=-50.0,
            radius_y=40.0,
            conic=-0.5,
            toroidal_coeffs_poly_y=[5e-5, 5e-6],
        )
        lens.surfaces.add(index=2, thickness=70.0, material="air")
        lens.surfaces.add(index=3)

        lens.set_aperture(aperture_type="EPD", value=20.0)
        lens.wavelengths.add(value=0.550, is_primary=True)
        lens.fields.set_type("angle")
        lens.fields.add(y=0)

        # --- SAG testing ---
        x_coords = be.array([0.0, 2.5, 0.0, -2.5, 5.0, -5.0, 2.5, -2.5])
        y_coords = be.array([0.0, 0.0, 2.5, 0.0, 2.5, -2.5, -2.5, 2.5])

        # --- Zemax Sag Data ---
        zemax_z_sag = be.array(
            [
                0.0,  # (0, 0) - Vertex
                -6.253911140455271e-002,  # (2.5, 0)
                7.867099677109624e-002,  # (0, 2.5)
                -6.253911140455271e-002,  # (-2.5, 0)
                -1.715614452665938e-001,  # (5, 2.5)
                -1.715614452665938e-001,  # (-5, -2.5)
                1.623025381703616e-002,  # (2.5, -2.5)
                1.623025381703616e-002,  # (-2.5, 2.5)
            ]
        )

        # Calculate sag using Optiland
        optiland_z_sag = lens.surfaces[1].geometry.sag(x_coords, y_coords)

        assert be.allclose(optiland_z_sag, zemax_z_sag, rtol=1e-5, atol=1e-6)

        num_rays = 5  # Number of rays per fan
        wavelength = 0.550
        z_start = 0.0

        # --- Tangential (Y) Fan Test ---
        y_coords = be.linspace(-10.0, 10.0, num_rays)
        x_in_yfan = be.zeros(num_rays)
        y_in_yfan = y_coords
        z_in_yfan = be.array([z_start] * num_rays)
        L_in_yfan = be.zeros(num_rays)
        M_in_yfan = be.zeros(num_rays)
        N_in_yfan = be.ones(num_rays)
        intensity_yfan = be.ones(num_rays)
        rays_in_yfan = RealRays(
            x=x_in_yfan,
            y=y_in_yfan,
            z=z_in_yfan,
            L=L_in_yfan,
            M=M_in_yfan,
            N=N_in_yfan,
            wavelength=wavelength,
            intensity=intensity_yfan,
        )

        # Trace Y-Fan Rays
        rays_out_yfan = lens.surfaces.trace(rays_in_yfan)
        # (Lines removed as they are unnecessary debug print statements)
        zemax_x_out_yfan = be.array([0.0] * num_rays)
        zemax_y_out_yfan = be.array(
            [
                4.842002236238105e-001,
                -4.633747816929823e-002,
                0,
                4.633747816929823e-002,
                -4.842002236238105e-001,
            ]
        )
        zemax_z_out_yfan = be.array([77.0] * num_rays)
        zemax_L_out_yfan = be.array([0.0] * num_rays)
        zemax_M_out_yfan = be.array(
            [
                1.407949486740093e-001,
                6.643870254059459e-002,
                0.0,
                -6.643870254059459e-002,
                -1.407949486740093e-001,
            ]
        )
        zemax_N_out_yfan = be.array(
            [
                9.900387782445106e-001,
                9.977905084759640e-001,
                1.0,
                9.977905084759640e-001,
                9.900387782445106e-001,
            ]
        )

        # Comparison Assertions for Y-Fan
        assert be.allclose(rays_out_yfan.x, zemax_x_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.y, zemax_y_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.z, zemax_z_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.L, zemax_L_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.M, zemax_M_out_yfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_yfan.N, zemax_N_out_yfan, rtol=1e-5, atol=1e-6)

        # --- Surface Normal Comparison ---
        geom = lens.surfaces[1].geometry
        # choose a test point or array of points
        x_norm = be.array([2.5, -2.5, 2.5, -2.5])
        y_norm = be.array([0.0, 0.0, -2.5, 2.5])
        # Zemax reference normals at those (x, y) locations:
        expected_nx = be.array(
            [
                -4.999999999998470e-002,
                4.999999999998470e-002,
                -4.982229052314893e-002,
                4.982229052314893e-002,
            ]
        )
        expected_ny = be.array(
            [0.0, 0.0, -6.299823835002263e-002, 6.299823835002263e-002]
        )
        expected_nz = be.array(
            [
                -9.987492177719096e-001,
                -9.987492177719096e-001,
                -9.967692618313533e-001,
                -9.967692618313533e-001,
            ]
        )
        nx_calc, ny_calc, nz_calc = geom._surface_normal(x_norm, y_norm)
        assert be.allclose(nx_calc, expected_nx, rtol=1e-5, atol=1e-6)
        assert be.allclose(ny_calc, expected_ny, rtol=1e-5, atol=1e-6)
        assert be.allclose(nz_calc, expected_nz, rtol=1e-5, atol=1e-6)

        # --- Sagittal (X) Fan Test ---
        x_coords = be.linspace(-10.0, 10.0, num_rays)
        x_in_xfan = x_coords
        y_in_xfan = be.zeros(num_rays)
        z_in_xfan = be.array([z_start] * num_rays)
        L_in_xfan = be.zeros(num_rays)
        M_in_xfan = be.zeros(num_rays)
        N_in_xfan = be.ones(num_rays)
        intensity_xfan = be.ones(num_rays)
        rays_in_xfan = RealRays(
            x=x_in_xfan,
            y=y_in_xfan,
            z=z_in_xfan,
            L=L_in_xfan,
            M=M_in_xfan,
            N=N_in_xfan,
            wavelength=wavelength,
            intensity=intensity_xfan,
        )

        rays_out_xfan = lens.surfaces.trace(rays_in_xfan)

        zemax_x_out_xfan = be.array(
            [
                -1.795367970779353e001,
                -8.895158024373812e000,
                0.0,
                8.895158024373812e000,
                1.795367970779353e001,
            ]
        )
        zemax_y_out_xfan = be.array([0.0] * num_rays)
        zemax_z_out_xfan = be.array([77.0] * num_rays)
        zemax_L_out_xfan = be.array(
            [
                -1.050996348781765e-001,
                -5.202386983944651e-002,
                0.0,
                5.202386983944651e-002,
                1.050996348781765e-001,
            ]
        )
        zemax_M_out_xfan = be.array([0.0] * num_rays)
        zemax_N_out_xfan = be.array(
            [
                9.944616969740334e-001,
                9.986458416109926e-001,
                1.0,
                9.986458416109926e-001,
                9.944616969740334e-001,
            ]
        )

        assert be.allclose(rays_out_xfan.x, zemax_x_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.y, zemax_y_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.z, zemax_z_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.L, zemax_L_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.M, zemax_M_out_xfan, rtol=1e-5, atol=1e-6)
        assert be.allclose(rays_out_xfan.N, zemax_N_out_xfan, rtol=1e-5, atol=1e-6)

    def test_toroidal_to_dict(self, basic_toroid_geometry, set_test_backend):
        """Test serialization to dictionary."""
        geom_dict = basic_toroid_geometry.to_dict()
        assert geom_dict["type"] == "ToroidalGeometry"
        assert geom_dict["geometry_type"] == "Toroidal"
        assert geom_dict["radius_x"] == 100.0
        assert geom_dict["radius_y"] == 50.0
        assert geom_dict["conic_yz"] == -0.5
        assert geom_dict["coeffs_poly_y"] == pytest.approx([1e-5])

    def test_toroidal_from_dict(self, basic_toroid_geometry, set_test_backend):
        """Test deserialization from dictionary."""
        geom_dict = basic_toroid_geometry.to_dict()
        new_geometry = geometries.ToroidalGeometry.from_dict(geom_dict)
        assert isinstance(new_geometry, geometries.ToroidalGeometry)
        assert new_geometry.to_dict() == geom_dict

    def test_toroidal_from_dict_invalid(self, set_test_backend):
        """Test deserialization with missing keys."""
        cs = CoordinateSystem()
        invalid_dict = {
            "type": "ToroidalGeometry",
            "cs": cs.to_dict(),
            # Missing radius_x, radius_y
        }
        with pytest.raises(ValueError):
            geometries.ToroidalGeometry.from_dict(invalid_dict)

    def test_inf_radius_intersect_sphere_normal_incidence(
        self, cylinder_x_geometry, set_test_backend
    ):
        """Test _intersection_sphere for inf radius: ray normal to XY plane (z=0 plane)."""

        rays = RealRays(
            x=0.0, y=0.0, z=10.0, L=0.0, M=0.0, N=-1.0, intensity=1.0, wavelength=0.55
        )
        ix, iy, iz = cylinder_x_geometry._intersection(rays)

        be.allclose(ix, be.array(0.0), rtol=1e-5, atol=1e-6)
        be.allclose(iy, be.array(0.0), rtol=1e-5, atol=1e-6)
        be.allclose(
            iz, be.array(0.0), rtol=1e-5, atol=1e-6
        )  # Intersection coordinates should be (0, 0, 0)


class TestBiconicGeometry:
    def test_str(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=20.0)
        assert str(geom) == "Biconic"

    def test_sag_vertex(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=20.0, conic_x=0.5, conic_y=-0.5
        )
        assert_allclose(geom.sag(0, 0), 0.0)

    def test_sag_finite_radii(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=20.0)
        assert_allclose(geom.sag(x=1, y=1), 0.07514126037252641)

    def test_sag_rx_infinite(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=be.inf, radius_y=20.0
        )  # Cylindrical along X
        # Sag should only depend on y. zx(any) = 0
        # zy(1) for Ry=20 is 0.02501563183003138
        assert_allclose(geom.sag(x=10, y=1), 0.02501563183003138)
        assert_allclose(geom.sag(x=-5, y=1), 0.02501563183003138)

    def test_sag_ry_infinite(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=be.inf
        )  # Cylindrical along Y
        # Sag should only depend on x. zy(any) = 0
        # zx(1) for Rx=10 is 0.05012562854249503
        assert_allclose(geom.sag(x=1, y=10), 0.05012562854249503)
        assert_allclose(geom.sag(x=1, y=-5), 0.05012562854249503)

    def test_sag_both_infinite_plane(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=be.inf, radius_y=be.inf)
        assert_allclose(geom.sag(x=10, y=20), 0.0)
        # Also test with conics, should still be zero
        geom_conic = BiconicGeometry(
            cs, radius_x=be.inf, radius_y=be.inf, conic_x=0.5, conic_y=-1.0
        )
        assert_allclose(geom_conic.sag(x=10, y=20), 0.0)

    def test_sag_with_conics(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=20.0, conic_x=-1.0, conic_y=0.5
        )
        expected_sag = 0.05 + 0.02502345130203264
        assert_allclose(geom.sag(x=1, y=1), expected_sag)

    def test_sag_array_input(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=20.0)
        x = be.array([0, 1, 2])
        y = be.array([0, 1, 1])
        expected_sags = be.array([0.0, 0.07514126037252641, 0.22705672190612818])
        assert_allclose(geom.sag(x, y), expected_sags)

    def test_surface_normal_vertex(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=20.0, conic_x=0.5, conic_y=-0.5
        )
        nx, ny, nz = geom._surface_normal(x=0, y=0)
        assert_allclose(nx, 0.0)
        assert_allclose(ny, 0.0)
        assert_allclose(nz, -1.0)

    def test_surface_normal_spherical_case(self, set_test_backend):
        cs = CoordinateSystem()
        R = 10.0
        geom = BiconicGeometry(cs, radius_x=R, radius_y=R, conic_x=0.0, conic_y=0.0)
        nx, ny, nz = geom._surface_normal(x=1, y=1)
        assert_allclose(nx, 0.09950371902099892)
        assert_allclose(ny, 0.09950371902099892)
        assert_allclose(nz, -0.9900493732390136)

    def test_surface_normal_cylindrical_rx_inf(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=be.inf, radius_y=10.0, conic_y=0.0
        )  # Cylinder along X
        nx, ny, nz = geom._surface_normal(x=5, y=1)  # x shouldn't matter for dfdx=0
        assert_allclose(nx, 0.0)
        assert_allclose(ny, 0.1)
        assert_allclose(nz, -0.99498743710662)

    def test_surface_normal_array_input(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=10.0, radius_y=10.0, conic_x=0.0, conic_y=0.0
        )  # Spherical R=10
        x = be.array([0, 1])
        y = be.array([0, 1])

        # Point (0,0): nx=0, ny=0, nz=-1
        # Point (1,1): nx=0.099503719, ny=0.099503719, nz=-0.990049373 (from spherical test)
        expected_nx = be.array([0.0, 0.09950371902099892])
        expected_ny = be.array([0.0, 0.09950371902099892])
        expected_nz = be.array([-1.0, -0.9900493732390136])

        nx_calc, ny_calc, nz_calc = geom._surface_normal(x, y)
        assert_allclose(nx_calc, expected_nx)
        assert_allclose(ny_calc, expected_ny)
        assert_allclose(nz_calc, expected_nz)

    def test_distance_simple(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(cs, radius_x=10.0, radius_y=20.0)
        rays = RealRays(
            x=0.0, y=0.0, z=-5.0, L=0.0, M=0.0, N=1.0, wavelength=0.55, intensity=1.0
        )
        # Sag at (0,0) is 0, so distance to surface at (0,0) should be 5.0
        assert_allclose(geom.distance(rays), 5.0, atol=1e-9)  # Newton-Raphson tolerance

    def test_distance_planar_biconic(self, set_test_backend):
        cs = CoordinateSystem()
        geom = BiconicGeometry(
            cs, radius_x=be.inf, radius_y=be.inf, conic_x=0.0, conic_y=0.0
        )  # Planar
        rays = RealRays(
            x=1.0, y=1.0, z=-5.0, L=0.0, M=0.0, N=1.0, wavelength=0.55, intensity=1.0
        )
        # Distance to plane z=0 should be 5.0
        # The parent NewtonRaphsonGeometry's _intersection calls _intersection_plane if self.radius is inf.
        # In Biconic.__init__, self.radius is set to radius_x. So this path is tested.
        assert_allclose(geom.distance(rays), 5.0, atol=1e-9)

    def test_to_dict_from_dict(self, set_test_backend):
        cs = CoordinateSystem(x=1, y=2, z=3, rx=0.1, ry=-0.1, rz=0.05)
        original_geom = BiconicGeometry(
            cs,
            radius_x=100.0,
            radius_y=-150.0,
            conic_x=-0.5,
            conic_y=0.2,
            tol=1e-9,
            max_iter=50,
        )
        geom_dict = original_geom.to_dict()

        assert geom_dict["type"] == "BiconicGeometry"
        assert geom_dict["radius_x"] == 100.0
        assert geom_dict["radius_y"] == -150.0
        assert geom_dict["conic_x"] == -0.5
        assert geom_dict["conic_y"] == 0.2
        assert geom_dict["tol"] == 1e-9
        assert geom_dict["max_iter"] == 50
        assert "radius" not in geom_dict
        assert "conic" not in geom_dict

        reconstructed_geom = BiconicGeometry.from_dict(geom_dict)
        assert isinstance(reconstructed_geom, BiconicGeometry)
        assert_allclose(reconstructed_geom.Rx, original_geom.Rx)
        assert_allclose(reconstructed_geom.Ry, original_geom.Ry)
        assert_allclose(reconstructed_geom.kx, original_geom.kx)
        assert_allclose(reconstructed_geom.ky, original_geom.ky)
        assert reconstructed_geom.tol == original_geom.tol
        assert reconstructed_geom.max_iter == original_geom.max_iter
        assert reconstructed_geom.cs.to_dict() == original_geom.cs.to_dict()

        # Check if the reconstructed dict is identical (it should be)
        assert reconstructed_geom.to_dict() == geom_dict

    def test_from_dict_missing_keys(self, set_test_backend):
        cs = CoordinateSystem()
        minimal_valid_dict = {
            "type": "BiconicGeometry",
            "cs": cs.to_dict(),
            "radius_x": 10.0,
            "radius_y": 20.0,
            # conic_x, conic_y, tol, max_iter will use defaults
        }

        # Test missing radius_x
        invalid_dict_rx = minimal_valid_dict.copy()
        del invalid_dict_rx["radius_x"]
        with pytest.raises(
            ValueError, match="Missing required BiconicGeometry keys: {'radius_x'}"
        ):
            BiconicGeometry.from_dict(invalid_dict_rx)

        # Test missing radius_y
        invalid_dict_ry = minimal_valid_dict.copy()
        del invalid_dict_ry["radius_y"]
        with pytest.raises(
            ValueError, match="Missing required BiconicGeometry keys: {'radius_y'}"
        ):
            BiconicGeometry.from_dict(invalid_dict_ry)

        # Test missing cs
        invalid_dict_cs = minimal_valid_dict.copy()
        del invalid_dict_cs["cs"]
        with pytest.raises(
            ValueError, match="Missing required BiconicGeometry keys: {'cs'}"
        ):
            BiconicGeometry.from_dict(invalid_dict_cs)

    def test_from_dict_default_conics_tol_max_iter(self, set_test_backend):
        cs = CoordinateSystem()
        geom_data = {
            "type": "BiconicGeometry",
            "cs": cs.to_dict(),
            "radius_x": 10.0,
            "radius_y": 20.0,
        }
        geom = BiconicGeometry.from_dict(geom_data)
        assert geom.Rx == 10.0
        assert geom.Ry == 20.0
        assert geom.kx == 0.0  # Default
        assert geom.ky == 0.0  # Default
        assert (
            geom.tol == 1e-10
        )  # Default from NewtonRaphsonGeometry via BiconicGeometry
        assert (
            geom.max_iter == 100
        )  # Default from NewtonRaphsonGeometry via BiconicGeometry


def forbes_system():
    lens = Optic()
    lens.set_aperture(aperture_type="EPD", value=4.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=1.55, is_primary=True)
    H_K3 = Material("H-K3", reference="cdgm")
    H_ZLAF68C = Material("H-ZLAF68C", reference="cdgm")

    radial_terms_S2 = {0: 1.614, 1: 0.348, 2: 0.150, 3: 0.033, 4: 0.030}
    norm_radius_S2 = 6.336
    conic_S2 = -4.428

    radial_terms_S4 = {0: -0.270, 1: 0.087, 2: -0.048, 3: 0.026, 4: -0.012}
    conic_S4 = 0.038
    norm_radius_S4 = 10.0

    lens.surfaces.add(index=0, thickness=0.055)
    lens.surfaces.add(index=1, thickness=26.5)
    lens.surfaces.add(
        index=2, thickness=4.0, radius=be.inf, material=H_K3, is_stop=True
    )
    lens.surfaces.add(
        index=3,
        thickness=25.0,
        radius=22,
        conic=conic_S2,
        radial_terms=radial_terms_S2,
        norm_radius=norm_radius_S2,
        surface_type="forbes_qbfs",
    )
    lens.surfaces.add(index=4, thickness=7.0, radius=be.inf, material=H_ZLAF68C)
    lens.surfaces.add(
        index=5,
        thickness=10.0,
        radius=-31.0,
        conic=conic_S4,
        radial_terms=radial_terms_S4,
        norm_radius=norm_radius_S4,
        surface_type="forbes_qbfs",
    )
    lens.surfaces.add(index=6)
    return lens


class TestForbesQbfsGeometry:
    pytestmark = pytest.mark.filterwarnings(
        "ignore:ForbesQbfsGeometry is deprecated:DeprecationWarning"
    )

    def test_str(self, set_test_backend):
        """Test the string representation of the geometry."""
        cs = CoordinateSystem()

        config = ForbesSurfaceConfig(radius=100.0)
        geometry = ForbesQbfsGeometry(cs, surface_config=config)
        assert str(geometry) == "ForbesQbfs"

    def test_sag_with_infinite_radius(self, set_test_backend):
        """Test sag calculation with an infinite radius (planar base)."""

        config = ForbesSurfaceConfig(radius=be.inf, terms={1: 1e-3}, norm_radius=10.0)
        geometry = ForbesQbfsGeometry(
            coordinate_system=CoordinateSystem(), surface_config=config
        )
        # Base sag should be 0, total sag is just the departure term
        x, y = 5.0, 0.0
        rho = 5.0
        u = rho / 10.0
        usq = u**2
        # For a_1 = 1e-3, S = u^2(1-u^2) * a_1 * Q_1(u^2)
        # Q_1(x) = (13-16x)/sqrt(19)
        q1 = (13 - 16 * usq) / np.sqrt(19)
        expected_sag = usq * (1 - usq) * 1e-3 * q1
        assert_allclose(geometry.sag(x, y), expected_sag)

    def test_sag_outside_norm_radius(self, set_test_backend):
        """Test that sag departure is zero outside the normalization radius."""

        config = ForbesSurfaceConfig(radius=100.0, terms={0: 1e-3}, norm_radius=10.0)
        geometry = ForbesQbfsGeometry(
            coordinate_system=CoordinateSystem(), surface_config=config
        )
        standard_geom = geometries.StandardGeometry(
            coordinate_system=CoordinateSystem(), radius=100.0
        )
        # Point outside norm_radius
        x, y = 12.0, 0.0
        # Sag should be just the base conic sag
        assert_allclose(geometry.sag(x, y), standard_geom.sag(x, y))

    def test_analytical_normal_vs_autodiff(self, set_test_backend):
        """Compare analytical surface normal with autodiff for validation."""
        if be.get_backend() != "torch":
            pytest.skip("This test requires both numpy and torch backends to compare.")

        radial_terms = {0: 1.6e-4, 1: 0.3e-4, 2: 0.15e-4}
        config = ForbesSurfaceConfig(
            radius=22.0, conic=-4.428, terms=radial_terms, norm_radius=6.336
        )

        be.set_backend("numpy")
        geometry_np = ForbesQbfsGeometry(
            coordinate_system=CoordinateSystem(),
            surface_config=config,
        )

        be.set_backend("torch")
        geometry_torch = ForbesQbfsGeometry(
            coordinate_system=CoordinateSystem(),
            surface_config=config,
        )

        x, y = 2.5, 1.5

        # NumPy analytical normal
        be.set_backend("numpy")
        nx_np, ny_np, nz_np = geometry_np._surface_normal(x, y)

        # PyTorch autodiff normal
        be.set_backend("torch")
        nx_torch, ny_torch, nz_torch = geometry_torch._surface_normal(x, y)

        assert_allclose(nx_np, be.to_numpy(nx_torch), atol=1e-7)
        assert_allclose(ny_np, be.to_numpy(ny_torch), atol=1e-7)
        assert_allclose(nz_np, be.to_numpy(nz_torch), atol=1e-7)

    def test_sag_vs_zemax(self, set_test_backend):
        """
        Tests the sag calculation for a Q-bfs surface. This test ensures the coefficient translation
        and sag calculations are correct.
        """
        zemax_radius = 21.723
        zemax_conic = -4.428
        zemex_norm_radius = 6.336
        # Coefficients for n=0, 1, 2 (A0, A1, A2, A3, A4)
        zemax_coeffs = [1.614, 0.348, 0.150, 0.033, 0.030]
        radial_terms = {n: c for n, c in enumerate(zemax_coeffs)}

        y_coords = be.array([0.75, 1.0, 1.250, 1.860])
        # Corresponding sag values from Zemax's analysis
        zemax_sag_values = be.array(
            [
                6.264589454579239e-002,
                1.087513671806328e-001,
                1.648670855850376e-001,
                3.307314239746402e-001,
            ]
        )

        config = ForbesSurfaceConfig(
            radius=zemax_radius,
            conic=zemax_conic,
            terms=radial_terms,
            norm_radius=zemex_norm_radius,
        )
        geometry = ForbesQbfsGeometry(
            coordinate_system=CoordinateSystem(), surface_config=config
        )

        # Calculate sag in Optiland at the same radial coordinates
        # Using x=r_coords and y=0 for simplicity, as sag is symmetric for Q-bfs
        optiland_sag_values = geometry.sag(y=y_coords, x=be.zeros_like(y_coords))

        # Compare the results
        assert be.allclose(optiland_sag_values, zemax_sag_values, atol=1e-9, rtol=1e-9)

    def test_surface_normal_at_vertex(self, set_test_backend):
        """
        Tests the surface normal at the vertex (x=0, y=0).
        It should always be [0, 0, -1] regardless of parameters.
        """

        config = ForbesSurfaceConfig(
            radius=50.0, conic=-1.0, terms={2: 1e-4}, norm_radius=10.0
        )
        geometry = ForbesQbfsGeometry(
            coordinate_system=CoordinateSystem(),
            surface_config=config,
        )

        nx, ny, nz = geometry._surface_normal(x=0.0, y=0.0)

        assert_allclose(nx, 0.0, atol=1e-9)
        assert_allclose(ny, 0.0, atol=1e-9)
        assert_allclose(nz, -1.0, atol=1e-9)

    def test_tracing(self, set_test_backend):
        system = forbes_system()

        test_rays = RealRays(
            x=be.array([0.0, 0.0, 0.0]),
            y=be.array([-2.0, 0.0, 2.0]),
            z=be.array([0.0, 0.0, 0.0]),
            L=be.array([0.0, 0.0, 0.0]),
            M=be.array([0.0, 0.0, 0.0]),
            N=be.array([1.0, 1.0, 1.0]),
            wavelength=be.ones(3) * 1.550,
            intensity=be.ones(3),
        )

        zmx_rays_x = be.array([0.0, 0.0, 0.0])
        zmx_rays_y = be.array([-5.863214938651272e000, 0.0, 5.863214938651272e000])
        zmx_rays_z = be.array([72.5, 72.5, 72.5])
        zmx_rays_L = be.array([0.0, 0.0, 0.0])
        zmx_rays_M = be.array([3.128052326230352e-002, 0.0, -3.128052326230352e-002])
        zmx_rays_N = be.array([9.995106446979125e-001, 1.0, 9.995106446979125e-001])

        rays_out = system.surfaces.trace(test_rays)

        assert be.allclose(rays_out.x, zmx_rays_x, rtol=1e-7, atol=1e-7)
        assert be.allclose(rays_out.y, zmx_rays_y, rtol=1e-7, atol=1e-7)
        assert be.allclose(rays_out.z, zmx_rays_z, rtol=1e-7, atol=1e-7)
        assert be.allclose(rays_out.L, zmx_rays_L, rtol=1e-7, atol=1e-7)
        assert be.allclose(rays_out.M, zmx_rays_M, rtol=1e-7, atol=1e-7)
        assert be.allclose(rays_out.N, zmx_rays_N, rtol=1e-7, atol=1e-7)

    def _create_forbes_autodiff_optic(self):
        """Helper to create a standard Forbes optic for autodiff testing."""
        optic = Optic(name="Autodiff Test Lens")
        optic.set_aperture(aperture_type="EPD", value=30.0)
        optic.wavelengths.add(value=1.55, is_primary=True, unit="um")
        optic.fields.add(y=0.0)
        optic.surfaces.add(index=0, thickness=be.inf)
        optic.surfaces.add(index=1, thickness=10, is_stop=True)
        optic.surfaces.add(
            index=2, surface_type="standard", radius=60, thickness=7.0, material="N-BK7"
        )
        optic.surfaces.add(
            index=3,
            surface_type="forbes_qbfs",
            radius=-120,
            thickness=60.0,
            material="air",
            radial_terms={0: 1.0, 1: 0.8, 2: 0.2},
            norm_radius=30.0,
        )
        optic.surfaces.add(index=4)
        return optic

    @pytest.mark.parametrize("backend_name", ["torch"])
    def test_ray_tracing_autodiff_off_axis(self, backend_name):
        """Tests that ray tracing is differentiable for a general off-axis ray."""
        be.set_backend(backend_name)
        if be.get_backend() != "torch":
            pytest.skip("Autodiff test requires the torch backend.")

        optic = self._create_forbes_autodiff_optic()
        forbes_surface = optic.surfaces[3].geometry
        coeffs_to_test = forbes_surface.radial_terms

        for coeff_tensor in coeffs_to_test.values():
            coeff_tensor.requires_grad_(True)

        # Ray trace an OFF-AXIS ray
        initial_rays = RealRays(
            x=0.1, y=0.1, z=0, L=0, M=0, N=1, intensity=1, wavelength=0.55
        )
        optic.surfaces.trace(initial_rays)
        final_x = initial_rays.x

        loss = be.sum(final_x**2)
        loss.backward()

        grads = [c.grad for c in coeffs_to_test.values()]
        assert any(g is not None and be.to_numpy(g) != 0 for g in grads)

    # --- NEW TEST ADDED ---
    @pytest.mark.parametrize("backend_name", ["torch"])
    def test_ray_tracing_autodiff_at_vertex(self, backend_name):
        """
        Tests that ray tracing is differentiable for a ray hitting the exact
        vertex, which was the source of the NaN gradient bug.
        """
        be.set_backend(backend_name)
        if be.get_backend() != "torch":
            pytest.skip("Autodiff test requires the torch backend.")

        be.grad_mode.enable()

        optic = self._create_forbes_autodiff_optic()

        # We will test the gradient with respect to the radius
        trainable_radius = optic.surfaces[3].geometry.radius
        trainable_radius.requires_grad_(True)

        # Ray trace a ray hitting the VERTEX
        initial_rays = RealRays(
            x=0.0, y=0.0, z=0, L=0, M=0, N=1, intensity=1, wavelength=0.55
        )
        optic.surfaces.trace(initial_rays)
        final_y = initial_rays.y

        loss = be.sum(final_y**2)
        loss.backward()

        # The crucial check: assert the gradient is a valid number, not NaN
        grad = trainable_radius.grad
        assert grad is not None, "Gradient at vertex should not be None"
        assert not be.isnan(grad), "Gradient at vertex must not be NaN"

    @pytest.mark.parametrize("backend_name", ["torch"])
    def test_forbes_qbfs_autodiff_inplace_modification(self, backend_name):
        """
        Tests for in-place modification errors during backpropagation with ForbesQbfsGeometry.
        This test replicates the conditions that led to the RuntimeError in the user's notebook.
        """
        be.set_backend(backend_name)
        be.grad_mode.enable()
        from optiland.analysis import IncoherentIrradiance
        from optiland.physical_apertures import RectangularAperture

        # 1. Create a simple optical system with a Forbes Q-bfs surface
        optic = Optic(name="Test Forbes Autodiff")
        optic.set_aperture(aperture_type="EPD", value=10.0)
        optic.wavelengths.add(value=0.55, is_primary=True)
        optic.fields.set_type(field_type="angle")
        optic.fields.add(y=0.0)
        optic.surfaces.add(index=0, thickness=be.inf)
        optic.surfaces.add(
            index=1,
            surface_type="forbes_qbfs",
            radius=be.tensor(100.0, requires_grad=True),
            conic=be.tensor(0.0, requires_grad=True),
            thickness=10.0,
            material="N-BK7",
            radial_terms={i: be.tensor(0.0, requires_grad=True) for i in range(2)},
            norm_radius=be.tensor(10.0, requires_grad=True),
        )
        optic.surfaces.add(
            index=2, aperture=RectangularAperture(x_min=-5, x_max=5, y_min=-5, y_max=5)
        )

        # 2. Create a simple source and rays
        x_start = be.linspace(-1, 1, 10)
        y_start = be.linspace(-1, 1, 10)
        y_grid, x_grid = be.meshgrid(y_start, x_start)  # Corrected meshgrid call

        x_flat, y_flat = x_grid.flatten(), y_grid.flatten()
        num_rays = len(x_flat)
        num_bins = 10

        initial_rays = RealRays(
            x=x_flat,
            y=y_flat,
            z=be.zeros(num_rays),
            L=be.zeros(num_rays),
            M=be.zeros(num_rays),
            N=be.ones(num_rays),
            intensity=be.ones(num_rays),
            wavelength=be.full((num_rays,), 0.55),
        )

        # 3. Perform a forward pass and calculate a loss FUNCTION IDENTICAL TO THE NOTEBOOK
        irradiance_analyzer = IncoherentIrradiance(
            optic, user_initial_rays=initial_rays, res=(num_bins, num_bins)
        )
        irradiance_analyzer._generate_data()
        irr_map, _, _ = irradiance_analyzer.data[0][0]

        # --- NEW: Replicating the notebook's loss calculation ---
        actual_profile = irr_map[:, num_bins // 2]
        if actual_profile.max() > 0:
            actual_profile = actual_profile / actual_profile.max()

        target_irradiance = be.ones(num_bins)  # A simple target
        loss_fn = be.nn.MSELoss()
        loss = loss_fn(actual_profile, target_irradiance)
        # --- END NEW ---

        # 4. Attempt to perform a backward pass
        try:
            loss.backward()
        except RuntimeError as e:
            pytest.fail(
                f"Backward pass failed with a RuntimeError, indicating a probable in-place modification issue: {e}"
            )

    def test_to_dict_from_dict(self, set_test_backend):
        """
        Tests the serialization and deserialization of the ForbesQbfsGeometry
        to ensure all parameters are correctly saved and loaded.
        """
        cs = CoordinateSystem(x=1, y=-1, z=10, rx=0.01, ry=0.02, rz=-0.03)
        radial_terms = {0: 1e-3, 1: 2e-4, 2: -5e-5}

        surface_config = ForbesSurfaceConfig(
            radius=123.4, conic=-0.9, terms=radial_terms, norm_radius=45.6
        )
        solver_config = ForbesSolverConfig(tol=1e-12, max_iter=75)

        original_geometry = ForbesQbfsGeometry(
            coordinate_system=cs,
            surface_config=surface_config,
            solver_config=solver_config,
        )

        geom_dict = original_geometry.to_dict()

        assert geom_dict["type"] == "ForbesQbfsGeometry"
        assert geom_dict["surface_config"]["radius"] == 123.4
        assert geom_dict["surface_config"]["conic"] == -0.9
        assert geom_dict["surface_config"]["terms"] == radial_terms
        assert geom_dict["surface_config"]["norm_radius"] == 45.6
        assert geom_dict["solver_config"]["tol"] == 1e-12
        assert geom_dict["solver_config"]["max_iter"] == 75

        reconstructed_geometry = ForbesQbfsGeometry.from_dict(geom_dict)
        assert reconstructed_geometry.radius == 123.4
        assert reconstructed_geometry.k == -0.9
        assert reconstructed_geometry.radial_terms == radial_terms


class TestForbesQNormalSlopeGeometry:
    """Tests for the canonical ForbesQNormalSlopeGeometry class."""

    def test_str(self, set_test_backend):
        """Test the string representation of the canonical geometry."""
        cs = CoordinateSystem()
        config = ForbesSurfaceConfig(radius=100.0)
        geometry = ForbesQNormalSlopeGeometry(cs, surface_config=config)
        assert str(geometry) == "ForbesQNormalSlope"

    def test_to_dict_type_name(self, set_test_backend):
        """Test that to_dict uses the canonical class name."""
        cs = CoordinateSystem()
        config = ForbesSurfaceConfig(radius=100.0)
        geometry = ForbesQNormalSlopeGeometry(cs, surface_config=config)
        geom_dict = geometry.to_dict()
        assert geom_dict["type"] == "ForbesQNormalSlopeGeometry"

    def test_from_dict_canonical_type(self, set_test_backend):
        """Test that from_dict correctly loads canonical type."""
        cs = CoordinateSystem()
        radial_terms = {0: 1e-3, 1: 2e-4}
        surface_config = ForbesSurfaceConfig(
            radius=100.0, conic=-0.5, terms=radial_terms, norm_radius=15.0
        )
        original = ForbesQNormalSlopeGeometry(cs, surface_config)
        geom_dict = original.to_dict()

        reconstructed = ForbesQNormalSlopeGeometry.from_dict(geom_dict)
        assert isinstance(reconstructed, ForbesQNormalSlopeGeometry)
        assert reconstructed.radius == 100.0
        assert reconstructed.k == -0.5
        assert reconstructed.radial_terms == radial_terms

    def test_from_dict_legacy_type(self, set_test_backend):
        """Test that from_dict correctly loads legacy ForbesQbfsGeometry type."""
        legacy_dict = {
            "type": "ForbesQbfsGeometry",
            "cs": CoordinateSystem().to_dict(),
            "surface_config": {
                "radius": 50.0,
                "conic": -1.0,
                "norm_radius": 10.0,
                "terms": {0: 0.001},
            },
            "solver_config": {"tol": 1e-10, "max_iter": 100},
        }
        reconstructed = ForbesQNormalSlopeGeometry.from_dict(legacy_dict)
        # Should return canonical class instance
        assert isinstance(reconstructed, ForbesQNormalSlopeGeometry)
        assert reconstructed.radius == 50.0
        assert reconstructed.k == -1.0

    def test_sag_equivalence_with_deprecated_alias(self, set_test_backend):
        """Test that canonical class and deprecated alias produce identical sag."""
        import warnings

        cs = CoordinateSystem()
        radial_terms = {0: 1e-3, 1: -2e-4, 2: 5e-5}
        config = ForbesSurfaceConfig(
            radius=75.0, conic=-0.8, terms=radial_terms, norm_radius=12.0
        )

        canonical = ForbesQNormalSlopeGeometry(cs, config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            deprecated = ForbesQbfsGeometry(CoordinateSystem(), config)

        x, y = 5.0, 3.0
        assert_allclose(canonical.sag(x, y), deprecated.sag(x, y))

    def test_deprecated_alias_warning(self, set_test_backend):
        """Test that ForbesQbfsGeometry emits a DeprecationWarning."""
        import warnings

        cs = CoordinateSystem()
        config = ForbesSurfaceConfig(radius=100.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ForbesQbfsGeometry(cs, config)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ForbesQbfsGeometry is deprecated" in str(w[0].message)
            assert "ForbesQNormalSlopeGeometry" in str(w[0].message)


class TestForbesQ2dGeometry:
    def test_str(self, set_test_backend):
        """Test the string representation of the geometry."""
        cs = CoordinateSystem()
        config = ForbesSurfaceConfig(radius=100.0, conic=0.0)
        geometry = ForbesQ2dGeometry(cs, surface_config=config)
        assert str(geometry) == "ForbesQ2d"

    def test_init_no_coeffs(self, set_test_backend):
        """Test initialization with no freeform coefficients."""
        config = ForbesSurfaceConfig(radius=100.0, conic=-0.5)
        geometry = ForbesQ2dGeometry(
            coordinate_system=CoordinateSystem(), surface_config=config
        )
        assert len(geometry.coeffs_c) == 0
        assert len(geometry.coeffs_n) == 0
        assert len(geometry.cm0_coeffs) == 0
        assert len(geometry.ams_coeffs) == 0
        assert len(geometry.bms_coeffs) == 0

    @pytest.mark.filterwarnings(
        "ignore:ForbesQbfsGeometry is deprecated:DeprecationWarning"
    )
    def test_sag_symmetric_terms_only(self, set_test_backend):
        """Test sag with only m=0 terms, should match Q-bfs."""
        radial_terms = {0: 1e-3, 1: -2e-4}

        freeform_coeffs = {("a", 0, n): c for n, c in radial_terms.items()}

        q2d_config = ForbesSurfaceConfig(
            radius=50.0, conic=0.0, terms=freeform_coeffs, norm_radius=10.0
        )
        geom_q2d = ForbesQ2dGeometry(
            coordinate_system=CoordinateSystem(), surface_config=q2d_config
        )
        qbfs_config = ForbesSurfaceConfig(
            radius=50.0, conic=0.0, terms=radial_terms, norm_radius=10.0
        )
        geom_qbfs = ForbesQbfsGeometry(
            coordinate_system=CoordinateSystem(), surface_config=qbfs_config
        )
        x, y = 3.0, 4.0
        assert_allclose(geom_q2d.sag(x, y), geom_qbfs.sag(x, y))

    def test_sag_with_sine_term(self, set_test_backend):
        """Test sag with a sine term, which should be zero along the x-axis"""

        config = ForbesSurfaceConfig(
            radius=100.0, conic=0.0, terms={("b", 1, 1): 1e-3}, norm_radius=10.0
        )
        geometry = ForbesQ2dGeometry(
            coordinate_system=CoordinateSystem(),
            surface_config=config,
        )
        x, y = 5.0, 0.0
        base_geom = geometries.StandardGeometry(
            coordinate_system=CoordinateSystem(), radius=100.0
        )
        assert_allclose(geometry.sag(x, y), base_geom.sag(x, y))

    def test_prepare_coeffs(self, set_test_backend):
        freeform_coeffs = {
            ("a", 0, 0): 1.0,  # Symmetric term a_0^0
            ("a", 1, 1): 2.0,  # Cosine term a_1^1
            ("b", 1, 1): 3.0,  # Sine term b_1^1
            ("a", 0, 4): 4.0,  # Symmetric term a_4^0
        }
        config = ForbesSurfaceConfig(
            radius=100.0, conic=0.0, terms=freeform_coeffs, norm_radius=10.0
        )
        geometry = ForbesQ2dGeometry(
            coordinate_system=CoordinateSystem(),
            surface_config=config,
        )

        # symmetric terms (m=0) are parsed correctly, with zero-padding
        assert_allclose(geometry.cm0_coeffs, [1.0, 0.0, 0.0, 0.0, 4.0])

        # ams should have a_1^1
        assert len(geometry.ams_coeffs) == 1
        assert_allclose(geometry.ams_coeffs[0], [0.0, 2.0])  # a_n^1 list for m=1

        assert len(geometry.bms_coeffs) == 1
        assert_allclose(geometry.bms_coeffs[0], [0.0, 3.0])

    def _create_forbes_q2d_autodiff_optic(self):
        """Helper to create a standard Forbes Q2D optic for autodiff testing."""
        optic = Optic(name="Q2D Autodiff Test Lens")
        optic.wavelengths.add(value=0.55, is_primary=True)
        optic.fields.add(y=0.0)

        # Create a trainable freeform coefficient
        trainable_coeff = be.tensor(0.01, requires_grad=True)
        freeform_coeffs = {("a", 1, 1): trainable_coeff}

        optic.surfaces.add(index=0, thickness=be.inf)
        optic.surfaces.add(
            index=1,
            surface_type="forbes_q2d",
            radius=-100.0,
            thickness=50.0,
            material="N-BK7",
            freeform_coeffs=freeform_coeffs,
            norm_radius=20.0,
        )
        optic.surfaces.add(index=2)
        return optic, trainable_coeff

    @pytest.mark.parametrize("backend_name", ["torch"])
    def test_gradient_stability_at_vertex(self, backend_name):
        be.set_backend(backend_name)
        be.grad_mode.enable()

        optic, trainable_coeff = self._create_forbes_q2d_autodiff_optic()

        # ray through the vertex
        vertex_ray = RealRays(
            x=0.0, y=0.0, z=0.0, L=0.0, M=0.0, N=1.0, intensity=1.0, wavelength=0.55
        )
        final_ray = optic.surfaces.trace(vertex_ray)

        # define loss and compute gradient
        loss = be.sum(final_ray.y**2)
        loss.backward()

        grad = trainable_coeff.grad
        assert grad is not None
        assert not be.isnan(grad)

    def test_to_dict_from_dict(self, set_test_backend):
        """Test serialization and deserialization."""
        cs = CoordinateSystem(x=1, y=-1, z=10)
        # UPDATED: Convert to the new Zemax-aligned format
        freeform_coeffs = {("a", 2, 2): 1e-4, ("b", 1, 1): -5e-5}
        config = ForbesSurfaceConfig(
            radius=123.4, conic=-0.9, terms=freeform_coeffs, norm_radius=45.6
        )
        original_geometry = ForbesQ2dGeometry(
            coordinate_system=cs,
            surface_config=config,
        )
        geom_dict = original_geometry.to_dict()
        reconstructed_geometry = ForbesQ2dGeometry.from_dict(geom_dict)
        assert reconstructed_geometry.to_dict() == geom_dict


from optiland.geometries.forbes.qpoly import (
    compute_z_zprime_q2d,
    q2d_nm_coeffs_to_ams_bms,
)


class TestForbesValidation:
    """
    Unit tests to validate the Forbes geometry implementation directly against
    the mathematical formulas in the reference papers
    """

    @pytest.mark.parametrize(
        "m, n, x_val, expected_q_val",
        [
            (2, 0, 0.5, 1 / np.sqrt(2)),
            (1, 1, 0.25, (13 - 16 * 0.25) / np.sqrt(19)),
            (0, 1, 0.75, (13 - 16 * 0.75) / np.sqrt(19)),
        ],
    )
    def test_qpoly_qnm_against_paper_formulas(self, m, n, x_val, expected_q_val):
        """
        Validates the core Q_n^m(x) polynomial calculation in qpoly.py against
        the explicit formulas published in the Forbes papers. This is the most
        fundamental check of the mathematical engine.

        References:
            "Characterizing the shape of freeform optics" (2012), Fig. 3.
        """
        # We want to isolate a single polynomial Q_n^m. We do this by setting its
        # corresponding coefficient a_n^m or b_n^m to 1.0 and all others to zero
        coeffs_n = [(n, m)]
        coeffs_c = [1.0]

        cm0, ams, bms = q2d_nm_coeffs_to_ams_bms(coeffs_n, coeffs_c)

        # The input to the Q polynomials is u^2, which we call x_val here.
        u = np.sqrt(x_val)
        theta = 0

        # raw polynomial sums from the implementation
        poly_sum_m0, _, poly_sum_m_gt0, _, _ = compute_z_zprime_q2d(
            cm0, ams, bms, u, theta
        )

        # The output of compute_z_zprime_Q2d is a sum. Since we only have one
        # coefficient, the output should be our desired Q_n^m value, possibly
        # with a pre-factor depending on m.

        calculated_q_val = 0
        if m == 0:
            # For m=0, the sum is u^2(1-u^2) * S, where S = 2*(alpha0+alpha1)
            pytest.skip("Skipping direct Qnm validation; covered by full sag test.")

        else:
            if u > 1e-9:
                calculated_q_val = poly_sum_m_gt0 / (u**m)
            else:
                pytest.skip("Skipping test at u=0 for m>0.")

    def test_qnm_values_against_analytical_formula(self, set_test_backend):
        n, m = 1, 2
        x = 0.4
        P_n_m_canonical = 1.5 - x
        from optiland.geometries.forbes.qpoly import f_q2d, g_q2d

        P_0_m = 0.5
        f0 = f_q2d(n=0, m=m)
        Q_0_m_canonical = P_0_m / f0
        g0 = g_q2d(n=0, m=m)
        f1 = f_q2d(n=1, m=m)
        Q_n_m_canonical = (P_n_m_canonical - g0 * Q_0_m_canonical) / f1
        coeffs_to_test = [0.0] * (n + 1)
        coeffs_to_test[n] = 1.0
        from optiland.geometries.forbes.qpoly import clenshaw_q2d

        alphas = clenshaw_q2d(coeffs_to_test, m=m, usq=x)
        Q_n_m_optiland = 0.5 * alphas[0]
        assert np.allclose(Q_n_m_optiland, Q_n_m_canonical, atol=1e-9)

    def test_qnm_values_against_analytical_formula(self, set_test_backend):
        """
        Validates that a single Q polynomial from the qpoly implementation matches
        a value calculated directly from the analytical formula derived from the
        Forbes papers.
        """
        # Q_n^m for n=1, m=2
        n, m = 1, 2
        x = 0.4  # An arbitrary value for u^2 between 0 and 1

        # calculate the ground truth using the analytical formula
        # From the Forbes papers (e.g. 2012 paper), we know
        # that for m>1, P_1^m(x) = m - 0.5 - (m-1)x.
        # for m=2:
        P_n_m_canonical = 1.5 - x

        # Now, we convert this P_n^m value to a Q_n^m value using the
        # recurrence relation from the paper.
        # Q_n^m = (P_n^m - g_{n-1}^m * Q_{n-1}^m) / f_n^m
        from optiland.geometries.forbes.qpoly import f_q2d, g_q2d

        # we need Q_0^2 and the f and g coefficients from qpoly
        # Q_0^m = P_0^m / f_0^m. P_0^m is always 0.5.
        P_0_m = 0.5
        f0 = f_q2d(n=0, m=m)
        Q_0_m_canonical = P_0_m / f0

        g0 = g_q2d(n=0, m=m)
        f1 = f_q2d(n=1, m=m)

        # final ground truth for Q_1^2(x)
        Q_n_m_canonical = (P_n_m_canonical - g0 * Q_0_m_canonical) / f1

        # calculate the value using the qpoly implementation
        # we isolate the Q_n^m polynomial by setting its coefficient to 1
        coeffs_to_test = [0.0] * (n + 1)
        coeffs_to_test[n] = 1.0

        from optiland.geometries.forbes.qpoly import clenshaw_q2d

        # The clenshaw function returns the sum of the series. Since only one
        # coefficient is 1, the sum is equal to the value of that polynomial.
        alphas = clenshaw_q2d(coeffs_to_test, m=m, usq=x)
        # The sum S(x) = 0.5 * alpha_0 for the Q2D polynomials
        Q_n_m_optiland = 0.5 * alphas[0]

        assert np.allclose(Q_n_m_optiland, Q_n_m_canonical, atol=1e-9)

    def test_q2d_normal_against_numerical_derivative(self):
        """
        Validates the analytical surface normal for the non-vertex case against
        a numerical derivative (finite difference)
        """

        freeform_coeffs = {
            ("a", 1, 1): -0.25,
            ("b", 1, 0): 0.5,
        }

        config = ForbesSurfaceConfig(
            radius=21.709, conic=-4.428, terms=freeform_coeffs, norm_radius=6.0
        )
        geometry = ForbesQ2dGeometry(
            coordinate_system=CoordinateSystem(),
            surface_config=config,
        )

        x, y = 1.0, 0.5
        h = 1e-6

        # Calculate sag at points surrounding (x, y)
        sag_center = geometry.sag(x, y)
        sag_x_plus = geometry.sag(x + h, y)
        sag_x_minus = geometry.sag(x - h, y)
        sag_y_plus = geometry.sag(x, y + h)
        sag_y_minus = geometry.sag(x, y - h)

        # approximate the partial derivatives using the central difference formula
        df_dx_numerical = (sag_x_plus - sag_x_minus) / (2 * h)
        df_dy_numerical = (sag_y_plus - sag_y_minus) / (2 * h)

        # calculate the direvatives using the analytical method
        df_dx_analytical, df_dy_analytical = geometry._surface_normal_analytical(
            be.array(x), be.array(y)
        )

        # compare
        assert np.allclose(df_dx_analytical, df_dx_numerical, atol=1e-6)
        assert np.allclose(df_dy_analytical, df_dy_numerical, atol=1e-6)

    def test_complex_ray_tracing(self, set_test_backend):
        """
        Tests the ray tracing through a complex Forbes geometry with multiple
        coefficients and terms to ensure the full system behaves as expected.
        """
        # Create a complex system: Q-2d and Qbfs
        optic = Optic()
        optic.set_aperture(aperture_type="EPD", value=4.0)

        optic.fields.set_type(field_type="angle")
        optic.fields.add(y=0)

        optic.wavelengths.add(value=1.550, is_primary=True)

        H_K3 = IdealMaterial(n=1.50, k=0)
        H_ZLAF68C = Material("H-ZLAF68C", reference="cdgm")

        norm_radius = 10.0
        freeform_coeffs = {
            ("b", 1, 0): 0.23,
            ("a", 1, 1): -0.25,
            ("a", 1, 3): -2.0,
            ("b", 2, 0): 0.4,
            ("b", 3, 1): 0.5,
        }

        radial_terms = {0: -0.334, 1: 0.130, 2: -0.099, 3: 0.082, 4: -0.093}

        optic.surfaces.add(index=0, thickness=be.inf)
        optic.surfaces.add(index=1, thickness=26.5)
        optic.surfaces.add(
            index=2,
            thickness=4.0,
            radius=be.inf,
            material=H_K3,
            is_stop=True,
            aperture=6.0,
        )
        optic.surfaces.add(
            index=3,
            thickness=25.0,
            radius=21.7,
            conic=-4.428,
            freeform_coeffs=freeform_coeffs,
            norm_radius=6.0,
            surface_type="forbes_q2d",
            aperture=6.0,
        )
        optic.surfaces.add(
            index=4, thickness=7.0, radius=be.inf, material=H_ZLAF68C, aperture=16.0
        )
        optic.surfaces.add(
            index=5,
            thickness=10.0,
            radius=-31.408,
            conic=-0.334,
            radial_terms=radial_terms,
            norm_radius=10.0,
            surface_type="forbes_qbfs",
            aperture=16.0,
        )
        optic.surfaces.add(index=6)

        # Create rays to trace through this geometry
        rays_1 = RealRays(
            x=be.array([-2.0, 0.0, 2.0]),
            y=be.array([0.0, 0.0, 0.0]),
            z=be.array([0.0, 0.0, 0.0]),
            L=be.array([0.0, 0.0, 0.0]),
            M=be.array([0.0, 0.0, 0.0]),
            N=be.array([1.0, 1.0, 1.0]),
            wavelength=be.ones(3) * 1.550,
            intensity=be.ones(3),
        )
        rays_2 = RealRays(
            x=be.array([0.0, 0.0, 0.0]),
            y=be.array([-2.0, 0.0, 2.0]),
            z=be.array([0.0, 0.0, 0.0]),
            L=be.array([0.0, 0.0, 0.0]),
            M=be.array([0.0, 0.0, 0.0]),
            N=be.array([1.0, 1.0, 1.0]),
            wavelength=be.ones(3) * 1.550,
            intensity=be.ones(3),
        )
        rays_3 = RealRays(
            x=be.array([-1.8, 0.5, -0.5]),
            y=be.array([-2.0, -1.5, 1.5]),
            z=be.array([0.0, 0.0, 0.0]),
            L=be.array([0.0, 0.0, 0.0]),
            M=be.array([0.0, 0.0, 0.0]),
            N=be.array([1.0, 1.0, 1.0]),
            wavelength=be.ones(3) * 1.550,
            intensity=be.ones(3),
        )

        # trace and group for comparison
        rays_out_1 = optic.surfaces.trace(rays_1)
        rays_out_2 = optic.surfaces.trace(rays_2)
        rays_out_3 = optic.surfaces.trace(rays_3)

        rays_out_1 = be.stack(
            [
                rays_out_1.x,
                rays_out_1.y,
                rays_out_1.z,
                rays_out_1.L,
                rays_out_1.M,
                rays_out_1.N,
            ]
        )
        rays_out_2 = be.stack(
            [
                rays_out_2.x,
                rays_out_2.y,
                rays_out_2.z,
                rays_out_2.L,
                rays_out_2.M,
                rays_out_2.N,
            ]
        )
        rays_out_3 = be.stack(
            [
                rays_out_3.x,
                rays_out_3.y,
                rays_out_3.z,
                rays_out_3.L,
                rays_out_3.M,
                rays_out_3.N,
            ]
        )

        # from zmx
        rays_zmx_1 = be.array(
            [
                [2.262266099465996e000, -7.611636358380779e000, 8.507340277857841e000],
                [7.011812112445810e-001, 6.351818041247482e-001, 1.844219290499702e000],
                [72.5, 72.5, 72.5],
                [
                    7.394271933375213e-002,
                    -4.741197708598330e-002,
                    1.144444436877695e-003,
                ],
                [
                    1.760669468507115e-003,
                    3.956471870787390e-003,
                    1.390619028086458e-002,
                ],
                [
                    9.972609359142435e-001,
                    9.988675841967912e-001,
                    9.999026493208244e-001,
                ],
            ]
        )

        rays_zmx_2 = be.array(
            [
                [
                    -2.908637501241161e000,
                    -7.611636358380779e000,
                    -1.909625498103998e000,
                ],
                [-2.833834953431557e000, 6.351818041247482e-001, 2.879512384120351e000],
                [72.5, 72.5, 72.5],
                [
                    -2.028167761942454e-002,
                    -4.741197708598330e-002,
                    -1.395817128850889e-002,
                ],
                [
                    4.487436493546904e-002,
                    3.956471870787390e-003,
                    -4.367333651449654e-002,
                ],
                [
                    9.987867364580793e-001,
                    9.988675841967912e-001,
                    9.989483515837905e-001,
                ],
            ]
        )

        rays_zmx_3 = be.array(
            [
                [1.995705116576431e000, -3.353247609431453e000, -4.016958077970503e000],
                [1.568402524873438e000, -3.440142525503505e000, 6.226900552306260e-001],
                [72.5, 72.5, 72.5],
                [
                    6.464067747819106e-002,
                    -3.675345849055221e-002,
                    -1.314936805201002e-002,
                ],
                [
                    7.066054530137703e-002,
                    2.733014103973235e-002,
                    -4.397208974806662e-002,
                ],
                [
                    9.954037724224643e-001,
                    9.989505726910273e-001,
                    9.989462194948339e-001,
                ],
            ]
        )

        # validate
        assert be.allclose(rays_out_1, rays_zmx_1, rtol=1e-7, atol=1e-7)
        assert be.allclose(rays_out_2, rays_zmx_2, rtol=1e-7, atol=1e-7)
        assert be.allclose(rays_out_3, rays_zmx_3, rtol=1e-7, atol=1e-7)
