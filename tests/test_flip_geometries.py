from __future__ import annotations

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    BiconicGeometry,
    ChebyshevPolynomialGeometry,
    EvenAsphere,
    OddAsphere,
    Plane,
    PolynomialGeometry,
    StandardGeometry,
    ToroidalGeometry,
    ZernikePolynomialGeometry,
)

# Default coordinate system for tests
cs = CoordinateSystem()


def test_flip_standard_geometry():
    # Test StandardGeometry
    initial_radius = 10.0
    initial_conic = -0.5
    geom = StandardGeometry(cs, radius=initial_radius, conic=initial_conic)

    # Store initial values
    assert geom.radius == initial_radius
    assert geom.k == initial_conic

    geom.flip()

    # Assertions after flip
    assert geom.radius == -initial_radius
    assert geom.k == initial_conic


def test_flip_plane_geometry():
    # Test Plane
    geom = Plane(cs)
    initial_radius = geom.radius  # Should be be.inf

    geom.flip()

    assert geom.radius == initial_radius  # be.inf should remain be.inf


def test_flip_biconic_geometry():
    # Test BiconicGeometry
    initial_rx = 10.0
    initial_ry = -20.0
    initial_kx = 0.1
    initial_ky = -0.2
    geom = BiconicGeometry(
        cs,
        radius_x=initial_rx,
        radius_y=initial_ry,
        conic_x=initial_kx,
        conic_y=initial_ky,
    )

    initial_cx_val = geom.cx
    initial_cy_val = geom.cy

    assert geom.Rx == initial_rx
    assert geom.Ry == initial_ry
    assert geom.kx == initial_kx
    assert geom.ky == initial_ky

    geom.flip()

    assert geom.Rx == -initial_rx
    assert geom.Ry == -initial_ry
    assert geom.kx == initial_kx
    assert geom.ky == initial_ky
    assert be.allclose(geom.cx, -initial_cx_val if initial_rx != 0 else 0.0)
    assert be.allclose(geom.cy, -initial_cy_val if initial_ry != 0 else 0.0)


def test_flip_chebyshev_geometry():
    initial_radius = 100.0
    initial_conic = -0.8
    initial_coeffs = [[0, 0.1], [0.2, 0.3]]
    geom = ChebyshevPolynomialGeometry(
        cs,
        radius=initial_radius,
        conic=initial_conic,
        coefficients=initial_coeffs,
        norm_x=1.0,
        norm_y=1.0,
    )

    assert geom.radius == initial_radius
    assert geom.k == initial_conic
    assert be.allclose(geom.coefficients, be.array(initial_coeffs))

    geom.flip()

    assert geom.radius == -initial_radius
    assert geom.k == initial_conic
    assert be.allclose(geom.coefficients, be.array(initial_coeffs))


def test_flip_even_asphere_geometry():
    initial_radius = 50.0
    initial_conic = 0.0
    initial_coeffs = [0.001, -0.0005]  # C2, C4
    geom = EvenAsphere(
        cs, radius=initial_radius, conic=initial_conic, coefficients=initial_coeffs
    )

    assert geom.radius == initial_radius
    assert geom.k == initial_conic
    assert geom.coefficients == initial_coeffs

    geom.flip()

    assert geom.radius == -initial_radius
    assert geom.k == initial_conic
    assert geom.coefficients == initial_coeffs


def test_flip_odd_asphere_geometry():
    initial_radius = 75.0
    initial_conic = -1.0
    # Coefficients for OddAsphere are C_i * r^(i+1)
    # e.g., coefficients[0] is for C_1 * r^1, coefficients[1] for C_2 * r^2
    initial_coeffs = [0.01, 0.002]
    geom = OddAsphere(
        cs, radius=initial_radius, conic=initial_conic, coefficients=initial_coeffs
    )

    assert geom.radius == initial_radius
    assert geom.k == initial_conic
    assert geom.coefficients == initial_coeffs

    geom.flip()

    assert geom.radius == -initial_radius
    assert geom.k == initial_conic
    assert geom.coefficients == [-c for c in initial_coeffs]


def test_flip_polynomial_geometry():
    initial_radius = 120.0
    initial_conic = 0.5
    # Ensure coefficients are structured as expected by PolynomialGeometry (list of lists or 2D array)
    initial_coeffs_list = [[0, 0.01, 0.02], [0.005, 0, 0]]
    geom = PolynomialGeometry(
        cs, radius=initial_radius, conic=initial_conic, coefficients=initial_coeffs_list
    )

    assert geom.radius == initial_radius
    assert geom.k == initial_conic
    # PolynomialGeometry might convert to be.array internally, ensure comparison is robust
    assert be.allclose(geom.coefficients, be.array(initial_coeffs_list))

    geom.flip()

    assert geom.radius == -initial_radius
    assert geom.k == initial_conic
    assert be.allclose(geom.coefficients, be.array(initial_coeffs_list))


def test_flip_toroidal_geometry():
    initial_r_rot = 200.0
    initial_r_yz = -100.0
    initial_k_yz = -0.7
    initial_coeffs_y = [0.0001, -0.00002]
    geom = ToroidalGeometry(
        cs,
        radius_x=initial_r_rot,
        radius_y=initial_r_yz,
        conic=initial_k_yz,
        coeffs_poly_y=initial_coeffs_y,
    )

    initial_c_yz_val = geom.c_yz
    # NewtonRaphsonGeometry's radius is set to radius_rotation in Toroidal __init__
    initial_base_radius = geom.radius

    assert geom.R_rot == initial_r_rot
    assert geom.R_yz == initial_r_yz
    assert geom.k_yz == initial_k_yz
    assert be.allclose(geom.coeffs_poly_y, be.array(initial_coeffs_y))
    assert geom.radius == initial_base_radius

    geom.flip()

    assert geom.R_rot == -initial_r_rot
    assert geom.R_yz == -initial_r_yz
    assert geom.k_yz == initial_k_yz
    assert be.allclose(geom.coeffs_poly_y, be.array(initial_coeffs_y))
    # Handle comparison for c_yz which might be float or array
    expected_c_yz = (
        -initial_c_yz_val if initial_r_yz != 0 and be.isfinite(initial_r_yz) else 0.0
    )
    if hasattr(geom.c_yz, "shape") or hasattr(expected_c_yz, "shape"):
        assert be.allclose(geom.c_yz, be.array(expected_c_yz))
    else:
        assert be.allclose(geom.c_yz, expected_c_yz)
    assert geom.radius == -initial_base_radius  # Check base radius also flipped


def test_flip_zernike_geometry():
    initial_radius = 300.0
    initial_conic = -0.2
    initial_coeffs = [0, 0, 0.05, -0.03]  # Z3, Z4 (indices 2, 3 in a 0-indexed list)
    geom = ZernikePolynomialGeometry(
        cs,
        radius=initial_radius,
        conic=initial_conic,
        coefficients=initial_coeffs,
        norm_radius=10.0,
    )

    assert geom.radius == initial_radius
    assert geom.k == initial_conic
    assert be.allclose(geom.coefficients, be.array(initial_coeffs))

    geom.flip()

    assert geom.radius == -initial_radius
    assert geom.k == initial_conic
    assert be.allclose(geom.coefficients, be.array(initial_coeffs))


def test_flip_biconic_zero_radius():
    geom = BiconicGeometry(cs, radius_x=0.0, radius_y=-20.0, conic_x=0.1, conic_y=-0.2)
    initial_cx_val = geom.cx  # Should be 0.0
    initial_cy_val = geom.cy
    geom.flip()
    assert geom.Rx == 0.0  # -0.0 is 0.0
    assert geom.Ry == 20.0
    assert be.allclose(geom.cx, initial_cx_val)  # cx should remain 0.0
    assert be.allclose(geom.cy, -initial_cy_val)


def test_flip_toroidal_zero_radius():
    geom = ToroidalGeometry(cs, radius_x=0.0, radius_y=-100.0, conic=-0.7)
    initial_c_yz_val = geom.c_yz
    geom.flip()
    assert geom.R_rot == 0.0  # -0.0 is 0.0
    assert geom.R_yz == 100.0
    expected_c_yz = -initial_c_yz_val if -100.0 != 0 and be.isfinite(-100.0) else 0.0
    if hasattr(geom.c_yz, "shape") or hasattr(expected_c_yz, "shape"):
        assert be.allclose(geom.c_yz, be.array(expected_c_yz))
    else:
        assert be.allclose(geom.c_yz, expected_c_yz)
    assert geom.R_rot == 0.0  # Rotation radius also flipped (was 0.0, remains 0.0)


def test_flip_standard_inf_radius():
    geom = StandardGeometry(cs, radius=be.inf, conic=0.0)
    geom.flip()
    assert geom.radius == -be.inf


def test_flip_biconic_inf_radius():
    geom = BiconicGeometry(
        cs, radius_x=be.inf, radius_y=-20.0, conic_x=0.1, conic_y=-0.2
    )
    initial_cx_val = geom.cx  # Should be 0.0
    initial_cy_val = geom.cy
    geom.flip()
    assert geom.Rx == -be.inf
    assert geom.Ry == 20.0
    assert be.allclose(geom.cx, initial_cx_val)  # cx should remain 0.0 as 1/inf is 0
    assert be.allclose(geom.cy, -initial_cy_val)


def test_flip_toroidal_inf_radius():
    geom = ToroidalGeometry(cs, radius_x=be.inf, radius_y=-100.0, conic=-0.7)
    initial_c_yz_val = geom.c_yz
    geom.flip()
    assert geom.R_rot == -be.inf
    assert geom.R_yz == 100.0
    expected_c_yz = -initial_c_yz_val if -100.0 != 0 and be.isfinite(-100.0) else 0.0
    if hasattr(geom.c_yz, "shape") or hasattr(expected_c_yz, "shape"):
        assert be.allclose(geom.c_yz, be.array(expected_c_yz))
    else:
        assert be.allclose(geom.c_yz, expected_c_yz)
    assert geom.R_rot == -be.inf
