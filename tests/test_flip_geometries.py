from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    BiconicGeometry,
    ChebyshevPolynomialGeometry,
    EvenAsphere,
    GridSagGeometry,
    OddAsphere,
    Plane,
    PolynomialGeometry,
    StandardGeometry,
    ToroidalGeometry,
    ZernikePolynomialGeometry,
)
from optiland.optic import Optic
from tests.utils import assert_allclose

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
    assert be.allclose(geom.coefficients, -be.array(initial_coeffs))


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
    assert geom.coefficients == [-c for c in initial_coeffs]


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
    assert be.allclose(geom.coefficients, -be.array(initial_coeffs_list))


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
    assert be.allclose(geom.coeffs_poly_y, -be.array(initial_coeffs_y))
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
    assert be.allclose(geom.coefficients, -be.array(initial_coeffs))


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


# --- sag'(x, y) == -sag(x, y) for each geometry in FLIP_CASES (#1103) ---


def _grid_sag_geometry():
    axis = np.linspace(-12.0, 12.0, 25)
    xx, yy = np.meshgrid(axis, axis)
    sag = 1e-3 * xx**2 + 2e-3 * yy**2 + 1e-4 * xx * yy
    return GridSagGeometry(CoordinateSystem(), list(axis), list(axis), sag)


# Factories run inside each test, after set_test_backend has chosen the backend.
FLIP_CASES = {
    "standard": lambda: StandardGeometry(CoordinateSystem(), radius=50.0, conic=-0.5),
    "even_asphere": lambda: EvenAsphere(
        CoordinateSystem(), radius=50.0, conic=-0.5, coefficients=[1e-5, 1e-4, -2e-7]
    ),
    "odd_asphere": lambda: OddAsphere(
        CoordinateSystem(), radius=50.0, conic=-0.5, coefficients=[1e-3, 1e-4, -2e-7]
    ),
    "polynomial": lambda: PolynomialGeometry(
        CoordinateSystem(),
        radius=120.0,
        conic=0.5,
        coefficients=[[1e-4, 2e-4, 1e-3], [5e-4, -3e-4, 2e-4]],
    ),
    "chebyshev": lambda: ChebyshevPolynomialGeometry(
        CoordinateSystem(),
        radius=100.0,
        conic=-0.8,
        coefficients=[[1e-3, 0.01], [0.02, 0.03]],
        norm_x=10.0,
        norm_y=10.0,
    ),
    "zernike": lambda: ZernikePolynomialGeometry(
        CoordinateSystem(),
        radius=300.0,
        conic=-0.2,
        coefficients=[1e-3, 2e-3, 0.05, -0.03, 4e-3, -2e-3],
        norm_radius=10.0,
    ),
    "toroidal_poly_y": lambda: ToroidalGeometry(
        CoordinateSystem(),
        radius_x=80.0,
        radius_y=40.0,
        conic=-0.3,
        coeffs_poly_y=[1e-4, -2e-7],
    ),
    "toroidal_no_poly": lambda: ToroidalGeometry(
        CoordinateSystem(), radius_x=80.0, radius_y=40.0, conic=-0.3
    ),
    "biconic": lambda: BiconicGeometry(
        CoordinateSystem(), radius_x=40.0, radius_y=60.0, conic_x=-0.3, conic_y=0.2
    ),
    "grid_sag": _grid_sag_geometry,
}

# Off-axis points (max radius ~5.4 mm) so non-symmetric terms contribute.
_X = [0.0, 3.0, 0.0, 4.0, -5.0, 2.5]
_Y = [0.0, 0.0, 5.0, 3.0, 2.0, -4.0]


@pytest.mark.parametrize("case", list(FLIP_CASES))
def test_flip_negates_sag(set_test_backend, case):
    """After flip, sag'(x, y) == -sag(x, y): the surface is mirrored in z."""
    geom = FLIP_CASES[case]()
    x, y = be.array(_X), be.array(_Y)
    before = be.to_numpy(geom.sag(x, y))
    assert np.max(np.abs(before)) > 1e-3  # guard against a vacuous comparison
    geom.flip()
    assert_allclose(geom.sag(x, y), -before)


@pytest.mark.parametrize("case", list(FLIP_CASES))
def test_double_flip_restores_sag(set_test_backend, case):
    """Flipping twice gives back the original surface."""
    geom = FLIP_CASES[case]()
    x, y = be.array(_X), be.array(_Y)
    before = be.to_numpy(geom.sag(x, y))
    geom.flip()
    geom.flip()
    assert_allclose(geom.sag(x, y), before)


@pytest.mark.parametrize(
    "case", ["even_asphere", "polynomial", "chebyshev", "zernike", "toroidal_poly_y"]
)
def test_flip_with_shared_coefficients(set_test_backend, case):
    """Geometries sharing one coefficient object (as a pickup does) both flip."""
    attr = "coeffs_poly_y" if case.startswith("toroidal") else "coefficients"
    source, target = FLIP_CASES[case](), FLIP_CASES[case]()
    setattr(target, attr, getattr(source, attr))
    x, y = be.array(_X), be.array(_Y)
    before = be.to_numpy(target.sag(x, y))
    source.flip()
    target.flip()
    assert_allclose(target.sag(x, y), -before)


@pytest.mark.parametrize("case", ["polynomial", "chebyshev", "toroidal_poly_y"])
def test_flip_with_plain_list_terms(set_test_backend, case):
    """Terms re-assigned as plain Python lists after construction still flip."""
    geom = FLIP_CASES[case]()
    x, y = be.array(_X), be.array(_Y)
    before = be.to_numpy(geom.sag(x, y))
    attr = "coeffs_poly_y" if case.startswith("toroidal") else "coefficients"
    setattr(geom, attr, be.to_numpy(getattr(geom, attr)).tolist())
    geom.flip()
    assert_allclose(geom.sag(x, y), -before)


def test_optic_flip_mirrors_aspheric_surface(set_test_backend):
    """The old back surface becomes the new front surface with sag negated."""
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf)
    lens.surfaces.add(
        index=1, thickness=5.0, radius=40.0, material="N-BK7", is_stop=True
    )
    lens.surfaces.add(
        index=2,
        thickness=50.0,
        radius=-60.0,
        conic=-1.2,
        surface_type="even_asphere",
        coefficients=[0.0, 2e-5, -3e-8],
    )
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=20.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    x, y = be.zeros(3), be.array([0.0, 4.0, 8.0])
    old_back_sag = lens.surfaces[2].geometry.sag(x, y)

    lens.updater.flip()

    assert_allclose(lens.surfaces[1].geometry.sag(x, y), -old_back_sag)
