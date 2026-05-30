from __future__ import annotations

import numpy as np

from optiland import optic
from optiland.geometries import (
    BiconicGeometry,
    EvenAsphere,
    ForbesQ2dGeometry,
    ForbesQNormalSlopeGeometry,
    NurbsGeometry,
    Plane,
    PlaneGrating,
    StandardGeometry,
    StandardGratingGeometry,
)


def test_scale_standard():
    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=10, thickness=5, material="Air")
    lens.surfaces.add(index=1)  # Add a second surface so first one is not "last"
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    assert isinstance(surface.geometry, StandardGeometry)
    assert surface.geometry.radius == 20
    assert surface.thickness == 10


def test_scale_even_asphere():
    lens = optic.Optic()
    # C1 (r^4) -> index 0. C2 (r^6) -> index 1.
    coeffs = [1e-3, 1e-5]
    lens.surfaces.add(
        index=0,
        radius=10,
        thickness=2,
        surface_type="even_asphere",
        coefficients=coeffs,
    )
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    assert isinstance(surface.geometry, EvenAsphere)
    assert surface.geometry.radius == 20

    s = 2.0
    expected_c0 = coeffs[0] * s ** (1 - 2 * (0 + 1))
    expected_c1 = coeffs[1] * s ** (1 - 2 * (1 + 1))

    assert np.isclose(surface.geometry.coefficients[0], expected_c0)
    assert np.isclose(surface.geometry.coefficients[1], expected_c1)


def test_scale_odd_asphere():
    lens = optic.Optic()
    # odd asphere: z = ... + sum(Ci * r^(i+1))
    # i=0 -> r^1.
    coeffs = [1e-2, 1e-4]
    lens.surfaces.add(
        index=0, radius=10, surface_type="odd_asphere", coefficients=coeffs
    )
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    s = 2.0
    expected_c0 = coeffs[0] * s ** (1 - (0 + 1))
    expected_c1 = coeffs[1] * s ** (1 - (1 + 1))

    assert np.isclose(surface.geometry.coefficients[0], expected_c0)
    assert np.isclose(surface.geometry.coefficients[1], expected_c1)


def test_scale_polynomial():
    lens = optic.Optic()
    # Cij * x^i * y^j
    # i=1, j=1 -> xy. s^(1-(1+1)) = s^-1.
    coeffs = [[0, 0], [0, 1e-3]]  # C11
    lens.surfaces.add(
        index=0, radius=10, surface_type="polynomial", coefficients=coeffs
    )
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    s = 2.0
    # coeffs[1][1] is x^1 y^1
    expected = 1e-3 * s ** (1 - (1 + 1))
    assert np.isclose(surface.geometry.coefficients[1][1], expected)


def test_scale_chebyshev():
    lens = optic.Optic()
    coeffs = [[0, 0], [0, 1e-3]]
    lens.surfaces.add(
        index=0,
        radius=10,
        surface_type="chebyshev",
        coefficients=coeffs,
        norm_x=1.0,
        norm_y=1.0,
    )
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    s = 2.0

    assert surface.geometry.norm_x == 2.0
    assert surface.geometry.norm_y == 2.0
    # Chebyshev coeffs scale linearly with s if norms are scaled
    assert surface.geometry.coefficients[1][1] == 1e-3 * s


def test_scale_zernike():
    lens = optic.Optic()
    coeffs = [0, 1e-3, 1e-4]
    lens.surfaces.add(
        index=0, radius=10, surface_type="zernike", coefficients=coeffs, norm_radius=1.0
    )
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    s = 2.0

    assert surface.geometry.norm_radius == 2.0
    assert np.allclose(surface.geometry.coefficients, np.array(coeffs) * s)


def test_scale_biconic():
    lens = optic.Optic()
    lens.surfaces.add(
        index=0, radius_x=10, surface_type="biconic", radius_y=10, conic_x=0, conic_y=0
    )  # using defaults
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    assert isinstance(surface.geometry, BiconicGeometry)
    assert surface.geometry.Rx == 20
    assert surface.geometry.Ry == 20


def test_scale_toroidal():
    lens = optic.Optic()
    coeffs = [1e-3]
    lens.surfaces.add(
        index=0,
        radius_x=10,
        surface_type="toroidal",
        radius_y=5,
        toroidal_coeffs_poly_y=coeffs,
    )
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    s = 2.0

    assert surface.geometry.R_rot == 20
    assert surface.geometry.R_yz == 10

    # Coeffs scaling: y^(2(i+1)). i=0 -> y^2.
    # s^(1 - 2) = s^-1.
    expected = coeffs[0] * s ** (1 - 2 * (1))
    assert np.isclose(surface.geometry.coeffs_poly_y[0], expected)


def test_scale_plane():
    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=5)
    lens.surfaces.add(index=1)
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    assert isinstance(surface.geometry, Plane)
    assert np.isinf(surface.geometry.radius)
    assert surface.thickness == 10


def test_scale_plane_grating():
    lens = optic.Optic()
    # PlaneGrating
    lens.surfaces.add(
        index=0,
        radius=np.inf,
        surface_type="grating",
        grating_order=1,
        grating_period=1.0,
        groove_orientation_angle=0.0,
    )
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    assert isinstance(surface.geometry, PlaneGrating)
    assert surface.geometry.grating_period == 2.0


def test_scale_standard_grating():
    lens = optic.Optic()
    lens.surfaces.add(
        index=0,
        radius=10,
        surface_type="grating",
        grating_order=1,
        grating_period=1.0,
        groove_orientation_angle=0.0,
    )
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    assert isinstance(surface.geometry, StandardGratingGeometry)
    assert surface.geometry.radius == 20
    assert surface.geometry.grating_period == 2.0


def test_scale_forbes_qbfs():
    lens = optic.Optic()
    # ForbesQbfs
    # terms: {m: val}
    terms = {0: 1e-3, 1: 1e-4}
    # Save original values before scaling (terms dict may be mutated)
    original_terms = {k: v for k, v in terms.items()}
    lens.surfaces.add(
        index=0,
        radius=10,
        surface_type="forbes_qbfs",
        norm_radius=1.0,
        radial_terms=terms,
    )
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    assert isinstance(surface.geometry, ForbesQNormalSlopeGeometry)
    assert surface.geometry.radius == 20
    assert surface.geometry.norm_radius == 2.0

    # Coefficients scale by s (linear with sag)
    s = 2.0
    assert np.allclose(surface.geometry.radial_terms[0], original_terms[0] * s)
    assert np.allclose(surface.geometry.radial_terms[1], original_terms[1] * s)


def test_scale_forbes_q2d():
    lens = optic.Optic()
    # ForbesQ2d
    # terms: {('a', m, n): val}
    terms = {("a", 0, 0): 1e-3}
    # Save original values before scaling (terms dict may be mutated)
    original_terms = {k: v for k, v in terms.items()}
    lens.surfaces.add(
        index=0,
        radius=10,
        surface_type="forbes_q2d",
        norm_radius=1.0,
        freeform_coeffs=terms,
    )
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    assert isinstance(surface.geometry, ForbesQ2dGeometry)
    assert surface.geometry.radius == 20
    assert surface.geometry.norm_radius == 2.0

    s = 2.0
    assert np.allclose(
        surface.geometry.freeform_coeffs[("a", 0, 0)], original_terms[("a", 0, 0)] * s
    )


def test_scale_nurbs():
    lens = optic.Optic()
    # Nurbs
    # Simple nurbs with control points.
    # P shape (ndim, n+1, m+1)
    P = np.zeros((3, 5, 5))
    P[0, :, :] = 1.0  # x=1
    P[1, :, :] = 2.0  # y=2
    P[2, :, :] = 3.0  # z=3

    lens.surfaces.add(
        index=0,
        radius=np.inf,
        surface_type="nurbs",
        control_points=P,
        nurbs_norm_x=1.0,
        nurbs_norm_y=1.0,
        nurbs_x_center=0.5,
        nurbs_y_center=0.5,
    )
    lens.updater.scale_system(2.0)

    surface = lens.surfaces[0]
    assert isinstance(surface.geometry, NurbsGeometry)

    s = 2.0
    expected_P = P * s
    assert np.allclose(surface.geometry.P, expected_P)
    assert surface.geometry.nurbs_norm_x == 1.0 * s
    assert surface.geometry.nurbs_norm_y == 1.0 * s
    assert surface.geometry.x_center == 0.5 * s
    assert surface.geometry.y_center == 0.5 * s
