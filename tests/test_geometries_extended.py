from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.forbes import jacobi
from optiland.geometries.nurbs import nurbs_basis_functions
from optiland.geometries.plane_grating import PlaneGrating
from optiland.geometries.standard_grating import StandardGratingGeometry
from optiland.rays import RealRays


def test_standard_grating_init(set_test_backend):
    cs = CoordinateSystem()
    g = StandardGratingGeometry(
        cs,
        radius=100.0,
        grating_order=1,
        grating_period=1.0,
        groove_orientation_angle=0.0,
        conic=0.0,
    )
    assert g.radius == 100.0
    assert g.grating_order == 1
    assert g.grating_period == 1.0


def test_standard_grating_sag(set_test_backend):
    cs = CoordinateSystem()
    g = StandardGratingGeometry(
        cs,
        radius=100.0,
        grating_order=1,
        grating_period=1.0,
        groove_orientation_angle=0.0,
    )
    assert g.sag(0, 0) == 0.0
    s = g.sag(10, 0)
    assert abs(s - 0.5) < 0.1


def test_standard_grating_serialization(set_test_backend):
    cs = CoordinateSystem()
    g = StandardGratingGeometry(
        cs,
        radius=100.0,
        grating_order=1,
        grating_period=2.0,
        groove_orientation_angle=0.5,
        conic=-1.0,
    )
    d = g.to_dict()

    g2 = StandardGratingGeometry.from_dict(d)
    assert g2.radius == 100.0
    assert g2.grating_order == 1
    assert g2.grating_period == 2.0
    assert g2.groove_orientation_angle == 0.5
    assert g2.k == -1.0


def test_plane_grating_init(set_test_backend):
    cs = CoordinateSystem()
    g = PlaneGrating(
        cs, grating_order=1, grating_period=1.0, groove_orientation_angle=0.0
    )
    assert g.grating_order == 1


def test_plane_grating_sag(set_test_backend):
    cs = CoordinateSystem()
    g = PlaneGrating(
        cs, grating_order=1, grating_period=1.0, groove_orientation_angle=0.0
    )
    assert g.sag(10, 10) == 0.0


def test_plane_grating_serialization(set_test_backend):
    cs = CoordinateSystem()
    g = PlaneGrating(
        cs, grating_order=1, grating_period=2.0, groove_orientation_angle=0.5
    )
    d = g.to_dict()

    g2 = PlaneGrating.from_dict(d)
    assert g2.grating_order == 1
    assert g2.grating_period == 2.0
    assert g2.groove_orientation_angle == 0.5


def test_standard_grating_distance(set_test_backend):
    cs = CoordinateSystem()
    g = StandardGratingGeometry(
        cs,
        radius=100.0,
        grating_order=1,
        grating_period=1.0,
        groove_orientation_angle=0.0,
    )

    rays = RealRays(
        x=be.array([0.0]),
        y=be.array([0.0]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        wavelength=0.55,
        intensity=1.0,
    )
    dist = g.distance(rays)
    val = dist[0]
    if hasattr(val, "item"):
        val = val.item()
    assert abs(val) < 1e-6


def test_standard_grating_vectors(set_test_backend):
    cs = CoordinateSystem()
    g = StandardGratingGeometry(
        cs,
        radius=100.0,
        grating_order=1,
        grating_period=1.0,
        groove_orientation_angle=0.0,
    )

    rays = RealRays(
        x=be.array([0.0]),
        y=be.array([0.0]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        wavelength=0.55,
        intensity=1.0,
    )

    nx, ny, nz = g.surface_normal(rays)
    val_nx = nx[0].item() if hasattr(nx[0], "item") else nx[0]
    val_nz = nz[0].item() if hasattr(nz[0], "item") else nz[0]
    assert val_nx == 0
    assert abs(val_nz + 1) < 1e-6

    gx, gy, gz = g.grating_vector(rays)
    val_gy = gy[0].item() if hasattr(gy[0], "item") else gy[0]
    # For angle 0, tangent is along Z? No.
    # Check _tangent: tx=1, ty=0, tz=dzdx.
    # Grating vector should be perpendicular to grooves.
    assert abs(val_gy) > 0.0


def test_plane_grating_vectors(set_test_backend):
    cs = CoordinateSystem()
    g = PlaneGrating(
        cs, grating_order=1, grating_period=1.0, groove_orientation_angle=0.0
    )

    rays = RealRays(
        x=be.array([0.0]),
        y=be.array([0.0]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        wavelength=0.55,
        intensity=1.0,
    )

    gx, gy, gz = g.grating_vector(rays)
    val_gy = gy[0].item() if hasattr(gy[0], "item") else gy[0]
    assert abs(val_gy - 1.0) < 1e-6  # Cos(0) = 1


def test_jacobi(set_test_backend):
    x = be.linspace(-1, 1, 10)
    # n=0 -> 1
    j0 = jacobi.jacobi(0, 0, 0, x)
    assert be.all(j0 == 1)

    # n=1, alpha=0, beta=0 -> x
    # jacobi(1, 0, 0, x) -> 1 + 2 * (x-1)/2 = 1 + x - 1 = x
    j1 = jacobi.jacobi(1, 0, 0, x)
    assert be.allclose(j1, x, atol=1e-5)


def test_jacobi_sum_clenshaw(set_test_backend):
    x = be.linspace(-1, 1, 10)
    s = [1.0, 1.0]  # 1*P0 + 1*P1 = 1 + x
    res = jacobi.jacobi_sum_clenshaw(s, 0, 0, x)
    # The result of the sum is in res[0]
    assert be.allclose(res[0], 1.0 + x, atol=1e-5)


def test_nurbs_basis(set_test_backend):
    # n=2 (3 basis funcs), p=1 (degree 1 -> linear), U=[0, 0, 0.5, 1, 1] (clamped)
    # n=1 (2 funcs). r = n+p+2.
    # Let's use simple case. p=1. U=[0, 0, 1, 1].
    # n = r - p - 2 = 4 - 1 - 2 = 1. So 2 basis funcs: N0,1 and N1,1.
    n = 1
    p = 1
    U = np.array([0.0, 0.0, 1.0, 1.0])
    u = np.array([0.0, 0.5, 1.0])

    N = nurbs_basis_functions.compute_basis_polynomials(n, p, U, u)
    # N shape: (n+1, Nu) -> (2, 3)
    # At u=0: N0=1, N1=0
    # At u=0.5: N0=0.5, N1=0.5 (linear)
    # At u=1: N0=0, N1=1

    assert N.shape == (2, 3)
    assert abs(N[0, 0] - 1.0) < 1e-6
    assert abs(N[1, 2] - 1.0) < 1e-6


def test_nurbs_basis_derivatives(set_test_backend):
    n = 1
    p = 1
    U = np.array([0.0, 0.0, 1.0, 1.0])
    u = np.array([0.5])
    # Deriv of linear N0(u)=1-u is -1. N1(u)=u is 1.

    ders = nurbs_basis_functions.compute_basis_polynomials_derivatives(n, p, U, u, 1)
    # ders shape: (n+1, Nu)
    assert abs(ders[0, 0] + 1.0) < 1e-6
    assert abs(ders[1, 0] - 1.0) < 1e-6
