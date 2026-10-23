from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.rays import RealRays

from .utils import assert_allclose


def test_coordinate_system_init(set_test_backend):
    # Test case 1: Initialize with default values
    cs = CoordinateSystem()
    assert cs.x == 0
    assert cs.y == 0
    assert cs.z == 0
    assert cs.rx == 0
    assert cs.ry == 0
    assert cs.rz == 0
    assert cs.reference_cs is None

    # Test case 2: Initialize with custom values
    ref_cs = CoordinateSystem(1, 2, 3, 0.1, 0.2, 0.3)
    cs = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, ref_cs)
    assert cs.x == 10
    assert cs.y == 20
    assert cs.z == 30
    assert cs.rx == 0.5
    assert cs.ry == 0.6
    assert cs.rz == 0.7
    assert cs.reference_cs == ref_cs


def test_coordinate_system_localize(set_test_backend):
    # Test case 1: Localize rays without reference coordinate system
    cs = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    rays = RealRays(1, 2, 5, 0, 0, 1, 1, 1)
    cs.localize(rays)
    assert rays.x == 0.0
    assert rays.y == 3.0
    assert rays.z == 3.0
    assert rays.L == 0.0
    assert rays.M == 0.0
    assert rays.N == 1.0

    # Test case 2: Localize rays with reference coordinate system
    ref_cs = CoordinateSystem(5, 5, 5, 0.2, 0.3, 0.4)
    cs = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, ref_cs)
    rays = RealRays(1, 2, 3, 0.1, 0.2, 0.3, 1, 1)
    cs.localize(rays)
    assert_allclose(rays.x, -1.826215)
    assert_allclose(rays.y, -26.51361)
    assert_allclose(rays.z, -32.55361)
    assert_allclose(rays.L, -0.0122750)
    assert_allclose(rays.M, 0.2697627)
    assert_allclose(rays.N, 0.2589931)


def test_coordinate_system_globalize(set_test_backend):
    # Test case 1: Globalize rays without reference coordinate system
    cs = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    rays = RealRays(0.0, 3.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    cs.globalize(rays)
    assert_allclose(rays.x, 1.0)
    assert_allclose(rays.y, 2.0)
    assert_allclose(rays.z, 5.0)
    assert_allclose(rays.L, 0.0)
    assert_allclose(rays.M, 0.0)
    assert_allclose(rays.N, 1.0)

    # Test case 2: Globalize rays with reference coordinate system
    ref_cs = CoordinateSystem(5, 5, 5, 0.2, 0.3, 0.4)
    cs = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, ref_cs)
    rays = RealRays(
        -23.63610642,
        -25.40225528,
        -23.08369058,
        0.23129557,
        0.2370124,
        0.17414787,
        1,
        1,
    )
    cs.globalize(rays)
    assert_allclose(rays.x, 4.654692)
    assert_allclose(rays.y, -12.50279)
    assert_allclose(rays.z, 21.51751)
    assert_allclose(rays.L, 0.0866579)
    assert_allclose(rays.M, 0.3526931)
    assert_allclose(rays.N, 0.08998877)


def test_coordinate_system_transform(set_test_backend):
    cs1 = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    cs2 = CoordinateSystem(10, 20, 30, 0.5, 0.6, 0.7, cs1)

    eff_translation, eff_rot_mat = cs2.get_effective_transform()
    assert be.allclose(eff_translation, be.array([11, 19, 32]))
    rot_mat = be.array(
        [
            [0.6312515, -0.35830835, 0.68784931],
            [0.5316958, 0.84560449, -0.04746188],
            [-0.56464247, 0.39568697, 0.72430014],
        ],
    )
    assert be.allclose(eff_rot_mat, rot_mat)


def test_coordinate_system_to_dict(set_test_backend):
    cs = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    cs_dict = cs.to_dict()
    assert cs_dict["x"] == 1
    assert cs_dict["y"] == -1.0
    assert cs_dict["z"] == 2.0
    assert cs_dict["rx"] == 0.0
    assert cs_dict["ry"] == 0.0
    assert cs_dict["rz"] == 0.0
    assert cs_dict["reference_cs"] is None


def test_coordinate_system_raw_assignment_to_dict(set_test_backend):
    # Regression test for issue #624: assigning a plain Python float to a
    # coordinate attribute after construction must not break to_dict().
    cs = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    cs.z = 10.0
    cs_dict = cs.to_dict()
    assert cs_dict["z"] == 10.0


def test_coordinate_system_assignment_preserves_array_invariant(set_test_backend):
    # Coordinates must remain backend arrays regardless of what is assigned.
    cs = CoordinateSystem()
    cs.x = 1.0
    cs.y = -2
    cs.z = 3.5
    cs.rx = 0.1
    cs.ry = 0.2
    cs.rz = 0.3
    for value in (cs.x, cs.y, cs.z, cs.rx, cs.ry, cs.rz):
        assert be.is_array_like(value)


def test_coordinate_system_raw_assignment_effective_transform(set_test_backend):
    # get_effective_transform() relies on .item(); a raw-float assignment
    # must not break it (same invariant as issue #624).
    cs = CoordinateSystem(1, -1.0, 2.0, 0.0, 0.0, 0.0)
    cs.z = 5.0
    translation, _ = cs.get_effective_transform()
    assert_allclose(translation, be.array([1.0, -1.0, 5.0]))


def test_coordinate_system_setter_autodiff_matches_fd(set_test_backend):
    # The setter routes assignment through be.array(); for an existing torch
    # tensor this must be identity-preserving so autograd still flows. Verify
    # the analytic gradient matches a central finite difference.
    if be.get_backend() != "torch":
        pytest.skip("autodiff vs. finite-difference check is torch-only")

    import torch

    def localized_y(rx_value):
        cs = CoordinateSystem(rx=rx_value)
        rays = RealRays(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        cs.localize(rays)
        return rays.y

    rx0 = 0.3
    rx_grad = torch.tensor(rx0, dtype=torch.float64, requires_grad=True)
    y = localized_y(rx_grad)
    y.backward()
    autodiff_grad = rx_grad.grad.item()

    eps = 1e-6
    with torch.no_grad():
        y_plus = localized_y(torch.tensor(rx0 + eps, dtype=torch.float64))
        y_minus = localized_y(torch.tensor(rx0 - eps, dtype=torch.float64))
    fd_grad = ((y_plus - y_minus) / (2 * eps)).item()

    assert abs(autodiff_grad - fd_grad) < 1e-6


@pytest.mark.parametrize("method", ["localize", "globalize"])
def test_coordinate_system_array_rotations_match_scalar(set_test_backend, method):
    # Array-valued rotations apply elementwise, one angle per ray, including
    # rays whose angle is exactly zero.
    rx = [0.0, 0.1, 0.0]
    ry = [0.2, 0.0, 0.0]
    rz = [0.3, 0.0, -0.4]
    x, y, z = [1.0, -2.0, 0.5], [2.0, 0.5, -1.0], [3.0, 1.0, 4.0]
    L, M, N = [0.0, 0.6, 0.0], [0.0, 0.0, 0.8], [1.0, 0.8, 0.6]

    cs = CoordinateSystem(1.0, -1.0, 2.0, be.array(rx), be.array(ry), be.array(rz))
    rays = RealRays(x, y, z, L, M, N, 1.0, 1.0)
    getattr(cs, method)(rays)

    for i in range(len(rx)):
        cs_i = CoordinateSystem(1.0, -1.0, 2.0, rx[i], ry[i], rz[i])
        ray_i = RealRays(x[i], y[i], z[i], L[i], M[i], N[i], 1.0, 1.0)
        getattr(cs_i, method)(ray_i)
        for attr in ("x", "y", "z", "L", "M", "N"):
            assert_allclose(getattr(rays, attr)[i], getattr(ray_i, attr))


@pytest.mark.parametrize("axis", ["rx", "ry", "rz"])
@pytest.mark.parametrize("method", ["localize", "globalize"])
def test_coordinate_system_zero_rotation_gradient(set_test_backend, axis, method):
    # A tracked rotation must reach the autograd graph even when its value is
    # exactly zero, and its gradient must match a central finite difference.
    if be.get_backend() != "torch":
        pytest.skip("autodiff vs. finite-difference check is torch-only")

    import torch

    def transformed_sum(angle):
        cs = CoordinateSystem(**{axis: angle})
        rays = RealRays(1.0, 2.0, 3.0, 0.6, 0.0, 0.8, 1.0, 1.0)
        getattr(cs, method)(rays)
        return rays.x + rays.y + rays.z + rays.L + rays.M + rays.N

    angle = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    transformed_sum(angle).sum().backward()
    assert angle.grad is not None

    eps = 1e-6
    with torch.no_grad():
        f_plus = transformed_sum(torch.tensor(eps, dtype=torch.float64))
        f_minus = transformed_sum(torch.tensor(-eps, dtype=torch.float64))
    fd_grad = ((f_plus - f_minus) / (2 * eps)).item()

    assert fd_grad != 0.0
    assert abs(angle.grad.item() - fd_grad) < 1e-6


def test_untilted_mirror_tilt_gradient(set_test_backend):
    # Tilting a mirror by a small angle deflects the reflected ray by twice
    # that angle, so d(M)/d(rx) = 2 at rx = 0. The gradient must be available
    # when the tilt starts at exactly zero.
    if be.get_backend() != "torch":
        pytest.skip("gradient check is torch-only")

    import torch

    from optiland.optic import Optic

    rx = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optic = Optic()
    optic.surfaces.add(index=0, z=-100)
    optic.surfaces.add(index=1, z=10, rx=rx, material="mirror")
    optic.surfaces.add(index=2, z=-50)

    rays = RealRays(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.55)
    optic.surfaces.trace(rays, skip=1)
    rays.M.sum().backward()

    assert rx.grad is not None
    assert_allclose(rx.grad, 2.0)


def test_coordinate_system_from_dict(set_test_backend):
    cs_dict = {"x": 1, "y": -1, "z": 2, "rx": 0, "ry": 0, "rz": 0, "reference_cs": None}
    cs = CoordinateSystem.from_dict(cs_dict)
    assert cs.x == 1
    assert cs.y == -1.0
    assert cs.z == 2.0
    assert cs.rx == 0.0
    assert cs.ry == 0.0
    assert cs.rz == 0.0
    assert cs.reference_cs is None
