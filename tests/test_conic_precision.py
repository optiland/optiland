"""Precision and differentiability contracts for conic root selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import StandardGeometry
from optiland.rays import RealRays


@pytest.fixture(params=["numpy", "torch-cpu", "torch-cuda"])
def conic_backend(request):
    """Exercise real dtypes and devices without changing global test fixtures."""
    name = request.param
    if name.startswith("torch"):
        torch = pytest.importorskip("torch")
        if name == "torch-cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA-enabled Torch and a GPU are required")
        be.set_backend("torch")
        be.set_device(name.split("-")[1])
        be.grad_mode.disable()
    else:
        be.set_backend("numpy")
    yield name
    if name.startswith("torch"):
        be.set_device("cpu")
        be.grad_mode.disable()
    be.set_precision("float64")
    be.set_backend("numpy")
    be.set_precision("float64")


@pytest.fixture(params=["float32", "float64"])
def conic_precision(conic_backend, request):
    be.set_precision(request.param)
    return request.param


def _rays(x, y, z, L, M, N):
    """Construct genuine float32 rays even on the NumPy backend."""
    values = np.broadcast_arrays(x, y, z, L, M, N)
    rays = RealRays(*values, np.ones_like(values[0]), 0.55)
    for key, value in zip(("x", "y", "z", "L", "M", "N"), values, strict=True):
        setattr(rays, key, be.array(np.atleast_1d(value)))
    return rays


def _scalar(value):
    return float(np.asarray(be.to_numpy(value)).reshape(-1)[0])


def test_near_forward_crossing_is_not_a_self_intersection(conic_precision):
    geometry = StandardGeometry(CoordinateSystem(), -32.0, -1.0)
    rays = _rays(0.0, 32.0 + 2**-13, -16.0, 0.0, -0.8, -0.6)
    y, z, m, n = (_scalar(v) for v in (rays.y, rays.z, rays.M, rays.N))
    # Independent, higher-precision quadratic for the actual stored inputs.
    a = m * m
    b = 2 * m * y + 64 * n
    c = y * y + 64 * z
    expected = c / (-0.5 * (b - np.sqrt(b * b - 4 * a * c)))
    actual = _scalar(geometry.distance(rays))
    assert 0 < actual < 0.001  # The competing root is about 140 mm away.
    np.testing.assert_allclose(actual, expected, rtol=2e-5)


@pytest.mark.parametrize("scale", [0.125, 1.0, 16.0])
def test_positive_small_discriminant_preserves_microlens_hit(conic_precision, scale):
    geometry = StandardGeometry(CoordinateSystem(), 0.01 * scale)
    rays = _rays(0.009999 * scale, 0.0, 0.009 * scale, 0.0, 0.0, 1.0)
    radius, x, z = (_scalar(v) for v in (geometry.radius, rays.x, rays.z))
    expected = radius - np.sqrt((radius - x) * (radius + x)) - z
    # The chosen ray is near tangent; float32 coefficient rounding is amplified.
    tolerance = 1e-4 if conic_precision == "float32" else 1e-10
    np.testing.assert_allclose(
        _scalar(geometry.distance(rays)), expected, rtol=tolerance
    )


@pytest.mark.parametrize("radius", [1e-12, 1e-9, 1e-3, 1.0, 1e3])
def test_small_nonzero_q_is_not_a_degenerate_equation(conic_precision, radius):
    geometry = StandardGeometry(CoordinateSystem(), radius)
    rays = _rays(0.0, 0.0, -radius, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(
        _scalar(geometry.distance(rays)), -_scalar(rays.z), rtol=5e-7
    )


def test_small_nonzero_quadratic_coefficient_retains_second_crossing(conic_precision):
    geometry = StandardGeometry(CoordinateSystem(), -1.0, -1.0)
    rays = _rays(0.0, 0.0, 0.0, 0.0, 1e-5, -np.sqrt(1 - 1e-10))
    expected = -2 * _scalar(rays.N) / _scalar(rays.M) ** 2
    np.testing.assert_allclose(_scalar(geometry.distance(rays)), expected, rtol=5e-7)


def test_exact_tangent_keeps_the_double_root(conic_precision):
    geometry = StandardGeometry(CoordinateSystem(), 1.0)
    rays = _rays(1.0, 0.0, -1.0, 0.0, 0.0, 1.0)
    assert _scalar(geometry.distance(rays)) == 2.0


def test_stationary_ray_has_no_unique_intersection(conic_precision):
    geometry = StandardGeometry(CoordinateSystem(), 1.0)
    rays = _rays(0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
    assert np.isnan(_scalar(geometry.distance(rays)))


def test_exact_self_crossing_retains_other_parabola_root(conic_precision):
    geometry = StandardGeometry(CoordinateSystem(), -25.4, -1.0)
    rays = _rays(0.0, 25.4, -12.7, 0.0, -1.0, 0.0)
    np.testing.assert_allclose(_scalar(geometry.distance(rays)), 50.8, rtol=1e-7)


@pytest.mark.parametrize("radius", [-25.4, 25.4])
@pytest.mark.parametrize("conic", [-2.0, -1.0, 0.0, 0.5])
@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
def test_rounded_sag_origin_does_not_create_a_self_hit(
    conic_precision, radius, conic, scale
):
    geometry = StandardGeometry(CoordinateSystem(), radius * scale, conic)
    rays = _rays(0, np.linspace(2, 20, 101) * scale, 0, 0, -1, 0)
    rays.z = geometry.sag(rays.x, rays.y)
    # A horizontal chord starting on the sag must cross at the opposite y.
    # Rounded sag values can leave a tiny nonzero implicit residual.
    actual = be.to_numpy(geometry.distance(rays))
    expected = 2 * be.to_numpy(rays.y)
    np.testing.assert_allclose(actual, expected, rtol=5e-7)


@pytest.mark.parametrize("scale", [0.125, 1.0, 16.0])
def test_near_tangent_crossing_is_not_a_rounded_self_hit(conic_precision, scale):
    from decimal import Decimal, localcontext

    # The origin is one small representable step outside a parabola. A nearly
    # tangent direction makes the first crossing much farther than roundoff
    # in position, even though the implicit residual is tiny.
    delta = 2**-24 if conic_precision == "float32" else 2**-53
    offset = 0.002 if conic_precision == "float32" else 1e-7
    direction = np.array([-np.sqrt(0.5), np.sqrt(0.5) - offset])
    direction /= np.linalg.norm(direction)
    geometry = StandardGeometry(CoordinateSystem(), -scale, -1.0)
    rays = _rays(0, scale, (-0.5 + delta) * scale, 0, *direction)
    with localcontext() as context:
        context.prec = 60
        y, z, m, n, radius = (
            Decimal.from_float(_scalar(value))
            for value in (rays.y, rays.z, rays.M, rays.N, geometry.radius)
        )
        a = m * m
        b = 2 * (m * y - n * radius)
        c = y * y - 2 * radius * z
        expected = float(c / (-Decimal("0.5") * (b - (b * b - 4 * a * c).sqrt())))
    np.testing.assert_allclose(
        _scalar(geometry.distance(rays)), expected, rtol=5e-7
    )


@pytest.mark.parametrize("radius", [-12.0, 12.0])
@pytest.mark.parametrize("conic", [-2.0, -1.0, -0.9999, 0.0, 0.5])
def test_physical_chords_preserve_forward_aperture_and_virtual_hits(
    conic_precision, radius, conic
):
    from decimal import Decimal, localcontext
    from optiland.physical_apertures import OffsetRadialAperture

    def point(x, y):
        with localcontext() as context:
            context.prec = 60
            dx, dy, r, k = map(Decimal.from_float, (x, y, radius, conic))
            r2 = dx * dx + dy * dy
            z = r2 / (r * (1 + (1 - (1 + k) * r2 / (r * r)).sqrt()))
        return np.array([x, y, float(z)])

    first = point(0.1 * abs(radius), 0.2 * abs(radius))
    second = point(0.3 * abs(radius), -0.55 * abs(radius))
    chord = second - first
    length = np.linalg.norm(chord)
    direction = chord / length
    fractions = np.array([-1.0, -0.25, 0.0, 0.25, 1.25, 2.0])
    positions = first[:, None] + chord[:, None] * fractions
    geometry = StandardGeometry(CoordinateSystem(), radius, conic)
    rays = _rays(*positions, *direction)
    # Before the first point choose it; between the two choose the second;
    # past both, retain the signed vertex-nearest (first point) fallback.
    expected = np.where(
        (fractions >= 0) & (fractions < 1), 1 - fractions, -fractions
    ) * length
    tolerance = 3e-6 if conic_precision == "float32" else 3e-13
    np.testing.assert_allclose(
        be.to_numpy(geometry.distance(rays)), expected, rtol=tolerance
    )
    aperture = OffsetRadialAperture(
        abs(radius) * 0.05, offset_x=second[0], offset_y=second[1]
    )
    expected = np.where(fractions < 1, 1 - fractions, -fractions) * length
    np.testing.assert_allclose(
        be.to_numpy(geometry.distance(rays, aperture)), expected, rtol=tolerance
    )


def test_small_hit_has_correct_coordinate_and_radius_gradients(conic_precision):
    if be.get_backend() != "torch":
        pytest.skip("Torch autograd contract")
    import torch

    x = be.array([0.006]).requires_grad_()
    z = be.array([-0.003]).requires_grad_()
    radius = be.array(0.01).requires_grad_()
    geometry = StandardGeometry(CoordinateSystem(), 0.01)
    geometry.radius = radius
    rays = SimpleNamespace(x=x, y=x * 0, z=z, L=x * 0, M=x * 0, N=x * 0 + 1)
    distance = geometry.distance(rays)
    gx, gz, gr = torch.autograd.grad(distance.sum(), (x, z, radius))
    root = np.sqrt(_scalar(radius) ** 2 - _scalar(x) ** 2)
    np.testing.assert_allclose(_scalar(gx), _scalar(x) / root, rtol=2e-6)
    np.testing.assert_allclose(_scalar(gz), -1.0, rtol=2e-6)
    np.testing.assert_allclose(_scalar(gr), 1 - _scalar(radius) / root, rtol=2e-6)
    assert distance.device == x.device


def test_regular_hit_gradients_survive_mixed_miss_and_tangent(conic_precision):
    if be.get_backend() != "torch":
        pytest.skip("Torch autograd contract")
    import torch

    x = be.array([0.0, 2.0, 1.0]).requires_grad_()
    z = be.array([-1.0, -1.0, -1.0]).requires_grad_()
    radius = be.array(1.0).requires_grad_()
    geometry = StandardGeometry(CoordinateSystem(), 1.0)
    geometry.radius = radius
    rays = SimpleNamespace(x=x, y=x * 0, z=z, L=x * 0, M=x * 0, N=x * 0 + 1)
    distance = geometry.distance(rays)
    assert torch.isnan(distance[1])
    assert _scalar(distance[2]) == 2.0
    gradients = torch.autograd.grad(distance[0], (x, z, radius))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
    np.testing.assert_allclose(be.to_numpy(gradients[1]), [-1, 0, 0], atol=1e-7)


def test_exact_tangent_is_detached_from_singular_derivative(conic_precision):
    if be.get_backend() != "torch":
        pytest.skip("Torch autograd contract")
    import torch

    x = be.array([1.0]).requires_grad_()
    z = be.array([-1.0]).requires_grad_()
    radius = be.array(1.0).requires_grad_()
    geometry = StandardGeometry(CoordinateSystem(), 1.0)
    geometry.radius = radius
    rays = SimpleNamespace(x=x, y=x * 0, z=z, L=x * 0, M=x * 0, N=x * 0 + 1)
    distance = geometry.distance(rays)
    assert _scalar(distance) == 2.0
    for gradient in torch.autograd.grad(distance.sum(), (x, z, radius)):
        assert torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("aperture_swap", [False, True])
def test_first_and_second_derivatives_for_all_ray_and_surface_parameters(
    conic_backend, aperture_swap
):
    if be.get_backend() != "torch":
        pytest.skip("Torch autograd contract")
    import torch
    from optiland.physical_apertures import OffsetRadialAperture

    be.set_precision("float64")
    if aperture_swap:
        values = [0.0, 30.0, -12.7, 0.0, -1.0, 0.0, -25.4, -1.0]
        aperture = OffsetRadialAperture(12.7, offset_y=-25.4)
    else:
        values = [1.0, 2.0, -3.0, 0.02, 0.01, 0.99975, -12.0, 0.5]
        aperture = None
    leaves = tuple(be.array([value]).requires_grad_() for value in values)
    geometry = StandardGeometry(CoordinateSystem(), values[6], values[7])

    def solve(*inputs):
        geometry.radius, geometry.k = inputs[-2:]
        rays = SimpleNamespace(
            **dict(zip(("x", "y", "z", "L", "M", "N"), inputs[:6], strict=True))
        )
        return geometry.distance(rays, aperture=aperture)

    assert torch.autograd.gradcheck(solve, leaves)
    assert torch.autograd.gradgradcheck(solve, leaves)


def test_tensor_dtype_is_preserved_when_backend_default_changes(conic_backend):
    if be.get_backend() != "torch":
        pytest.skip("Torch tensor dtype contract")
    import torch

    be.set_precision("float64")
    geometry = StandardGeometry(CoordinateSystem(), -12.0, 0.5)
    rays = _rays(1, 2, -3, 0, 0, 1)
    expected = geometry.distance(rays)
    be.set_precision("float32")
    actual = geometry.distance(rays)
    assert actual.dtype == torch.float64
    assert actual.device == rays.x.device
    assert torch.equal(actual, expected)


def test_oap_systems_keep_physical_rays_on_each_device(conic_precision):
    from .test_conic_root_selection import (
        _build_double_oap_relay,
        _build_oap_collimator,
    )

    collimator = _build_oap_collimator()
    rays = collimator.trace(
        Hx=0, Hy=0, wavelength=0.633, num_rays=21, distribution="line_y"
    )
    assert np.count_nonzero(be.to_numpy(rays.i) > 0) == 21
    assert (be.to_numpy(rays.M) > 0).all()
    tolerance = 5e-5 if conic_precision == "float32" else 1e-13
    np.testing.assert_allclose(be.to_numpy(rays.y), 40, rtol=tolerance)

    relay = _build_double_oap_relay(0.0)
    rays = relay.trace(
        Hx=0, Hy=0, wavelength=0.4861, num_rays=41, distribution="line_y"
    )
    assert np.count_nonzero(be.to_numpy(rays.i) > 0) == 41
    for component in (rays.L, rays.M, rays.N):
        values = be.to_numpy(component)
        assert np.ptp(values) < tolerance
