"""Regressions for chief-referenced OPD with a very distant exit pupil (#984)."""

from __future__ import annotations

from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

import optiland.backend as be
from optiland.distribution import create_distribution
from optiland.materials import Material
from optiland.optic import Optic
from optiland.wavefront import OPD
from optiland.wavefront.reference_geometry import ReferenceGeometry, SphericalReference
from optiland.wavefront.strategy import ChiefRayStrategy, create_strategy
from tests.utils import assert_allclose


def _relay(shift):
    radius, thickness, wavelength = 51.5, 4.0, 0.55
    index = Material("N-BK7").n(wavelength).item()
    focal_length = radius / (index - 1)
    front_focal_distance = focal_length - thickness / index
    optic = Optic()
    surfaces = optic.surfaces
    surfaces.add(index=0, radius=np.inf, thickness=front_focal_distance)
    surfaces.add(index=1, radius=np.inf, thickness=thickness, material="N-BK7")
    surfaces.add(index=2, radius=-radius, thickness=focal_length + shift)
    surfaces.add(index=3, radius=np.inf, thickness=focal_length - shift, is_stop=True)
    surfaces.add(index=4, radius=radius, thickness=thickness, material="N-BK7")
    surfaces.add(index=5, radius=np.inf, thickness=front_focal_distance)
    surfaces.add(index=6)
    optic.set_aperture("objectNA", 0.05)
    optic.fields.set_type("object_height")
    optic.fields.add(y=0)
    optic.fields.add(y=5)
    optic.wavelengths.add(wavelength, is_primary=True)
    optic.obj_space_telecentric = shift == 0
    optic.updater.update()
    return optic


def test_telecentric_relay_is_continuous(set_test_backend):
    """The issue's 4f relay must not jump to hundreds of thousands of waves."""
    results = []
    for shift in (0.1, 0.001, 0.0, -0.001, -0.1):
        optic = _relay(shift)
        wavefront = OPD(optic, (0, 0), 0.55, num_rays=32)
        data = wavefront.get_data((0, 0), 0.55)
        assert be.all(be.isfinite(data.opd))
        assert be.all(data.intensity > 0)
        assert_allclose(data.opd[0], 0, atol=1e-10)
        results.append(wavefront.rms())
    # Independent acceptance value from the issue's finite-pupil limit.
    assert_allclose(results[2], 0.28985, atol=5e-6, rtol=0)
    assert_allclose(be.stack(results), be.ones(5) * results[2], atol=3e-6, rtol=0)


@pytest.mark.parametrize("field", [(0, 0), (0, 1)])
def test_telecentric_wavefront_matches_high_precision_paths(set_test_backend, field):
    """Compare every OPD sample with an 80-digit sphere-intersection oracle."""
    optic = _relay(0)
    wavefront = OPD(optic, field, 0.55, num_rays=3)
    data = wavefront.get_data(field, 0.55)
    chief = optic.trace_generic(*field, Px=0, Py=0, wavelength=0.55)
    rays = optic.trace(*field, 0.55, None, wavefront.distribution)
    center = [float(getattr(chief, axis).item()) for axis in "xyz"]
    positions = np.column_stack([be.to_numpy(getattr(rays, axis)) for axis in "xyz"])
    directions = np.column_stack([be.to_numpy(getattr(rays, axis)) for axis in "LMN"])
    ray_opds = be.to_numpy(rays.opd)
    expected = []
    with localcontext() as ctx:
        ctx.prec = 80
        radius = Decimal(data.radius)
        chief_opd = Decimal(float(chief.opd.item()))
        n_image = Decimal(float(optic.surfaces.n(0.55)[-1].item()))
        for position, direction, opd in zip(
            positions, directions, ray_opds, strict=True
        ):
            v = [
                Decimal(float(p)) - Decimal(c)
                for p, c in zip(position, center, strict=True)
            ]
            u = [Decimal(float(d)) for d in direction]
            norm = sum(d * d for d in u).sqrt()
            u = [d / norm for d in u]
            h = sum(d * p for d, p in zip(u, v, strict=True))
            q = sum(p * p for p in v) - h * h
            distance = h + (radius * radius - q).sqrt()
            difference = chief_opd - Decimal(float(opd)) + n_image * (distance - radius)
            expected.append(float(difference / Decimal("0.00055")))
    assert_allclose(data.opd, be.array(expected), atol=3e-9, rtol=1e-9)


@pytest.mark.parametrize("radius", [10.0, -10.0, 1e6, -1e6, 1e18, -1e18, 1e100, -1e100])
def test_sphere_offset_retains_small_paths(set_test_backend, radius):
    """An oblique unit ray has an analytically known longitudinal projection."""
    rays = SimpleNamespace(
        x=be.array([0.001]),
        y=be.array([0.002]),
        z=be.array([0.0]),
        L=be.array([0.6]),
        M=be.array([0.8]),
        N=be.array([0.0]),
    )
    sphere = SphericalReference((0, 0, 0), radius)
    with localcontext() as ctx:
        ctx.prec = 220
        r = Decimal(abs(radius))
        # h = .0022 and squared transverse displacement = .00000016.
        expected = Decimal("1.5") * (
            Decimal("0.0022") + (r * r - Decimal("0.00000016")).sqrt() - r
        )
    assert_allclose(
        sphere._relative_path_length(rays, 1.5), float(expected), atol=1e-16, rtol=1e-13
    )


@pytest.mark.parametrize("radius", [0.0, 1.0, -1.0, 10.0, -10.0])
def test_offset_preserves_intersection_root_convention(set_test_backend, radius):
    """Cover inside/outside, forward/backward, tangent and missed intersections."""
    rays = SimpleNamespace(
        x=be.array([0.0, 0.0, 0.0, 1.0, 2.0]),
        y=be.zeros(5),
        z=be.array([0.0, 20.0, -20.0, 0.0, 0.0]),
        L=be.zeros(5),
        M=be.zeros(5),
        N=be.ones(5),
    )
    sphere = SphericalReference((0, 0, 0), radius)
    expected = sphere.path_length(rays, 1.5) - 1.5 * abs(radius)
    assert_allclose(sphere._relative_path_length(rays, 1.5), expected, atol=1e-12)


@pytest.mark.parametrize("radius", [0.0, 5.0, -5.0])
def test_sphere_nonunit_direction_contract(set_test_backend, radius):
    """Relative paths use distance; absolute paths retain the ray parameter."""
    scales = be.array([0.25, 2.0, 3.0, 0.5, 4.0, 1.25])
    rays = SimpleNamespace(
        x=be.array([0.0, 1.2, 6.0, -6.0, 5.0, 6.0]),
        y=be.zeros(6),
        z=be.array([0.0, 1.6, 8.0, -8.0, 0.0, 2.0]),
        L=be.array([0.6, 0.6, 0.6, 0.6, 0.0, 0.0]) * scales,
        M=be.zeros(6),
        N=be.array([0.8, 0.8, 0.8, 0.8, 1.0, 1.0]) * scales,
    )
    original_directions = [be.copy(getattr(rays, axis)) for axis in "LMN"]
    sphere = SphericalReference((0, 0, 0), radius)
    # The first four rays lie on a diameter. The last two are tangent/missed
    # intersections at R=5; the existing discriminant clamp projects a miss.
    distances = be.array(
        [0.0, 2.0, 10.0, -10.0, 0.0, 2.0]
        if radius == 0
        else [5.0, 7.0, 5.0, -5.0, 0.0, 2.0]
    )
    assert_allclose(sphere.path_length(rays, 1.5), 1.5 * distances / scales, atol=1e-12)
    assert_allclose(
        sphere._relative_path_length(rays, 1.5),
        1.5 * (distances - abs(radius)),
        atol=1e-12,
    )
    assert sphere.radius == radius
    for axis, original in zip("LMN", original_directions, strict=True):
        assert_allclose(getattr(rays, axis), original, atol=0, rtol=0)


@pytest.mark.parametrize("radius", [1e18, -1e18])
def test_sphere_offset_normalizes_roundoff_in_directions(set_test_backend, radius):
    """A tiny norm error must not become R times that error in the OPD."""
    rays = SimpleNamespace(
        x=be.array([0.0]),
        y=be.array([0.0]),
        z=be.array([0.001]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1 + 2e-16]),
    )
    sphere = SphericalReference((0, 0, 0), radius)
    assert_allclose(sphere._relative_path_length(rays, 1.0), 0.001, atol=1e-16)


def test_sphere_offset_is_translation_invariant(set_test_backend):
    rays = SimpleNamespace(
        x=be.array([0.125]),
        y=be.array([-0.25]),
        z=be.array([0.5]),
        L=be.array([0.6]),
        M=be.array([0.8]),
        N=be.array([0.0]),
    )
    expected = SphericalReference((0, 0, 0), 1e18)._relative_path_length(rays, 1.0)
    center = (1e6, -1e6, 2e6)
    for axis, offset in zip("xyz", center, strict=True):
        setattr(rays, axis, getattr(rays, axis) + offset)
    actual = SphericalReference(center, 1e18)._relative_path_length(rays, 1.0)
    assert_allclose(actual, expected, atol=1e-15)


def test_sphere_offset_torch_gradient():
    torch = pytest.importorskip("torch")
    be.set_backend("torch")
    try:
        position = torch.tensor(
            [0.001, 0.002, 0.0], dtype=torch.float64, requires_grad=True
        )
        rays = SimpleNamespace(
            x=position[0:1],
            y=position[1:2],
            z=position[2:3],
            L=torch.tensor([0.6], dtype=torch.float64),
            M=torch.tensor([0.8], dtype=torch.float64),
            N=torch.tensor([0.0], dtype=torch.float64),
        )
        SphericalReference((0, 0, 0), 1e18)._relative_path_length(
            rays, 1.5
        ).sum().backward()
        torch.testing.assert_close(
            position.grad,
            torch.tensor([0.9, 1.2, 0.0], dtype=torch.float64),
            atol=1e-14,
            rtol=1e-14,
        )
    finally:
        be.set_backend("numpy")


class _DelegatedReference(ReferenceGeometry):
    """A reference with spherical mathematics but no spherical type identity."""

    def __init__(self, sphere):
        self.sphere = sphere

    def path_length(self, rays, n_medium):
        return self.sphere.path_length(rays, n_medium)

    def _relative_path_length(self, rays, n_medium):
        return self.sphere._relative_path_length(rays, n_medium)

    @property
    def radius(self):
        return self.sphere.radius


@pytest.mark.parametrize("strategy_name", ["chief_ray", "centroid", "best_fit"])
def test_strategies_use_relative_geometry_contract(
    set_test_backend, monkeypatch, strategy_name
):
    """All consumers must preserve wavefront variation with a huge common path."""
    optic = _relay(0)
    distribution = create_distribution("hexapolar")
    distribution.generate_points(3)
    chief_strategy = ChiefRayStrategy(optic, distribution)
    chief = optic.trace_generic(0, 1, Px=0, Py=0, wavelength=0.55)
    reference = _DelegatedReference(chief_strategy._create_reference_geometry(chief))
    baseline = chief_strategy.compute_wavefront_data((0, 1), 0.55)

    strategy = create_strategy(strategy_name, optic, distribution)
    monkeypatch.setattr(strategy, "_create_reference_geometry", lambda rays: reference)
    actual = strategy.compute_wavefront_data((0, 1), 0.55)
    expected = baseline.opd
    if strategy_name != "chief_ray":
        expected = expected - be.mean(expected[baseline.intensity > 0])
    assert be.std(expected) > 0.1
    assert_allclose(actual.opd, expected, atol=1e-9, rtol=1e-9)
    for axis in ("pupil_x", "pupil_y", "pupil_z"):
        assert_allclose(getattr(actual, axis), getattr(baseline, axis), rtol=1e-14)
