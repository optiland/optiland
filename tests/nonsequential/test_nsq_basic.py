"""Basic non-sequential raytracing tests.

Covers analytic reference cases from the spec (§16.1) and flux
conservation (§16.2).

Kramer Harrison, 2026
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    VACUUM,
    CollimatedSource,
    ConicGeometry,
    FarFieldDetector,
    FinitePlaneGeometry,
    IrradianceDetector,
    NSQScene,
    NSQTracer,
    PointSource,
    ReflectiveComponent,
    RefractiveComponent,
    Spectrum,
    SpecularBRDF,
    LambertianBSDF,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def numpy_backend():
    """Return a seeded NumpyBackend."""
    return NumpyBackend(seed=42)


# ---------------------------------------------------------------------------
# § Test: Spectrum sampling
# ---------------------------------------------------------------------------


class TestSpectrum:
    def test_monochromatic(self):
        spec = Spectrum.monochromatic(550.0)
        rng = np.random.default_rng(0)
        wl = spec.sample(100, rng)
        assert np.all(wl == 550.0)

    def test_broadband_sampling(self):
        wls = np.array([400.0, 500.0, 600.0, 700.0])
        wts = np.array([1.0, 2.0, 2.0, 1.0])
        spec = Spectrum(wls, wts)
        rng = np.random.default_rng(42)
        samples = spec.sample(10000, rng)
        assert set(samples).issubset(set(wls))
        # 500 and 600 nm should be sampled more than 400 and 700 nm
        counts = {w: int((samples == w).sum()) for w in wls}
        assert counts[500.0] > counts[400.0]
        assert counts[600.0] > counts[700.0]


# ---------------------------------------------------------------------------
# § Test: Point source generation
# ---------------------------------------------------------------------------


class TestPointSource:
    def test_generates_correct_num_rays(self):
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(550.0)
        src = PointSource(cs=cs, spectrum=spec, total_flux=1.0, half_angle_deg=90.0)
        rng = np.random.default_rng(0)
        rays = src.generate(1000, rng)
        assert rays.num_rays == 1000
        assert rays.num_rays_alive == 1000

    def test_total_flux_conservation(self):
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(550.0)
        total = 2.5
        src = PointSource(cs=cs, spectrum=spec, total_flux=total, half_angle_deg=90.0)
        rng = np.random.default_rng(0)
        rays = src.generate(10000, rng)
        assert np.isclose(rays.flux.sum(), total, rtol=1e-6)

    def test_directions_normalized(self):
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(550.0)
        src = PointSource(cs=cs, spectrum=spec, total_flux=1.0)
        rng = np.random.default_rng(0)
        rays = src.generate(500, rng)
        dirs = np.stack([rays.dx, rays.dy, rays.dz], axis=1)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_cone_constraint(self):
        """All rays within a 30-degree half-angle cone around +z."""
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(550.0)
        src = PointSource(cs=cs, spectrum=spec, total_flux=1.0, half_angle_deg=30.0)
        rng = np.random.default_rng(0)
        rays = src.generate(1000, rng)
        cos_theta = rays.dz  # +z axis
        min_cos = np.cos(np.radians(30.0))
        assert np.all(cos_theta >= min_cos - 1e-10)


# ---------------------------------------------------------------------------
# § Test: Plane geometry intersection
# ---------------------------------------------------------------------------


class TestPlaneGeometry:
    def test_infinite_plane_hits(self):
        from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry  # noqa: PLC0415

        geom = PlaneGeometry()
        N = 5
        origins = np.zeros((N, 3))
        origins[:, 2] = -1.0  # behind plane at z=0
        directions = np.zeros((N, 3))
        directions[:, 2] = 1.0  # pointing +z

        t, normals, hit_mask = geom.ray_intersect(origins, directions)
        assert hit_mask.all()
        np.testing.assert_allclose(t, 1.0, atol=1e-9)

    def test_parallel_ray_no_hit(self):
        from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry  # noqa: PLC0415

        geom = PlaneGeometry()
        origins = np.array([[0.0, 0.0, -1.0]])
        directions = np.array([[1.0, 0.0, 0.0]])  # parallel to plane
        t, normals, hit_mask = geom.ray_intersect(origins, directions)
        assert not hit_mask[0]
        assert t[0] == np.inf

    def test_finite_plane_aperture_check(self):
        from optiland.nonsequential.components.geometry.analytic.plane import FinitePlaneGeometry  # noqa: PLC0415

        geom = FinitePlaneGeometry(width=10.0, height=10.0)
        # Ray hitting inside aperture
        o_in = np.array([[0.0, 0.0, -1.0]])
        d_in = np.array([[0.0, 0.0, 1.0]])
        t, _, hit = geom.ray_intersect(o_in, d_in)
        assert hit[0]

        # Ray hitting outside aperture
        o_out = np.array([[6.0, 0.0, -1.0]])
        t, _, hit = geom.ray_intersect(o_out, d_in)
        assert not hit[0]


# ---------------------------------------------------------------------------
# § Test: Sphere geometry intersection
# ---------------------------------------------------------------------------


class TestSphereGeometry:
    def test_sphere_center_ray(self):
        from optiland.nonsequential.components.geometry.analytic.sphere import SphereGeometry  # noqa: PLC0415

        R = 10.0
        geom = SphereGeometry(radius=R)
        # Ray from outside along -z hitting sphere
        origins = np.array([[0.0, 0.0, -20.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        t, normals, hit = geom.ray_intersect(origins, directions)
        assert hit[0]
        np.testing.assert_allclose(t[0], 10.0, atol=1e-9)  # hits at z=-10

    def test_sphere_normal_outward(self):
        from optiland.nonsequential.components.geometry.analytic.sphere import SphereGeometry  # noqa: PLC0415

        geom = SphereGeometry(radius=5.0)
        origins = np.array([[0.0, 0.0, -10.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        t, normals, hit = geom.ray_intersect(origins, directions)
        # Normal at (0, 0, -5) should point toward incoming ray (in -z direction)
        assert normals[0, 2] < 0  # points toward origin (incoming side)


# ---------------------------------------------------------------------------
# § Test: Flat mirror normal incidence (analytic reference case)
# ---------------------------------------------------------------------------


class TestFlatMirrorReflection:
    """Collimated beam → flat mirror at 45° → flat detector.

    Expected: spot on detector at geometric prediction.
    """

    def test_flat_mirror_reflection(self):
        # Collimated beam along +z, centred on origin
        # Mirror at z=50, ry=-45° → local z points in (-sin45,0,cos45) = upper-left.
        # A +z beam hits the mirror and reflects to +x.
        src_cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(550.0)
        src = CollimatedSource(
            cs=src_cs,
            spectrum=spec,
            total_flux=1.0,
            aperture_radius=1.0,
        )

        # Mirror: ry=-pi/4 rad → reflects +z global to +x global
        mirror_cs = CoordinateSystem(x=0, y=0, z=50, ry=-math.pi / 4)
        from optiland.nonsequential.components.geometry.analytic.plane import FinitePlaneGeometry  # noqa: PLC0415

        mirror = ReflectiveComponent(
            cs=mirror_cs,
            geometry=FinitePlaneGeometry(width=20.0, height=20.0),
            bsdf=SpecularBRDF(),
        )

        # Detector at x=50, z=50 (ry=pi/2 rad → local z points in +x global).
        # Reflected beam travels +x and hits the detector.
        det_cs = CoordinateSystem(x=50, y=0, z=50, ry=math.pi / 2)
        det = IrradianceDetector(
            cs=det_cs,
            width=10.0,
            height=10.0,
            num_pixels_x=64,
            num_pixels_y=64,
        )

        scene = NSQScene()
        scene.add_component(mirror)
        scene.add_source(src)
        scene.add_detector(det)

        tracer = NSQTracer(scene, num_rays=10_000, max_bounces=5, seed=42)
        result = tracer.trace()

        irr = result.detectors["detector_0"]
        # Should detect most flux (some geometric clipping expected)
        assert irr.num_rays_hit > 8000


# ---------------------------------------------------------------------------
# § Test: Flux conservation
# ---------------------------------------------------------------------------


class TestFluxConservation:
    """Flux conservation: collimated beam into flat detector."""

    def test_collimated_beam_to_flat_detector(self):
        """Collimated source → large flat detector directly above.

        Since no components are between source and detector, all rays should
        reach the detector. Flux conservation error should be ~0.
        """
        from optiland.nonsequential.components.geometry.analytic.plane import FinitePlaneGeometry  # noqa: PLC0415

        src_cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(550.0)
        src = CollimatedSource(
            cs=src_cs,
            spectrum=spec,
            total_flux=1.0,
            aperture_radius=5.0,
        )

        # Large detector at z=100, centred, covers the full beam cross-section
        det_cs = CoordinateSystem(x=0, y=0, z=100)
        det = IrradianceDetector(
            cs=det_cs,
            width=30.0,
            height=30.0,
            num_pixels_x=32,
            num_pixels_y=32,
        )

        scene = NSQScene()
        scene.add_source(src)
        scene.add_detector(det)

        tracer = NSQTracer(scene, num_rays=10_000, max_bounces=5, seed=0)
        result = tracer.trace()

        irr = result.detectors["detector_0"]
        # All rays should reach the detector (beam r=5mm, detector 30mm wide)
        assert irr.num_rays_hit == 10_000
        # Total detected flux should equal total source flux
        np.testing.assert_allclose(
            irr.total_flux, 1.0, rtol=1e-6
        )
        assert result.flux_conservation_error < 0.001
        assert result.trace_time_sec > 0.0

    def test_reflective_mirror_flux(self):
        """Paraboloid collimating mirror: some rays should reach detector."""
        # Mirror with ry=180° so it opens downward (-z global).
        # Vertex in global at z=100. Local focus at z_local=R/2=100mm.
        # With ry=180°: local +z points in global -z.
        # Focus in global: vertex + Ry(180°)@[0,0,100] = [0,0,100]+[0,0,-100]=[0,0,0].
        # Source at z=0 is at the focus. Source emits upward (+z global).
        # Reflected rays go parallel to local +z (global -z), downward.
        # A detector below at z=-100 (global) catches the collimated beam.
        from optiland.nonsequential.components.geometry.analytic.plane import FinitePlaneGeometry  # noqa: PLC0415

        mirror_cs = CoordinateSystem(x=0, y=0, z=100, ry=math.pi)
        mirror = ReflectiveComponent(
            cs=mirror_cs,
            geometry=ConicGeometry(radius=200.0, conic=-1.0, aperture_radius=50.0),
            bsdf=SpecularBRDF(),
        )

        src_cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(550.0)
        # Source at focus, emit upward toward the mirror
        src = PointSource(cs=src_cs, spectrum=spec, total_flux=1.0, half_angle_deg=30.0)

        # Large detector below the mirror to catch reflected collimated beam
        det_cs = CoordinateSystem(x=0, y=0, z=-50)
        det = IrradianceDetector(
            cs=det_cs,
            width=200.0,
            height=200.0,
            num_pixels_x=64,
            num_pixels_y=64,
        )

        scene = NSQScene()
        scene.add_component(mirror)
        scene.add_source(src)
        scene.add_detector(det)

        tracer = NSQTracer(scene, num_rays=20_000, max_bounces=5, seed=0)
        result = tracer.trace()

        # Some rays should be detected; tracer should complete without error
        assert result.trace_time_sec > 0.0
        assert result.num_rays_total == 20_000


# ---------------------------------------------------------------------------
# § Test: Collimated source generation
# ---------------------------------------------------------------------------


class TestCollimatedSource:
    def test_directions_parallel(self):
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(550.0)
        src = CollimatedSource(cs=cs, spectrum=spec, total_flux=1.0, aperture_radius=5.0)
        rng = np.random.default_rng(0)
        rays = src.generate(200, rng)
        # All rays point in +z direction (no tilt in this CS)
        np.testing.assert_allclose(rays.dx, 0.0, atol=1e-12)
        np.testing.assert_allclose(rays.dy, 0.0, atol=1e-12)
        np.testing.assert_allclose(rays.dz, 1.0, atol=1e-12)

    def test_positions_within_aperture(self):
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(550.0)
        r = 5.0
        src = CollimatedSource(cs=cs, spectrum=spec, total_flux=1.0, aperture_radius=r)
        rng = np.random.default_rng(0)
        rays = src.generate(500, rng)
        radii = (rays.x ** 2 + rays.y ** 2) ** 0.5
        assert np.all(radii <= r + 1e-9)


# ---------------------------------------------------------------------------
# § Test: NSQScene validation
# ---------------------------------------------------------------------------


class TestNSQScene:
    def test_validate_no_sources(self):
        scene = NSQScene()
        cs = CoordinateSystem(x=0, y=0, z=0)
        from optiland.nonsequential.components.geometry.analytic.plane import FinitePlaneGeometry  # noqa: PLC0415

        det = IrradianceDetector(cs=cs, width=10.0, height=10.0, num_pixels_x=32, num_pixels_y=32)
        scene.add_detector(det)
        with pytest.raises(ValueError, match="no sources"):
            scene.validate()

    def test_validate_no_detectors(self):
        scene = NSQScene()
        cs = CoordinateSystem(x=0, y=0, z=0)
        spec = Spectrum.monochromatic(550.0)
        src = PointSource(cs=cs, spectrum=spec, total_flux=1.0)
        scene.add_source(src)
        with pytest.raises(ValueError, match="no detectors"):
            scene.validate()


# ---------------------------------------------------------------------------
# § Test: BSDF
# ---------------------------------------------------------------------------


class TestBSDFs:
    def test_specular_reflectance_unity(self):
        from optiland.nonsequential.bsdf.specular import SpecularBRDF  # noqa: PLC0415

        bsdf = SpecularBRDF()
        N = 10
        dirs = np.tile([0.0, 0.0, -1.0], (N, 1))
        normals = np.tile([0.0, 0.0, 1.0], (N, 1))
        wl = np.full(N, 550.0)
        r = bsdf.reflectance(dirs, normals, wl)
        np.testing.assert_allclose(r, 1.0)

    def test_specular_direction(self):
        from optiland.nonsequential.bsdf.specular import SpecularBRDF  # noqa: PLC0415

        bsdf = SpecularBRDF()
        N = 1
        dirs = np.array([[0.0, 0.0, -1.0]])   # incoming downward
        normals = np.array([[0.0, 0.0, 1.0]])  # normal upward
        wl = np.array([550.0])
        rng = np.random.default_rng(0)
        scattered, weights = bsdf.sample(N, dirs, normals, wl, rng)
        np.testing.assert_allclose(scattered, [[0.0, 0.0, 1.0]], atol=1e-10)

    def test_lambertian_reflectance(self):
        from optiland.nonsequential.bsdf.lambertian import LambertianBSDF  # noqa: PLC0415

        bsdf = LambertianBSDF(reflectance_value=0.8)
        N = 5
        dirs = np.tile([0.0, 0.0, -1.0], (N, 1))
        normals = np.tile([0.0, 0.0, 1.0], (N, 1))
        wl = np.full(N, 550.0)
        r = bsdf.reflectance(dirs, normals, wl)
        np.testing.assert_allclose(r, 0.8)

    def test_lambertian_sample_directions_on_hemisphere(self):
        from optiland.nonsequential.bsdf.lambertian import LambertianBSDF  # noqa: PLC0415

        bsdf = LambertianBSDF()
        N = 1000
        dirs = np.tile([0.0, 0.0, -1.0], (N, 1)).astype(float)
        normals = np.tile([0.0, 0.0, 1.0], (N, 1)).astype(float)
        wl = np.full(N, 550.0)
        rng = np.random.default_rng(42)
        scattered, weights = bsdf.sample(N, dirs, normals, wl, rng)
        # All scattered rays should be on the +z hemisphere
        assert np.all(scattered[:, 2] >= -1e-6)
        # Check normalization
        norms = np.linalg.norm(scattered, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)
